import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
from typing import Optional, Tuple


class FGM(nn.Module):
    """
    Final FGM:
    - Soft band-pass (learnable center + width)
    - Stage-aware frequency bias (no hard high/low)
    - Structure-aware gating
    - Residual output
    """

    def __init__(
        self,
        dim: int,
        layer_idx: int,
        gate_reduction: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.layer_idx = layer_idx

        # -------------------------
        # Stage-aware init
        # -------------------------
        def _init_freq(layer_idx):
            if layer_idx < 2:
                # 退后一步：从0.9降到0.75，避开纯噪声，抓住最锐利的边缘
                # 带宽0.2：覆盖 [0.55 - 0.95]，保留丰富的中高频纹理
                return 0.75, 0.20   
            else:
                # 前进一步：从0.1升到0.25，避开全局亮度偏移(DC分量)
                # 带宽0.25：覆盖 [0.0 - 0.50]，确保宏观结构和低频过渡都被涵盖
                return 0.25, 0.25   


        def inv_sigmoid(x):
            eps = 1e-6
            x = torch.clamp(x, eps, 1 - eps)
            return torch.log(x / (1 - x))

        c_init, w_init = _init_freq(layer_idx)

        self.f_center = nn.Parameter(
            inv_sigmoid(torch.tensor(c_init, dtype=torch.float32))
        )
        self.f_width_raw = nn.Parameter(
            inv_sigmoid(torch.tensor(w_init * 2.0, dtype=torch.float32))
        )

        # spectral shrink
        self.spectral_threshold = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))

        # -------------------------
        # Gate
        # -------------------------
        hidden = max(dim // gate_reduction, 8)
        in_ch = dim * 2 + 1

        self.gate_conv = nn.Sequential(
            nn.Conv3d(in_ch, hidden, kernel_size=1, bias=False),
            nn.InstanceNorm3d(hidden, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.Conv3d(hidden, dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.output_proj = nn.Conv3d(dim, dim, kernel_size=1, bias=True)

        # cache
        self.register_buffer("_freq_grid", torch.empty(0), persistent=False)
        self._cached_shape: Optional[Tuple[int, int, int]] = None
        self._cached_meta: Optional[Tuple[torch.device, torch.dtype]] = None

    # -------------------------
    # Frequency grid
    # -------------------------
    def _get_frequency_grid(self, shape, device, dtype):
        D, H, W = shape
        meta = (device, dtype)

        if (
            self._cached_shape == (D, H, W)
            and self._cached_meta == meta
            and self._freq_grid.numel() > 0
        ):
            return self._freq_grid

        kz = torch.fft.fftfreq(D, device=device, dtype=dtype)
        ky = torch.fft.fftfreq(H, device=device, dtype=dtype)
        kx = torch.fft.rfftfreq(W, device=device, dtype=dtype)

        gz, gy, gx = torch.meshgrid(kz, ky, kx, indexing="ij")
        radius = torch.sqrt(gx * gx + gy * gy + gz * gz)
        radius = radius / (radius.max() + 1e-8)

        self._freq_grid = radius
        self._cached_shape = (D, H, W)
        self._cached_meta = meta

        return self._freq_grid

    # -------------------------
    # Band-pass
    # -------------------------
    def _ordered_band(self):
        c = torch.sigmoid(self.f_center)
        w = 0.5 * torch.sigmoid(self.f_width_raw)

        f_low = (c - w).clamp(0.0, 1.0)
        f_high = (c + w).clamp(0.0, 1.0)

        return f_low, f_high

    # -------------------------
    # Stage-aware bias（关键新增）
    # -------------------------
    def _frequency_bias(self, grid):
        c = torch.sigmoid(self.f_center)

        if self.layer_idx < 2:
            # 浅层：偏高频
            bias = torch.sigmoid((grid - c) * 8.0)
        else:
            # 深层：偏低频
            bias = torch.sigmoid((c - grid) * 8.0)

        return bias

    # -------------------------
    # Structure
    # -------------------------
    def _get_structure(self, x):
        mean = F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)
        mean2 = F.avg_pool3d(x * x, kernel_size=3, stride=1, padding=1)
        var = (mean2 - mean * mean).relu_()
        return var.mean(dim=1, keepdim=True)

    # -------------------------
    # Frequency filtering
    # -------------------------
    def _apply_frequency(self, x):
        _, _, D, H, W = x.shape
        out_dtype = x.dtype

        with torch.autocast(device_type="cuda", enabled=False):
            x32 = x.float()

            X = fft.rfftn(x32, dim=(-3, -2, -1), norm="ortho")

            # spectral shrink
            mag = torch.abs(X)
            thr = F.softplus(self.spectral_threshold).to(mag.dtype)
            scale = (mag - thr).relu_() / (mag + 1e-8)
            X = X * scale

            # band-pass
            grid = self._get_frequency_grid((D, H, W), x32.device, x32.dtype)
            f_low, f_high = self._ordered_band()

            low = torch.sigmoid((f_low - grid) * 10.0)
            high = torch.sigmoid((grid - f_high) * 10.0)
            band_mask = low * high

            # 🔴 关键：stage-aware bias
            bias = self._frequency_bias(grid)

            mask = band_mask * bias

            X = X * mask[None, None, ...]
            x_out = fft.irfftn(X, s=(D, H, W), dim=(-3, -2, -1), norm="ortho")

        return x_out.to(out_dtype)

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, x):
        x_fre = self._apply_frequency(x)
        structure = self._get_structure(x)

        gate = self.gate_conv(torch.cat([x, x_fre, structure], dim=1))

        x_fused = x * gate + x_fre * (1.0 - gate)

        out = self.output_proj(x_fused)

        return x + out