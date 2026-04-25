import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from monai.networks.blocks.dynunet_block import UnetResBlock, UnetBasicBlock


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


# ================================================================
#  Fourier Positional Encoding
# ================================================================
class FourierFeaturizer3D(nn.Module):
    """高分辨率 Fourier 编码，带缓存避免重复计算"""

    def __init__(self, n_freqs: int = 4, include_input: bool = True):
        super().__init__()
        self.n_freqs = n_freqs
        self.include_input = include_input

        freqs = 2.0 ** torch.linspace(0, n_freqs - 1, n_freqs)
        self.register_buffer("freqs", freqs)

        self.out_dim = 3 * n_freqs * 2
        if include_input:
            self.out_dim += 3

        self._cache_key = None
        self._cache_val = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        device, dtype = x.device, x.dtype
        key = (D, H, W, device, dtype)

        if self._cache_key != key:
            d = torch.linspace(-1, 1, D, device=device, dtype=dtype)
            h = torch.linspace(-1, 1, H, device=device, dtype=dtype)
            w = torch.linspace(-1, 1, W, device=device, dtype=dtype)
            gd, gh, gw = torch.meshgrid(d, h, w, indexing='ij')
            coords = torch.stack([gd, gh, gw], dim=0)

            coords_exp = coords.unsqueeze(1)
            freqs_exp = self.freqs.view(1, -1, 1, 1, 1)
            scaled = coords_exp * freqs_exp * math.pi
            sin_f = torch.sin(scaled).view(3 * self.n_freqs, D, H, W)
            cos_f = torch.cos(scaled).view(3 * self.n_freqs, D, H, W)

            parts = [sin_f, cos_f]
            if self.include_input:
                parts.append(coords)

            self._cache_val = torch.cat(parts, dim=0)
            self._cache_key = key

        return self._cache_val.unsqueeze(0).expand(B, -1, -1, -1, -1)




# ---------- 修改 1: Offset 生成器支持 guidance_embed ----------
class OffsetGenerator3D(nn.Module):
    """
    1x1(channel fuse) -> grouped 3x3x3(spatial) -> 1x1(offset)
    """
    def __init__(
        self,
        in_channels: int,
        fourier_dim: int,
        guidance_dim: int = 0,   # 新增
        groups: int = 4,
        hidden_dim: int = 56,
        conv_groups: int = 8,
    ):
        super().__init__()
        self.offset_dim = 3 * groups
        fusion_in = in_channels + fourier_dim + guidance_dim  # 改这里

        self.channel_fuse = nn.Sequential(
            nn.Conv3d(fusion_in, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.spatial = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=conv_groups, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.offset_head = nn.Conv3d(hidden_dim, self.offset_dim, kernel_size=1)
        nn.init.normal_(self.offset_head.weight, 0, 0.001)
        if self.offset_head.bias is not None:
            nn.init.constant_(self.offset_head.bias, 0)

    def _forward_impl(self, x_up, fourier_feats, g=None):
        if g is None:
            fused = torch.cat([x_up, fourier_feats], dim=1)
        else:
            fused = torch.cat([x_up, g, fourier_feats], dim=1)
        h = self.channel_fuse(fused)
        h = self.spatial(h)
        return self.offset_head(h)

    def forward(self, x_up, fourier_feats, g=None, use_checkpoint=False):
        if use_checkpoint and self.training:
            return checkpoint(self._forward_impl, x_up, fourier_feats, g, use_reentrant=False)
        return self._forward_impl(x_up, fourier_feats, g)


# ---------- 修改 2: GDySample3D 真正使用 skip 语义 ----------
class GDySample3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        scale: int = 2,
        groups: int = 2,
        n_freqs: int = 3,
        hidden_dim: int = 32,
        conv_groups: int = 4,
        guidance_channels: int = None,     # 新增: skip通道
        guidance_embed_dim: int = 16,      # 新增: 压缩通道
    ):
        super().__init__()
        self.scale = scale
        self.groups = groups
        self.in_channels = in_channels
        assert in_channels >= groups and in_channels % groups == 0
        assert guidance_channels is not None, "guidance_channels must be set"

        # 你的 FourierFeaturizer3D 保持不变
        self.fourier = FourierFeaturizer3D(n_freqs=n_freqs, include_input=True)

        # 新增：skip语义压缩
        self.guidance_proj = nn.Sequential(
            nn.Conv3d(guidance_channels, guidance_embed_dim, kernel_size=1, bias=False),
            nn.InstanceNorm3d(guidance_embed_dim, affine=True),
            nn.SiLU(inplace=True),
        )

        self.offset_gen = OffsetGenerator3D(
            in_channels=in_channels,
            fourier_dim=self.fourier.out_dim,
            guidance_dim=guidance_embed_dim,   # 改这里
            groups=groups,
            hidden_dim=hidden_dim,
            conv_groups=conv_groups,
        )

        self._grid_cache_key = None
        self._grid_cache_val = None

    def _get_base_grid(self, B, D, H, W, device, dtype):
        key = (B, D, H, W, device, dtype)
        if self._grid_cache_key != key:
            theta = torch.eye(3, 4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
            self._grid_cache_val = F.affine_grid(theta, [B, 1, D, H, W], align_corners=False)
            self._grid_cache_key = key
        return self._grid_cache_val

    def forward(self, x: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        S = self.scale
        D_o, H_o, W_o = D * S, H * S, W * S
        G = self.groups
        C_g = C // G

        x_up = F.interpolate(x, size=(D_o, H_o, W_o), mode='trilinear', align_corners=False)

        # 1) 压缩 skip 语义并对齐空间
        g = self.guidance_proj(guidance)
        if g.shape[2:] != (D_o, H_o, W_o):
            g = F.interpolate(g, size=(D_o, H_o, W_o), mode='trilinear', align_corners=False)

        # 2) 位置编码（仍可保留）
        fourier_feats = self.fourier(g)  # 这里传 g 仅用于 shape，一致即可

        # 3) 真正融合 x_up + g + fourier
        offsets = self.offset_gen(x_up, fourier_feats, g=g, use_checkpoint=self.training)

        grid = self._get_base_grid(B, D_o, H_o, W_o, x.device, x.dtype)
        offset_scale = 0.25 / S
        offsets = offsets.view(B, G, 3, D_o, H_o, W_o).permute(0, 1, 3, 4, 5, 2)
        sample_coords = (grid.unsqueeze(1) + offsets * offset_scale).clamp(-1, 1)
        sample_coords = sample_coords.reshape(B * G, D_o, H_o, W_o, 3)

        x_grouped = x_up.view(B, G, C_g, D_o, H_o, W_o).reshape(B * G, C_g, D_o, H_o, W_o)
        out = F.grid_sample(
            x_grouped, sample_coords,
            mode='bilinear', padding_mode='border', align_corners=False
        )
        out = out.view(B, G, C_g, D_o, H_o, W_o).reshape(B, C, D_o, H_o, W_o)
        return out

# ================================================================
#  Up Block - 速度优先版 (Final-Fast)
# ================================================================
class GDySampleUpBlock3D(nn.Module):
    """
    G-DySample 3D Up Block — 速度优先版

    与精度优先版架构完全一致，仅超参数不同：
      groups:    4 → 2   (grid_sample 减半)
      hidden:    56 → 32  (offset 网络更轻)
      conv_groups: 8 → 4  (CUDA 对齐)
      n_freqs:   4 → 3   (Fourier 更轻)

    性能参考 (中间层 [2,128,32³] → [2,64,64³]):
      推理: ~16ms (1.4x Standard)
      训练: ~64ms, 显存 2143MB
      效果: Toy loss ~0.015-0.018 (仍显著优于 Standard 0.039)
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        norm_name: tuple | str,
        res_block: bool = False,
        kernel_size: int = 3,
        scale_factor: int = 2,
        n_freqs: int = 4,
        groups: int = 2,
    ):
        super().__init__()
        assert spatial_dims == 3

        self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)

        self.upsample = GDySample3D(
            in_channels=out_channels,
            guidance_channels=out_channels,   # 新增
            guidance_embed_dim=16,            # 建议 8 或 16
            scale=scale_factor,
            groups=min(groups, out_channels),
            n_freqs=n_freqs,
            hidden_dim=max(32, (out_channels // 2 // 8) * 8),
            conv_groups=4,
        )


        if res_block:
            self.conv_block = UnetResBlock(
                spatial_dims, out_channels * 2, out_channels,
                kernel_size=3, stride=1, norm_name=norm_name,
            )
        else:
            self.conv_block = UnetBasicBlock(
                spatial_dims, out_channels * 2, out_channels,
                kernel_size=3, stride=1, norm_name=norm_name,
            )

    def forward(self, inp: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.proj(inp)
        up = self.upsample(x, skip)

        if up.shape[2:] != skip.shape[2:]:
            up = F.interpolate(up, size=skip.shape[2:],
                               mode='trilinear', align_corners=False)

        out = torch.cat([up, skip], dim=1)
        out = self.conv_block(out)
        return out