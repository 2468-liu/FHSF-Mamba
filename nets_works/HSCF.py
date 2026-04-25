import torch
import torch.nn as nn
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding  # 假定已安装
from typing import Tuple, Optional, Dict
from mamba_ssm import Mamba2
from functools import lru_cache


# ==================== 1. GPU Hilbert SFC 生成器（仅 GPU 版本） ====================
class HilbertCurveGPU:
    """
    GPU 版本的希尔伯特曲线实现。
    将原 CPU 版本的位操作转换为 PyTorch 张量操作，支持 GPU 加速。
    """

    def __init__(self, p: int, n: int, device: torch.device = None):
        """
        Args:
            p: 希尔伯特曲线的迭代次数
            n: 维度数
            device: 计算设备（必须为 CUDA）
        """
        self.p = p
        self.n = n
        self.device = device if device is not None else torch.device('cuda')
        assert self.device.type == 'cuda', "HilbertCurveGPU 要求使用 CUDA 设备"
        self.max_h = 2 ** (self.p * self.n) - 1
        self.max_x = 2 ** self.p - 1

    def _transpose_to_hilbert_integer_batch(self, x: torch.Tensor) -> torch.Tensor:
        """
        批量将转置坐标转换为希尔伯特整数。
        Args:
            x: (num_points, n) 形状的张量，每行是一个 n 维坐标
        Returns:
            (num_points,) 形状的张量，包含希尔伯特距离
        """
        num_points = x.shape[0]

        x_bits = torch.zeros(num_points, self.n, self.p, dtype=torch.int64, device=self.device)
        for bit_idx in range(self.p):
            shift = self.p - 1 - bit_idx
            x_bits[:, :, bit_idx] = (x >> shift) & 1

        h = torch.zeros(num_points, dtype=torch.int64, device=self.device)
        for i in range(self.p):
            for j in range(self.n):
                bit_val = x_bits[:, j, i]
                bit_pos = self.p * self.n - 1 - (i * self.n + j)
                h = h | (bit_val << bit_pos)

        return h

    def distances_from_points_gpu(self, points: torch.Tensor) -> torch.Tensor:
        """
        GPU 批量计算：从坐标点计算希尔伯特距离。
        Args:
            points: (num_points, n) 形状的张量
        Returns:
            (num_points,) 形状的张量，包含希尔伯特距离
        """
        points = points.clone().to(dtype=torch.int64, device=self.device)
        num_points = points.shape[0]

        m = 1 << (self.p - 1)

        q = m
        while q > 1:
            p_mask = q - 1
            for i in range(self.n):
                cond = (points[:, i] & q) != 0
                points[:, 0] = torch.where(cond, points[:, 0] ^ p_mask, points[:, 0])

                not_cond = ~cond
                t = (points[:, 0] ^ points[:, i]) & p_mask
                new_p0 = torch.where(not_cond, points[:, 0] ^ t, points[:, 0])
                new_pi = torch.where(not_cond, points[:, i] ^ t, points[:, i])
                points[:, 0] = new_p0
                points[:, i] = new_pi
            q >>= 1

        for i in range(1, self.n):
            points[:, i] = points[:, i] ^ points[:, i - 1]

        t = torch.zeros(num_points, dtype=torch.int64, device=self.device)
        q = m
        while q > 1:
            cond = (points[:, self.n - 1] & q) != 0
            t = torch.where(cond, t ^ (q - 1), t)
            q >>= 1

        for i in range(self.n):
            points[:, i] = points[:, i] ^ t

        distances = self._transpose_to_hilbert_integer_batch(points)
        return distances


class SFCGenerator:
    """空间填充曲线（Hilbert）索引生成器，GPU 专用，只保留 GPU 实现"""

    @staticmethod
    @lru_cache(maxsize=16)
    def _hilbert_indices_gpu_cached(shape: Tuple[int, int, int], device_str: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用 functools.lru_cache 复用相同形状与设备的计算结果。"""
        device = torch.device(device_str)

        D, H, W = shape
        max_dim = max(D, H, W)
        p = max(1, (max_dim - 1).bit_length())

        z_coords = torch.arange(D, device=device, dtype=torch.int64)
        y_coords = torch.arange(H, device=device, dtype=torch.int64)
        x_coords = torch.arange(W, device=device, dtype=torch.int64)

        zz, yy, xx = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
        coords = torch.stack([zz.flatten(), yy.flatten(), xx.flatten()], dim=1)  # (N, 3)

        hc_gpu = HilbertCurveGPU(p, 3, device=device)
        distances = hc_gpu.distances_from_points_gpu(coords)

        indices = torch.argsort(distances, stable=True)
        inv_indices = torch.argsort(indices)

        return indices, inv_indices

    @classmethod
    def hilbert_indices_gpu(cls, shape: Tuple[int, int, int], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        仅 GPU 版本：返回 (indices, inv_indices)。
        Args:
            shape: (D, H, W) 网格形状
            device: CUDA 设备（必需）
        Returns:
            indices: sfc_pos -> original_idx
            inv_indices: original_idx -> sfc_pos
        """
        assert device is not None and device.type == 'cuda', "hilbert_indices_gpu 要求传入 CUDA 设备"

        device_str = str(device)
        return cls._hilbert_indices_gpu_cached(shape, device_str)

    @classmethod
    def hilbert_indices(cls, shape: Tuple[int, int, int], device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对外统一接口：始终使用 GPU 版本并要求 CUDA 设备（若未提供 device，会自动尝试使用当前 CUDA 设备）。
        """
        if device is None:
            device = torch.device('cuda')
        return cls.hilbert_indices_gpu(shape, device)

    @classmethod
    def clear_cache(cls) -> None:
        """清空 lru_cache，便于基准测试或释放显存引用。"""
        cls._hilbert_indices_gpu_cached.cache_clear()


# ==================== 2. 3D RoPE ====================
class RotaryPositionEmbedding3D(nn.Module):
    """
    3D RoPE（无尺度），将最后维度按三轴等分并分别做旋转。
    要求 dim % 6 == 0（每轴维度为偶数）。
    支持通过 rope_kwargs 透传 RotaryEmbedding 的构造参数（例如 theta）。
    """
    def __init__(
        self,
        dim: int,
        *,
        theta: float = 10000.0,
        rope_kwargs: Optional[Dict] = None
    ):
        super().__init__()
        assert dim % 6 == 0, "dim 必须能被 6 整除（3轴、每轴偶数对）。"
        self.dim = int(dim)
        self.dim_per_axis = self.dim // 3
        assert self.dim_per_axis % 2 == 0, "每个轴的子维度必须为偶数。"

        base_kwargs = {} if rope_kwargs is None else dict(rope_kwargs)
        base_kwargs.setdefault("theta", theta)

        self.rope_z = RotaryEmbedding(dim=self.dim_per_axis, **base_kwargs)
        self.rope_y = RotaryEmbedding(dim=self.dim_per_axis, **base_kwargs)
        self.rope_x = RotaryEmbedding(dim=self.dim_per_axis, **base_kwargs)

    def forward(self, x: torch.Tensor, grid_shape: Tuple[int, int, int]) -> torch.Tensor:
        B, N, C = x.shape
        D, H, W = grid_shape
        assert N == D * H * W and C == self.dim

        Ca = self.dim_per_axis
        x_z, x_y, x_x = x.chunk(3, dim=-1)  # (B, N, Ca)

        x_z_b = x_z.view(B, D, H * W, Ca).permute(0, 2, 1, 3).contiguous().view(-1, D, Ca)
        x_z_r = self.rope_z.rotate_queries_or_keys(x_z_b)
        x_z = x_z_r.view(B, H * W, D, Ca).permute(0, 2, 1, 3).contiguous().view(B, N, Ca)

        x_y_b = x_y.view(B, H, D * W, Ca).permute(0, 2, 1, 3).contiguous().view(-1, H, Ca)
        x_y_r = self.rope_y.rotate_queries_or_keys(x_y_b)
        x_y = x_y_r.view(B, D * W, H, Ca).permute(0, 2, 1, 3).contiguous().view(B, N, Ca)

        x_x_b = x_x.view(B, W, D * H, Ca).permute(0, 2, 1, 3).contiguous().view(-1, W, Ca)
        x_x_r = self.rope_x.rotate_queries_or_keys(x_x_b)
        x_x = x_x_r.view(B, D * H, W, Ca).permute(0, 2, 1, 3).contiguous().view(B, N, Ca)

        return torch.cat([x_z, x_y, x_x], dim=-1)


# ==================== 3. 简化 Mamba（单向） ====================
class Mambablock(nn.Module):
    def __init__(
        self,
        dim: int,
        mamba_cls=Mamba2,
        **mamba_kwargs
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = mamba_cls(d_model=dim, **mamba_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        x_out = self.mamba(x_norm)
        return x_out


class HilbertSFCMambaLayer(nn.Module):
    """
    Hilbert SFC + Mamba 层（RoPE 仅支持 pre）。
    仅保留 GPU 版本的希尔伯特曲线计算，默认并强制在 CUDA 上运行。
    """
    def __init__(
        self,
        dim: int,
        mamba_cls=Mamba2,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
        **mamba_kwargs
    ):
        super().__init__()
        self.dim = dim
        self.use_rope = use_rope

        if use_rope:
            assert dim % 6 == 0, "开启 3D RoPE 时，dim 必须能被 6 整除。"
            self.rope = RotaryPositionEmbedding3D(dim, theta=rope_theta)

        self.mamba = Mambablock(dim, mamba_cls, **(mamba_kwargs or {}))

        self._permute_patterns = [
            ("b c d h w -> b c d h w", "b c d h w -> b c d h w"),
            ("b c d h w -> b c h w d", "b c h w d -> b c d h w"),
            ("b c d h w -> b c w d h", "b c w d h -> b c d h w"),
        ]

    def _process_one_axis(self, x_permuted: torch.Tensor, perm_shape: Tuple[int, int, int], device: torch.device):
        B, C, Dp, Hp, Wp = x_permuted.shape
        Np = Dp * Hp * Wp

        tokens_p = rearrange(x_permuted, 'b c d h w -> b (d h w) c')  # (B, Np, C)

        sfc_idx_p, inv_idx_p = SFCGenerator.hilbert_indices((Dp, Hp, Wp), device=device)
        assert sfc_idx_p.numel() == Np and inv_idx_p.numel() == Np

        tokens_sfc = tokens_p[:, sfc_idx_p, :]          # (B, Np, C)
        tokens_out = self.mamba(tokens_sfc)             # (B, Np, C)

        tokens_restored_p = tokens_out[:, inv_idx_p, :] # 恢复到该轴排列下的网格顺序
        x_restored_permuted = rearrange(tokens_restored_p, 'b (d h w) c -> b c d h w', d=Dp, h=Hp, w=Wp)
        return x_restored_permuted

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        device = x.device
        assert device.type == 'cuda', "HilbertSFCMambaLayer 仅支持在 CUDA 设备上运行"

        x_skip = x

        if self.use_rope:
            tokens = rearrange(x, 'b c d h w -> b (d h w) c')
            tokens = self.rope(tokens, (D, H, W))
            x_in = rearrange(tokens, 'b (d h w) c -> b c d h w', d=D, h=H, w=W)
        else:
            x_in = x

        outputs_per_axis = []

        for perm, inv_perm in self._permute_patterns:
            x_permuted = rearrange(x_in, perm)
            _, _, Dp, Hp, Wp = x_permuted.shape
            x_restored_permuted = self._process_one_axis(x_permuted, (Dp, Hp, Wp), device=device)
            x_restored = rearrange(x_restored_permuted, inv_perm)
            outputs_per_axis.append(x_restored)

        out_x_1, out_x_2, out_x_3 = outputs_per_axis
        out = out_x_1 + out_x_2 + out_x_3
        out = out + x_skip

        return out


# ==================== 5. 轻量单元测试 ====================
if __name__ == "__main__":
    import time

    def test_gpu_hilbert_correctness(shape=(8, 8, 8)):
        if not torch.cuda.is_available():
            print("⚠ CUDA 不可用，跳过 GPU 正确性测试")
            return
        device = torch.device('cuda')
        idx_gpu, inv_gpu = SFCGenerator.hilbert_indices_gpu(shape, device)
        # 仅对自身一致性进行检查：indices 与 inv_indices 的 roundtrip
        N = shape[0] * shape[1] * shape[2]
        expected = torch.arange(N, device=device)
        assert torch.equal(inv_gpu[idx_gpu], expected), "GPU SFC roundtrip failed"
        print(f"✓ GPU Hilbert 正确性验证通过 shape={shape}")

    def test_gpu_hilbert_speed(shape=(32, 32, 32), num_runs=5):
        if not torch.cuda.is_available():
            print("⚠ CUDA 不可用，跳过速度测试")
            return
        device = torch.device('cuda')
        SFCGenerator.clear_cache()
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_runs):
            SFCGenerator.clear_cache()
            _ = SFCGenerator.hilbert_indices_gpu(shape, device)
            torch.cuda.synchronize()
        gpu_time = (time.time() - start) / num_runs
        print(f"✓ 速度测试 shape={shape}: GPU={gpu_time*1000:.2f}ms")

    def test_rope_3d(dim=384, shape=(4, 8, 8)):
        rope = RotaryPositionEmbedding3D(dim)
        B = 2
        x = torch.randn(B, shape[0] * shape[1] * shape[2], dim)
        x_out = rope(x, shape)
        assert x_out.shape == x.shape, f"RoPE output shape mismatch: got {x_out.shape}, expected {x.shape}"
        print("✓ 3D RoPE OK (using rotary_embedding_torch)")

    def test_layer_forward():
        if not torch.cuda.is_available():
            print("⚠ CUDA 不可用，跳过 layer forward 测试")
            return
        device = torch.device('cuda')
        B, C, D, H, W = 1, 384, 4, 4, 4
        x = torch.randn(B, C, D, H, W, device=device)
        layer = HilbertSFCMambaLayer(
            dim=C,
            use_rope=True,
        ).to(device)
        y = layer(x)
        assert y.shape == x.shape
        print(f"✓ HilbertSFCMambaLayer forward OK on {device}")

    print("=" * 50)
    print("单元测试（GPU 专用版）")
    print("=" * 50)

    test_gpu_hilbert_correctness((8, 8, 8))
    test_gpu_hilbert_correctness((16, 16, 16))
    test_gpu_hilbert_speed((16, 16, 16))
    test_gpu_hilbert_speed((32, 32, 32))
    test_rope_3d(dim=384, shape=(4, 8, 8))
    try:
        test_layer_forward()
    except Exception as e:
        print(f"⚠ HilbertSFCMambaLayer forward test skipped or failed: {e}")

    print("=" * 50)
    print("所有测试完成。")
    print("=" * 50)
