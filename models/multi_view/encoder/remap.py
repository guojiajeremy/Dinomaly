import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _find_coprime_multiplier(n: int) -> int:
    if n <= 1:
        return 1
    cand = max(1, n // 2)
    if cand % 2 == 0:
        cand += 1
    for delta in range(n):
        x1 = cand + delta
        if x1 < n and _gcd(x1, n) == 1:
            return x1
        x2 = cand - delta
        if x2 > 0 and _gcd(x2, n) == 1:
            return x2
    return 1


def _find_coprime_step(n: int) -> int:
    if n <= 1:
        return 1
    for s in [3, 5, 7, 11, 13, 17]:
        if s < n and _gcd(s, n) == 1:
            return s
    for s in range(2, n):
        if _gcd(s, n) == 1:
            return s
    return 1


class SpatialChannelRemap(nn.Module):
    """
    更接近函数式的非学习 remap 模块。

    用法:
        remap = SpatialChannelRemap()

        y = remap(x, out_channels=768, out_size=(37, 37))
        y = remap(x, out_channels=768, scale_factor=(1.25, 1.25))

    说明:
    - 输入通道数从 x 动态读取
    - 输入空间大小从 x 动态读取
    - 输出通道数在 forward 指定
    - 输出空间大小在 forward 指定
    """

    def __init__(
        self,
        upsample_mode: str = "bilinear",
        downsample_mode: str = "area",
        align_corners: bool = False,
        n_phases: int = 16,
        perm_offset: int = 0,
        cache_size: int = 16,
    ):
        super().__init__()

        self.upsample_mode = upsample_mode
        self.downsample_mode = downsample_mode
        self.align_corners = align_corners
        self.n_phases = max(1, int(n_phases))
        self.perm_offset = int(perm_offset)
        self.cache_size = int(cache_size)

        # 轻量缓存，不注册为 buffer；更像运行时 memoization
        self._cache: Dict[Any, Dict[str, torch.Tensor]] = {}

    @staticmethod
    def _build_channel_order(
        in_channels: int,
        perm_offset: int,
        device: torch.device,
    ) -> torch.Tensor:
        perm_multiplier = _find_coprime_multiplier(in_channels)
        base = torch.arange(in_channels, dtype=torch.long, device=device)
        channel_order = (perm_multiplier * base + (perm_offset % in_channels)) % in_channels
        return channel_order.long()

    @staticmethod
    def _build_channel_remap_table(
        channel_order: torch.Tensor,
        in_channels: int,
        out_channels: int,
        n_phases: int,
    ) -> torch.Tensor:
        """
        返回 [n_phases, out_channels]
        """
        device = channel_order.device

        base_pos = torch.floor(
            torch.arange(out_channels, device=device, dtype=torch.float32)
            * (in_channels / out_channels)
        ).long()

        shifts = torch.floor(
            torch.arange(n_phases, device=device, dtype=torch.float32)
            * (in_channels / max(1, n_phases))
        ).long()

        pos = (base_pos.unsqueeze(0) + shifts.unsqueeze(1)) % in_channels
        return channel_order[pos].long()

    @staticmethod
    def _build_phase_map(
        h_out: int,
        w_out: int,
        n_phases: int,
        device: torch.device,
    ) -> torch.Tensor:
        if n_phases == 1:
            return torch.zeros((h_out, w_out), dtype=torch.long, device=device)

        step = _find_coprime_step(n_phases)
        u = torch.arange(h_out, dtype=torch.long, device=device).unsqueeze(1)
        v = torch.arange(w_out, dtype=torch.long, device=device).unsqueeze(0)
        return (u + step * v) % n_phases

    def _resolve_out_size(
        self,
        h_in: int,
        w_in: int,
        out_size: Optional[Tuple[int, int]],
        scale_factor: Optional[Tuple[float, float]],
    ) -> Tuple[int, int]:
        if out_size is not None:
            return int(out_size[0]), int(out_size[1])

        if scale_factor is None:
            raise ValueError("Either out_size or scale_factor must be provided")

        sh, sw = scale_factor
        return max(1, int(round(h_in * sh))), max(1, int(round(w_in * sw)))

    def _spatial_resize(self, x: torch.Tensor, out_size: Tuple[int, int]) -> torch.Tensor:
        h_in, w_in = x.shape[-2:]
        h_out, w_out = out_size

        if (h_in, w_in) == (h_out, w_out):
            return x

        is_upsample = (h_out > h_in) or (w_out > w_in)
        mode = self.upsample_mode if is_upsample else self.downsample_mode

        if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            return F.interpolate(
                x,
                size=out_size,
                mode=mode,
                align_corners=self.align_corners,
            )

        return F.interpolate(x, size=out_size, mode=mode)

    def _make_cache_key(
        self,
        in_channels: int,
        out_channels: int,
        h_out: int,
        w_out: int,
        device: torch.device,
    ):
        return (
            in_channels,
            out_channels,
            h_out,
            w_out,
            device.type,
            device.index,
            self.n_phases,
            self.perm_offset,
        )

    def _get_or_build_indices(
        self,
        in_channels: int,
        out_channels: int,
        h_out: int,
        w_out: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        返回 [H_out, W_out, C_out] 的 remap_idx
        """
        key = self._make_cache_key(in_channels, out_channels, h_out, w_out, device)

        cached = self._cache.get(key, None)
        if cached is not None:
            return cached["remap_idx"]

        n_phases = max(1, min(self.n_phases, in_channels))

        channel_order = self._build_channel_order(
            in_channels=in_channels,
            perm_offset=self.perm_offset,
            device=device,
        )
        remap_table = self._build_channel_remap_table(
            channel_order=channel_order,
            in_channels=in_channels,
            out_channels=out_channels,
            n_phases=n_phases,
        )
        phase_map = self._build_phase_map(
            h_out=h_out,
            w_out=w_out,
            n_phases=n_phases,
            device=device,
        )
        remap_idx = remap_table[phase_map]  # [H_out, W_out, C_out]

        if len(self._cache) >= self.cache_size:
            # 简单 FIFO 风格：删最早插入的一项
            first_key = next(iter(self._cache))
            self._cache.pop(first_key)

        self._cache[key] = {
            "remap_idx": remap_idx,
        }
        return remap_idx

    def clear_cache(self):
        self._cache.clear()

    def forward(
        self,
        x: torch.Tensor,
        out_channels: Optional[int] = None,
        out_size: Optional[Tuple[int, int]] = None,
        scale_factor: Optional[Tuple[float, float]] = None,
    ) -> torch.Tensor:
        """
        x:
            [B, C, H, W] or [C, H, W]

        参数:
            out_channels:
                目标输出通道数；若为 None，则默认保持输入通道数不变
            out_size:
                目标输出空间尺寸
            scale_factor:
                目标空间缩放比例；当 out_size 为 None 时生效
        """
        single = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            single = True

        if x.dim() != 4:
            raise ValueError("Input must be [B,C,H,W] or [C,H,W]")

        b, in_channels, h_in, w_in = x.shape

        if out_channels is None:
            out_channels = in_channels
        if out_channels <= 0:
            raise ValueError("out_channels must be positive")

        h_out, w_out = self._resolve_out_size(
            h_in=h_in,
            w_in=w_in,
            out_size=out_size,
            scale_factor=scale_factor,
        )

        # 完全 identity 时直接返回
        if (h_in, w_in) == (h_out, w_out) and in_channels == out_channels:
            return x.squeeze(0) if single else x

        z = self._spatial_resize(x, (h_out, w_out))  # [B, C_in, H_out, W_out]

        remap_idx = self._get_or_build_indices(
            in_channels=in_channels,
            out_channels=out_channels,
            h_out=h_out,
            w_out=w_out,
            device=z.device,
        )

        z = z.permute(0, 2, 3, 1)  # [B, H_out, W_out, C_in]
        idx = remap_idx.unsqueeze(0).expand(b, -1, -1, -1)  # [B, H_out, W_out, C_out]
        y = torch.gather(z, dim=3, index=idx)
        y = y.permute(0, 3, 1, 2).contiguous()

        if single:
            y = y.squeeze(0)
        return y