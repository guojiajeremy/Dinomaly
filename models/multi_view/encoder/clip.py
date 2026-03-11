from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Sequence, Union


class ClipExtractor(nn.Module):
    """
    CLIP ViT 中间层特征提取器

    统一接口:
        feats = extractor(x)

    返回:
        List[Tensor]，每个元素形状 (B, C, H, W)

    支持:
        - n = int           取最后 n 层
        - n = Sequence[int] 取指定 block index

    假设视觉塔具有以下成员:
        - conv1
        - class_embedding
        - positional_embedding
        - ln_pre
        - transformer.resblocks
    """

    def __init__(
        self,
        model: nn.Module,
        n: Union[int, Sequence[int]] = 1,
        norm: bool = False,
    ):
        super().__init__()

        self.model = model
        self.n = n
        self.norm = bool(norm)

        self.num_prefix_tokens = 1  # CLS only

    # -------------------------------------------------
    # utilities
    # -------------------------------------------------

    def _get_blocks(self) -> List[nn.Module]:
        if not hasattr(self.model, "transformer"):
            raise AttributeError("CLIP visual model has no attribute 'transformer'")

        if not hasattr(self.model.transformer, "resblocks"):
            raise AttributeError("CLIP visual transformer has no attribute 'resblocks'")

        return list(self.model.transformer.resblocks)

    def _select_block_indices(
        self,
        total_blocks: int,
        n: Union[int, Sequence[int]],
    ) -> List[int]:
        if isinstance(n, int):
            if n <= 0:
                return []
            n = min(n, total_blocks)
            return list(range(total_blocks - n, total_blocks))

        indices = [int(i) for i in n]
        indices = [i for i in indices if 0 <= i < total_blocks]
        return indices

    def _patch_embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        将图像转成 CLIP ViT tokens，输出 (B, N, C)

        流程基本对齐 OpenAI CLIP VisionTransformer.forward：
            conv1 -> flatten patches
            prepend cls token
            add pos embed
            ln_pre
        """
        if not hasattr(self.model, "conv1"):
            raise AttributeError("CLIP visual model has no attribute 'conv1'")
        if not hasattr(self.model, "class_embedding"):
            raise AttributeError("CLIP visual model has no attribute 'class_embedding'")
        if not hasattr(self.model, "positional_embedding"):
            raise AttributeError("CLIP visual model has no attribute 'positional_embedding'")
        if not hasattr(self.model, "ln_pre"):
            raise AttributeError("CLIP visual model has no attribute 'ln_pre'")

        x = self.model.conv1(x)                 # (B, C, H', W')
        B, C, H, W = x.shape

        x = x.reshape(B, C, H * W).permute(0, 2, 1)   # (B, HW, C)

        cls = self.model.class_embedding.to(x.dtype)
        cls = cls + torch.zeros(B, 1, C, dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)                # (B, 1+HW, C)

        pos = self.model.positional_embedding.to(x.dtype)
        if pos.dim() != 2:
            raise RuntimeError(
                f"Expected positional_embedding shape (N,C), got {tuple(pos.shape)}"
            )
        if pos.shape[0] != x.shape[1]:
            raise RuntimeError(
                f"Positional embedding length mismatch: "
                f"expected {x.shape[1]}, got {pos.shape[0]}"
            )

        x = x + pos
        x = self.model.ln_pre(x)

        return x

    def _tokens_to_bchw(
        self,
        x: torch.Tensor,
        patch_hw: tuple[int, int],
    ) -> torch.Tensor:
        """
        (B, N, C) -> (B, C, H, W)
        去掉 CLS token，只保留 patch tokens
        """
        if x.dim() != 3:
            raise ValueError(f"Expected (B,N,C), got {tuple(x.shape)}")

        B, N, C = x.shape
        if N <= self.num_prefix_tokens:
            raise RuntimeError(
                f"Token count too small: N={N}, prefix={self.num_prefix_tokens}"
            )

        H, W = patch_hw
        num_patches = H * W
        patch_tokens = x[:, self.num_prefix_tokens:, :]

        if patch_tokens.shape[1] != num_patches:
            raise RuntimeError(
                f"Patch count mismatch: expected {num_patches}, "
                f"got {patch_tokens.shape[1]}"
            )

        x = patch_tokens.transpose(1, 2).reshape(B, C, H, W).contiguous()
        return x

    def _final_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        可选：对中间层输出做一次最终 norm。
        默认关闭，因为 CLIP 一般只在最后通过 ln_post。
        """
        if not self.norm:
            return x

        if hasattr(self.model, "ln_post") and isinstance(self.model.ln_post, nn.Module):
            return self.model.ln_post(x)

        return x

    # -------------------------------------------------
    # forward
    # -------------------------------------------------

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W)

        Returns:
            List[(B, C, H, W)]
        """
        if x.dim() != 4:
            raise ValueError(f"Expected (B,C,H,W), got {tuple(x.shape)}")

        # 先拿 patch grid，后面 reshape 要严格对齐
        with torch.no_grad():
            patch = self.model.conv1(x)
            patch_hw = (int(patch.shape[2]), int(patch.shape[3]))

        blocks = self._get_blocks()
        total_blocks = len(blocks)
        take_indices = set(self._select_block_indices(total_blocks, self.n))

        x = self._patch_embed(x)

        outputs: List[torch.Tensor] = []

        # CLIP transformer block 通常吃 (N, B, C)
        x = x.permute(1, 0, 2)   # (N, B, C)

        for i, blk in enumerate(blocks):
            x = blk(x)

            if i in take_indices:
                out = x.permute(1, 0, 2)   # (B, N, C)
                out = self._final_norm(out)
                out = self._tokens_to_bchw(out, patch_hw)
                outputs.append(out)

        return outputs