from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Sequence, Union


class DinoExtractor(nn.Module):
    """
    DINO / DINOv2 ViT 中间层特征提取器

    统一接口:
        feats = extractor(x)

    返回:
        List[Tensor]  每个元素形状 (B, C, H, W)

    支持:
        - n = int          取最后 n 层
        - n = List[int]    取指定 block index
    """

    def __init__(
        self,
        model: nn.Module,
        n: Union[int, Sequence[int]] = 1,
        norm: bool = True,
    ):
        super().__init__()

        self.model = model
        self.n = n
        self.norm = norm

        self.patch_size = int(model.patch_size)
        self.num_register_tokens = int(getattr(model, "num_register_tokens", 0))

        # CLS + register tokens
        self.num_prefix_tokens = 1 + self.num_register_tokens

    # -------------------------------------------------
    # utilities
    # -------------------------------------------------

    def _flatten_blocks(self) -> List[nn.Module]:
        """
        将 chunked blocks 展平成一个 Block 列表

        兼容:
            ModuleList([Block,...])
            ModuleList([BlockChunk,...])
        """
        flat_blocks: List[nn.Module] = []

        for blk in self.model.blocks:

            if isinstance(blk, nn.ModuleList):
                for sub in blk:
                    if isinstance(sub, nn.Identity):
                        continue
                    flat_blocks.append(sub)

            else:
                flat_blocks.append(blk)

        return flat_blocks

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

        indices = [
            i for i in indices
            if 0 <= i < total_blocks
        ]

        return indices

    def _tokens_to_bchw(
        self,
        x: torch.Tensor,
        input_hw: tuple[int, int],
    ) -> torch.Tensor:

        """
        (B,N,C) -> (B,C,H,W)

        去掉
            CLS token
            register tokens
        """

        if x.dim() != 3:
            raise ValueError(
                f"Expected (B,N,C), got {tuple(x.shape)}"
            )

        B, N, C = x.shape

        if N <= self.num_prefix_tokens:
            raise RuntimeError(
                f"Token count too small: "
                f"N={N}, prefix={self.num_prefix_tokens}"
            )

        patch_tokens = x[:, self.num_prefix_tokens:, :]

        num_patches = patch_tokens.shape[1]

        H_in, W_in = input_hw

        if H_in % self.patch_size != 0 or W_in % self.patch_size != 0:
            raise RuntimeError(
                f"Input size {input_hw} not divisible "
                f"by patch_size={self.patch_size}"
            )

        H = H_in // self.patch_size
        W = W_in // self.patch_size

        if H * W != num_patches:
            raise RuntimeError(
                f"Patch count mismatch: expected {H*W} "
                f"patches but got {num_patches}"
            )

        x = patch_tokens.transpose(1, 2).reshape(
            B,
            C,
            H,
            W
        ).contiguous()

        return x

    # -------------------------------------------------
    # forward
    # -------------------------------------------------

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:

        """
        Args
            x: (B,3,H,W)

        Returns
            List[(B,C,H,W)]
        """

        if x.dim() != 4:
            raise ValueError(
                f"Expected (B,C,H,W), got {tuple(x.shape)}"
            )

        input_hw = (
            int(x.shape[2]),
            int(x.shape[3]),
        )

        # flatten blocks
        blocks = self._flatten_blocks()

        total_blocks = len(blocks)

        take_indices = set(
            self._select_block_indices(
                total_blocks,
                self.n,
            )
        )

        # prepare tokens
        x = self.model.prepare_tokens_with_masks(x)

        outputs: List[torch.Tensor] = []

        for i, blk in enumerate(blocks):

            x = blk(x)

            if i in take_indices:

                out = self.model.norm(x) if self.norm else x

                out = self._tokens_to_bchw(
                    out,
                    input_hw,
                )

                outputs.append(out)

        return outputs