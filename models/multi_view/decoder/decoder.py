from __future__ import annotations

from functools import partial
import torch.nn as nn
from models.vision_transformer import Block as VitBlock ,LinearAttention2


from typing import List

import torch


class Decoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_blocks: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_eps: float = 1e-8,
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.registers = nn.Parameter(torch.randn(1, 4, embed_dim))
        decoder = []
        for _ in range(num_blocks):
            blk = VitBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=partial(nn.LayerNorm, eps=norm_eps),
                attn=LinearAttention2,
            )
            decoder.append(blk)

        self.decoder = nn.ModuleList(decoder)

    def forward(self, x) -> List[torch.Tensor]:

        outputs: List[torch.Tensor] = []
        # add cls token and registers to the beginning of the sequence
        for blk in self.decoder:
            x = blk(x)
            outputs.append(x)

        return outputs