from __future__ import annotations

from functools import partial
from collections import defaultdict
import torch.nn as nn
from models.multi_view.decoder.decoder_adapters import DecoderAdapter
from models.vision_transformer import Block as VitBlock ,LinearAttention2


from typing import Any, Dict, List

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
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), self.registers.expand(x.size(0), -1, -1), x], dim=1)
        # add cls token and registers to the beginning of the sequence
        for blk in self.decoder:
            x = blk(x)
            outputs.append(x[:,1+4:, :])  # remove cls token and registers, keep the rest as output

        return outputs[::-1]  # reverse the order of outputs to match the order of target layers in encoder
    
class Decoder_with_adapter(Decoder):
    def __init__(
        self,
        encoder_configs: Dict[str, Dict[str, Any]],
        *,
        embed_dim: int,
        num_heads: int,
        num_blocks: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_eps: float = 1e-8,
        num_register_tokens: int = 0,
        num_conv: int = 2,
    ):
        super().__init__(embed_dim, num_heads, num_blocks, mlp_ratio, qkv_bias, norm_eps)
        self.adapters = defaultdict(dict)
        self.fuse_layer_decoder: Dict[str, List[List[int]]] = {}

        for name, cfg in encoder_configs.items():
            self.fuse_layer_decoder[name] = cfg.get("fuse_layer_decoder", [])
            for a, b in cfg.get("adapter_layers", []):
                adapter = DecoderAdapter(embed_dim=embed_dim, num_conv=num_conv)
                self.adapters[name][b] = {
                    "a": a,
                    "adapter": adapter
                }

                self.add_module(f"adapter_{name}_{a}_{b}", adapter)
            
            
        
    def forward(self, x):
        decoder_outputs = super().forward(x)   # List[Tensor]

        # 先收集每个 encoder 每一层的 adapter 输出
        encoder_layer_outputs = {}

        for name, layer_map in self.adapters.items():
            encoder_layer_outputs[name] = {}
            for b, info in layer_map.items():
                a = info["a"]
                adapter = info["adapter"]
                if not (0 <= b < len(decoder_outputs)):
                    raise ValueError(
                        f"Invalid adapter layer mapping for encoder '{name}': b={b} is out of range "
                        f"for decoder outputs of length {len(decoder_outputs)}"
                    )
                encoder_layer_outputs[name][a] = adapter(decoder_outputs[b])

        # 再按 fuse_layer_decoder 做组内平均
        de_dict = {}

        for name, groups in self.fuse_layer_decoder.items():
            fused_groups = []
            layer_outputs = encoder_layer_outputs[name]

            for group in groups:   # 例如 [0,1]
                missing = [a for a in group if a not in layer_outputs]
                if missing:
                    raise ValueError(
                        f"Missing adapter outputs for encoder '{name}' at layer(s) {missing}. "
                        f"Check encoder_configs['{name}']['adapter_layers'] and fuse_layer_decoder."
                    )
                feats = [layer_outputs[a] for a in group]
                fused = torch.stack(feats, dim=0).mean(dim=0)
                fused_groups.append(fused)

            de_dict[name] = fused_groups

        return de_dict