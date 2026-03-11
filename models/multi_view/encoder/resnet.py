from __future__ import annotations

from typing import List, Sequence, Union

import torch
import torch.nn as nn


class ResNetExtractor(nn.Module):
    """
    ResNet / WideResNet 中间层提取器

    统一接口:
        feats = extractor(x)

    返回:
        List[Tensor]，每个元素形状 (B, C, H, W)

    支持:
        - n = int           取最后 n 个 stage
        - n = Sequence[int] 取指定 stage index

    stage 定义:
        0 -> layer1
        1 -> layer2
        2 -> layer3
        3 -> layer4
    """

    def __init__(
        self,
        model: nn.Module,
        n: Union[int, Sequence[int]] = 4,
    ):
        super().__init__()
        self.model = model
        self.n = n

        self._check_backbone()

    def _check_backbone(self) -> None:
        required = [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
        ]
        for name in required:
            if not hasattr(self.model, name):
                raise AttributeError(
                    f"Backbone missing required attribute '{name}'"
                )

    def _select_indices(
        self,
        total_layers: int,
        n: Union[int, Sequence[int]],
    ) -> List[int]:
        if isinstance(n, int):
            if n <= 0:
                return []
            n = min(n, total_layers)
            return list(range(total_layers - n, total_layers))

        indices = [int(i) for i in n]
        indices = [i for i in indices if 0 <= i < total_layers]
        return indices

    def _forward_stem(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        return x

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W)

        Returns:
            List[(B, C, H, W)]
        """
        if x.dim() != 4:
            raise ValueError(f"Expected (B,C,H,W), got {tuple(x.shape)}")

        take_indices = set(self._select_indices(4, self.n))

        x = self._forward_stem(x)

        outputs: List[torch.Tensor] = []

        x = self.model.layer1(x)
        if 0 in take_indices:
            outputs.append(x)

        x = self.model.layer2(x)
        if 1 in take_indices:
            outputs.append(x)

        x = self.model.layer3(x)
        if 2 in take_indices:
            outputs.append(x)

        x = self.model.layer4(x)
        if 3 in take_indices:
            outputs.append(x)

        return outputs