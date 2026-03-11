from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import DonutSwinModel, VisionEncoderDecoderModel


class DonutExtractor(nn.Module):
    """
    Donut Swin feature extractor

    接口:
        feats = extractor(pixel_values)

    返回:
        List[(B,C,H,W)]

    返回顺序:
        stage1
        stage2
        stage3
        stage4
    """

    def __init__(
        self,
        model_name: str = "naver-clova-ix/donut-base",
        trainable: bool = False,
        do_resize: bool = True,
        image_size: Optional[Tuple[int, int]] = None,
        cache_dir: Optional[str] = None,
        revision: Optional[str] = None,
        local_files_only: Optional[bool] = None,
    ):
        super().__init__()

        if local_files_only is None:
            local_files_only = bool(
                str(os.environ.get("HF_HUB_OFFLINE", "")).strip() == "1"
                or str(os.environ.get("TRANSFORMERS_OFFLINE", "")).strip() == "1"
            )

        pretrained_kwargs = dict(
            cache_dir=cache_dir,
            revision=revision,
            local_files_only=local_files_only,
        )

        # 优先加载完整 Donut 模型
        try:
            full = VisionEncoderDecoderModel.from_pretrained(
                model_name,
                **pretrained_kwargs,
            )
            self.encoder = full.encoder
        except Exception:
            self.encoder = DonutSwinModel.from_pretrained(
                model_name,
                **pretrained_kwargs,
            )

        self.do_resize = bool(do_resize)

        if image_size is not None:
            self.encoder.config.image_size = image_size

        if not trainable:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def _resize_if_needed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if not self.do_resize:
            return pixel_values

        size = getattr(self.encoder.config, "image_size", None)

        if size is None:
            return pixel_values

        if isinstance(size, int):
            size = (size, size)

        if pixel_values.shape[-2:] == tuple(size):
            return pixel_values

        return F.interpolate(
            pixel_values,
            size=size,
            mode="bicubic",
            align_corners=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> List[torch.Tensor]:

        if pixel_values.dim() != 4:
            raise ValueError(
                f"Expected input (B,C,H,W), got {tuple(pixel_values.shape)}"
            )

        pixel_values = self._resize_if_needed(pixel_values)

        outputs = self.encoder(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

        if not hasattr(outputs, "reshaped_hidden_states"):
            raise RuntimeError(
                "Donut model did not return reshaped_hidden_states."
            )

        reshaped = outputs.reshaped_hidden_states

        if reshaped is None or len(reshaped) == 0:
            raise RuntimeError(
                "reshaped_hidden_states is empty."
            )

        # 第0个是 embedding 输出
        feats = list(reshaped)[1:]

        if len(feats) == 0:
            raise RuntimeError(
                "No stage features found in reshaped_hidden_states."
            )

        return feats