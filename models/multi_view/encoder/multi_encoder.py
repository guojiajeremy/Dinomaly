from __future__ import annotations

import os
os.environ["PYTHONPATH"] = "/root/Dinomaly"

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from models.multi_view.encoder.remap import SpatialChannelRemap

from models.multi_view.encoder.clip import ClipExtractor
from models.multi_view.encoder.dino import DinoExtractor
from models.multi_view.encoder.donut import DonutExtractor
from models.multi_view.encoder.resnet import ResNetExtractor



class MultiEncoder(nn.Module):
    """Build and register multiple feature extractors.

    `encoder_configs` is a dict keyed by encoder name (e.g. "dino", "clip").
    Each config can include:
      - common: `fuse_layer_encoder`, `target_layers`
      - dino:   `backbone` (str), `n` (int|list[int]), `norm` (bool), `trainable` (bool)
      - clip:   `arch` (str), `pretrained` (str), `cache_dir` (str), `force_quick_gelu` (bool), `n`, `norm`
      - resnet: `arch` (str), `pretrained` (bool), `n`
      - donut:  `model_name` (str), `cache_dir` (str), `local_files_only` (bool), `do_resize` (bool), `image_size` (tuple[int,int])

    You may also pass a pre-built backbone as `model` in any config to bypass auto-loading.
    """

    def __init__(
        self,
        encoder_configs: Dict[str, Dict[str, Any]],
        *,
        default_cache_dir: Optional[str] = None,
        output_dims = 768,
        output_h = 37,
        output_w = 37
    ):
        super().__init__()

        if default_cache_dir is None:
            default_cache_dir = str(Path(__file__).resolve().parents[3] / "backbones" / "weights")

        self.extractors: nn.ModuleDict = nn.ModuleDict()
        self.fuse_layer_encoders: Dict[str, Any] = {}
        self.target_layers: Dict[str, Any] = {}

        for name, config in encoder_configs.items():
            cfg = dict(config or {})

            self.fuse_layer_encoders[name] = cfg.get("fuse_layer_encoder", [])
            self.target_layers[name] = cfg.get("target_layers", [])

            if name == "dino":
                extractor = self._build_dino(cfg)
            elif name == "clip":
                extractor = self._build_clip(cfg, default_cache_dir=default_cache_dir)
            elif name == "resnet":
                extractor = self._build_resnet(cfg)
            elif name == "donut":
                extractor = self._build_donut(cfg, default_cache_dir=default_cache_dir)
            else:
                raise ValueError(f"Unsupported encoder type: {name}")

            self.extractors[name] = extractor

    # ------------------------
    # builders
    # ------------------------

    def _build_dino(self, cfg: Dict[str, Any]) -> nn.Module:
        model = cfg.get("model", None)
        if model is None:
            backbone = cfg.get("backbone", "dinov2reg_vit_base_14")
            from models.vit_encoder import load as load_vit

            model = load_vit(backbone)

        trainable = bool(cfg.get("trainable", False))
        if not trainable:
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)

        n = cfg.get("n", 1)
        norm = bool(cfg.get("norm", True))
        return DinoExtractor(model=model, n=n, norm=norm)

    def _build_clip(self, cfg: Dict[str, Any], *, default_cache_dir: str) -> nn.Module:
        model = cfg.get("model", None)
        if model is None:
            try:
                import open_clip
            except Exception as e:
                raise RuntimeError(
                    "open_clip is required for CLIP encoder. Install open-clip-torch."
                ) from e

            arch = cfg.get("arch", "ViT-B-16")
            pretrained = cfg.get("pretrained", "openai")
            force_quick_gelu = cfg.get(
                "force_quick_gelu",
                True if str(pretrained).lower() == "openai" else False,
            )
            cache_dir = cfg.get("cache_dir", str(Path(default_cache_dir) / "open_clip"))

            clip_model, _, _ = open_clip.create_model_and_transforms(
                arch,
                pretrained=pretrained,
                force_quick_gelu=force_quick_gelu,
                cache_dir=cache_dir,
                force_image_size=518
            )
            
            model = clip_model.visual

        trainable = bool(cfg.get("trainable", False))
        if not trainable:
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)

        n = cfg.get("n", 1)
        norm = bool(cfg.get("norm", False))
        return ClipExtractor(model=model, n=n, norm=norm)

    def _build_resnet(self, cfg: Dict[str, Any]) -> nn.Module:
        model = cfg.get("model", None)
        if model is None:
            arch = cfg.get("arch", "wide_resnet50_2")
            pretrained = bool(cfg.get("pretrained", True))

            import torchvision
            try:
                from torchvision.models import get_model, get_model_weights

                weights = None
                if pretrained:
                    weights_enum = get_model_weights(arch)
                    weights = weights_enum.DEFAULT
                model = get_model(arch, weights=weights)
            except Exception:
                # Fallback for older torchvision APIs
                if not hasattr(torchvision.models, arch):
                    raise ValueError(f"Unsupported torchvision resnet arch: {arch}")
                fn = getattr(torchvision.models, arch)
                model = fn(pretrained=pretrained)

        trainable = bool(cfg.get("trainable", False))
        if not trainable:
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)

        n = cfg.get("n", 4)
        return ResNetExtractor(model=model, n=n)

    def _build_donut(self, cfg: Dict[str, Any], *, default_cache_dir: str) -> nn.Module:
        # DonutExtractor already loads weights internally
        model_name = cfg.get("model_name", "naver-clova-ix/donut-base")
        cache_dir = cfg.get("cache_dir", str(Path(default_cache_dir) / "hf"))
        local_files_only = cfg.get("local_files_only", None)
        do_resize =False
        image_size = cfg.get("image_size", None)
        trainable = False
        revision = cfg.get("revision", None)

        return DonutExtractor(
            model_name=model_name,
            trainable=trainable,
            do_resize=do_resize,
            image_size=image_size,
            cache_dir=cache_dir,
            revision=revision,
            local_files_only=local_files_only,
        )

    # ------------------------
    # forward
    # ------------------------

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        '''return a dict keyed by encoder name, 
        each value is another dict containing "target" and "fused_group_feats".'''
        results = {} #key is encoder name, value is a dict, which is "result" in the following loop.
        for name, extractor in self.extractors.items():
            remaper =SpatialChannelRemap() 
            result = {}
            if name not in self.fuse_layer_encoders:
                raise ValueError(f"Missing fuse_layer_encoder for encoder {name}")
            if name not in self.target_layers:
                raise ValueError(f"Missing target_layers for encoder {name}")   
            outputs = extractor(x) #outputs is list, containing features from encoders different layers.
            idx = self.target_layers[name]  # e.g. [1,3,5]
            target = torch.stack([remaper(outputs[i],out_channels = 768, out_size=[37,37]) for i in idx], dim=1).mean(dim=1)
            result["target"] = target
            fused_group_feats = []
            # now, scale and channels change not supported, so resnet and donut will lead to failure.
            # next, introduce scale and channel transformations to enable multi-scale and multi-channel fusion.
            for i, fuse_group in enumerate(self.fuse_layer_encoders[name]):  # e.g. [[1,3],[5,7]]
                fused_group_feats.append(torch.stack([remaper(outputs[j], out_channels=768, out_size=[37,37]) for j in fuse_group], dim=1).mean(dim=1))
            result["fused_group_feats"] = fused_group_feats
            result["outputs"] = outputs
            results[name] = result
            
        return results

    