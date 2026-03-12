import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from sklearn.cluster import KMeans
import math
from typing import Dict, List, Optional, Mapping
from dataclasses import dataclass

from models.multi_view.encoder.multi_encoder import MultiEncoder


# =============================================================================
# Multi-Modal Feature Fusion Module (from imiku)
# =============================================================================


@dataclass
class FusionConfig:
    """Configuration for GroupWiseFeatureFuser."""

    modality_channels: Mapping[str, int]
    mid_channels: int = 64
    out_channels: int = 64
    fusion_type: str = "light_transformer"  # "light_transformer" or "masked_attn"
    use_modality_embedding: bool = True


class _ModalityAdapter(nn.Module):
    """Adapter to align different encoder channels to mid_channels."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _LightModalityFusion(nn.Module):
    """Lightweight transformer fusion across modalities at each spatial position.

    Inputs:
        - tok: (B*H*W, M, C) modality tokens per position
        - keep_mask: (B, M) boolean; True means this modality is present

    Output:
        - (B, C, H, W)
    """

    def __init__(
        self,
        channels: int,
        *,
        ff_mult: int = 2,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.channels = int(channels)
        self.scale = float(self.channels) ** -0.5

        self.ln_attn = nn.LayerNorm(self.channels)
        self.qkv = nn.Linear(self.channels, 3 * self.channels, bias=False)
        self.proj = nn.Linear(self.channels, self.channels, bias=True)
        self.drop = nn.Dropout(proj_dropout)
        self.attn_drop = nn.Dropout(attn_dropout)

        self.ln_ff = nn.LayerNorm(self.channels)
        self.ff = nn.Sequential(
            nn.Linear(self.channels, ff_mult * self.channels),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.Linear(ff_mult * self.channels, self.channels),
            nn.Dropout(proj_dropout),
        )

        self.group_null_map = nn.Parameter(torch.zeros(1, self.channels, 1, 1))
        nn.init.zeros_(self.group_null_map)

    def forward(
        self, tok: torch.Tensor, keep_mask: torch.Tensor, *, H: int, W: int
    ) -> torch.Tensor:
        if tok.dim() != 3:
            raise ValueError(f"Expected tok=(BHW,M,C), got {tuple(tok.shape)}")
        if keep_mask.dim() != 2:
            raise ValueError(f"Expected keep_mask=(B,M), got {tuple(keep_mask.shape)}")

        S, M, C = tok.shape
        B = int(keep_mask.shape[0])
        if keep_mask.shape[1] != M:
            raise ValueError(
                f"keep_mask second dim {keep_mask.shape[1]} must match tok M={M}"
            )
        if C != self.channels:
            raise ValueError(
                f"tok channels C={C} must match fusion channels={self.channels}"
            )
        if H <= 0 or W <= 0:
            raise ValueError("H and W must be positive")
        if S != B * H * W:
            raise ValueError(f"tok first dim S={S} must equal B*H*W={B * H * W}")

        if keep_mask.dtype != torch.bool:
            keep_mask = keep_mask.to(dtype=torch.bool)
        if not torch.any(keep_mask):
            return self.group_null_map.expand(B, self.channels, H, W)

        keep_mask_flat = keep_mask.repeat_interleave(H * W, dim=0)  # (BHW, M)

        # attention block
        x = tok
        x_in = self.ln_attn(x)
        q, k, v = self.qkv(x_in).chunk(3, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(~keep_mask_flat[:, None, :], -1e9)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v
        out = self.drop(self.proj(out))
        x = x + out

        # FFN
        x = x + self.ff(self.ln_ff(x))

        # pool modalities -> fused token per position
        w = keep_mask_flat.to(dtype=x.dtype).unsqueeze(-1)  # (BHW,M,1)
        denom = w.sum(dim=1).clamp(min=1.0)
        fused = (x * w).sum(dim=1) / denom
        fused = fused.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return fused


class GroupWiseFeatureFuser(nn.Module):
    """Fuse multiple encoder features into a single spatial feature map using Transformer.

    - Supports different input channels per modality via adapters (Conv1x1)
    - Learnable modality embeddings for each encoder
    - Transformer-based fusion for complex cross-modal interactions
    """

    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        if config.mid_channels <= 0 or config.out_channels <= 0:
            raise ValueError("Channel counts must be positive")
        self.config = config
        self.modalities = list(config.modality_channels.keys())

        # Adapters to align different encoder channels to mid_channels
        self.adapters = nn.ModuleDict(
            {
                k: _ModalityAdapter(in_ch, config.mid_channels)
                for k, in_ch in config.modality_channels.items()
            }
        )

        # Learnable modality identity embeddings (broadcast over H,W)
        self.use_modality_embedding = config.use_modality_embedding
        self.modality_embeddings = nn.ParameterDict()
        if self.use_modality_embedding:
            for key in self.modalities:
                self.modality_embeddings[key] = nn.Parameter(
                    torch.zeros(1, config.mid_channels, 1, 1)
                )
                nn.init.trunc_normal_(self.modality_embeddings[key], std=0.02)

        # Fusion module
        fusion_type = config.fusion_type.lower()
        if fusion_type not in {"light_transformer", "masked_attn"}:
            raise ValueError(f"Unsupported fusion_type={fusion_type!r}")
        self.fusion_type = fusion_type
        self.fusion = _LightModalityFusion(config.mid_channels)

        # Final projection to output channels
        self.final_conv = nn.Conv2d(
            config.mid_channels, config.out_channels, 1, bias=True
        )

    def forward(
        self,
        features: Mapping[str, torch.Tensor],
        *,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse multiple modality features.

        Args:
            features: Dict mapping modality name to feature tensor (B, C, H, W)
            modality_mask: Optional (B, M) boolean mask for modality dropout

        Returns:
            Fused feature tensor (B, out_channels, H, W)
        """
        # Check all modalities are present
        for k in self.modalities:
            if k not in features:
                raise KeyError(f"Missing modality '{k}' in feature map")

        # Apply adapters and add modality embeddings
        adapted: Dict[str, torch.Tensor] = {}
        ref: Optional[torch.Tensor] = None
        for k in self.modalities:
            a = self.adapters[k](features[k])
            if self.use_modality_embedding and k in self.modality_embeddings:
                a = a + self.modality_embeddings[k]
            adapted[k] = a
            if ref is None:
                ref = a

        if ref is None:
            raise ValueError("No available modalities to infer spatial size")

        B, C, H, W = ref.shape
        M = len(self.modalities)

        # Build tokens in a fixed modality order
        tokens: List[torch.Tensor] = []
        avail = torch.ones(B, M, dtype=torch.bool, device=ref.device)

        for idx, k in enumerate(self.modalities):
            t = adapted[k]
            if (t.shape[-2], t.shape[-1]) != (H, W):
                t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
            tokens.append(t.permute(0, 2, 3, 1).contiguous().view(B * H * W, 1, C))

        tok = torch.cat(tokens, dim=1)  # (BHW, M, C)

        # Handle modality mask
        keep = avail
        if modality_mask is not None:
            if modality_mask.dim() == 1:
                if modality_mask.numel() != M:
                    raise ValueError(
                        f"modality_mask has {modality_mask.numel()} elements, expected {M}"
                    )
                modality_mask = modality_mask.view(1, M).expand(B, M)
            if modality_mask.shape != (B, M):
                raise ValueError(
                    f"modality_mask must be (B,M)={(B, M)}, got {tuple(modality_mask.shape)}"
                )
            if modality_mask.dtype != torch.bool:
                modality_mask = modality_mask.to(dtype=torch.bool)
            keep = keep & modality_mask

        # Ensure each sample has at least one modality
        keep_sum = keep.sum(dim=1)
        needs_fix = keep_sum == 0
        if torch.any(needs_fix):
            first_avail = torch.zeros(B, dtype=torch.long, device=ref.device)
            keep[needs_fix, :] = False
            keep[needs_fix, 0] = True

        # Apply fusion
        fused_mid = self.fusion(tok, keep, H=H, W=W)

        return self.final_conv(fused_mid)

    def summary(self) -> str:
        return (
            f"GroupWiseFeatureFuser(modalities={len(self.modalities)}, "
            f"mid={self.config.mid_channels}, out={self.config.out_channels}, "
            f"fusion={self.fusion_type})"
        )


from models.multi_view.encoder.multi_encoder import MultiEncoder


class ViTill(nn.Module):
    def __init__(
        self,
        encoder,
        bottleneck,
        decoder,
        target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
        fuse_layer_encoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
        fuse_layer_decoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
        mask_neighbor_size=0,
        remove_class_token=False,
        encoder_require_grad_layer=[],
    ) -> None:
        super(ViTill, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer

        if not hasattr(self.encoder, "num_register_tokens"):
            self.encoder.num_register_tokens = 0
        self.mask_neighbor_size = mask_neighbor_size

    def forward(self, x):
        x = self.encoder.prepare_tokens(x)
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                if i in self.encoder_require_grad_layer:
                    x = blk(x)
                else:
                    with torch.no_grad():
                        x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)
        side = int(
            math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens)
        )

        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens :, :] for e in en_list]

        x = self.fuse_feature(en_list)
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        if self.mask_neighbor_size > 0:
            attn_mask = self.generate_mask(side, x.device)
        else:
            attn_mask = None

        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [
            self.fuse_feature([en_list[idx] for idx in idxs])
            for idxs in self.fuse_layer_encoder
        ]
        de = [
            self.fuse_feature([de_list[idx] for idx in idxs])
            for idxs in self.fuse_layer_decoder
        ]

        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + self.encoder.num_register_tokens :, :] for e in en]
            de = [d[:, 1 + self.encoder.num_register_tokens :, :] for d in de]

        en = [
            e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous()
            for e in en
        ]
        de = [
            d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous()
            for d in de
        ]
        return en, de

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)

    def generate_mask(self, feature_size, device="cuda"):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        h, w = feature_size, feature_size
        hm, wm = self.mask_neighbor_size, self.mask_neighbor_size
        mask = torch.ones(h, w, h, w, device=device)
        for idx_h1 in range(h):
            for idx_w1 in range(w):
                idx_h2_start = max(idx_h1 - hm // 2, 0)
                idx_h2_end = min(idx_h1 + hm // 2 + 1, h)
                idx_w2_start = max(idx_w1 - wm // 2, 0)
                idx_w2_end = min(idx_w1 + wm // 2 + 1, w)
                mask[
                    idx_h1, idx_w1, idx_h2_start:idx_h2_end, idx_w2_start:idx_w2_end
                ] = 0
        mask = mask.view(h * w, h * w)
        if self.remove_class_token:
            return mask
        mask_all = torch.ones(
            h * w + 1 + self.encoder.num_register_tokens,
            h * w + 1 + self.encoder.num_register_tokens,
            device=device,
        )
        mask_all[
            1 + self.encoder.num_register_tokens :,
            1 + self.encoder.num_register_tokens :,
        ] = mask
        return mask_all


class ViTill_test(nn.Module):
    def __init__(
        self,
        encoder: MultiEncoder,
        bottleneck,
        decoder,
        encoder_names=["dino", "clip"],
        mid_channels=64,
        out_channels=768,
        fuse_layer_decoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
        mask_neighbor_size=0,
        remove_class_token=False,
        encoder_require_grad_layer=[],
    ) -> None:
        super(ViTill_test, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.encoder_names = encoder_names
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer

        if not hasattr(self.encoder, "num_register_tokens"):
            self.encoder.num_register_tokens = 0
        self.mask_neighbor_size = mask_neighbor_size

        # Create feature fuser for multi-modal fusion
        modality_channels = {name: 768 for name in encoder_names}
        fusion_config = FusionConfig(
            modality_channels=modality_channels,
            mid_channels=mid_channels,
            out_channels=out_channels,
            fusion_type="light_transformer",
            use_modality_embedding=True,
        )
        self.fuser = GroupWiseFeatureFuser(fusion_config)

    def forward(self, x):
        outputs = self.encoder(x)

        # Extract target features from all encoders
        features_to_fuse = {}
        for name in self.encoder_names:
            if name in outputs:
                features_to_fuse[name] = outputs[name]["target"]

        # Fuse multi-modal features using Transformer
        fused_target = self.fuser(features_to_fuse)

        # Use fused features for encoding
        # Keep original encoder features for skip connections (if using multiple encoders)
        if len(self.encoder_names) == 1:
            en = outputs[self.encoder_names[0]]["fused_group_feats"]
        else:
            en = outputs[self.encoder_names[0]]["fused_group_feats"]

        x = fused_target.reshape(
            fused_target.shape[0], fused_target.shape[1], -1
        ).permute(0, 2, 1)

        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        attn_mask = None

        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x)
            de_list.append(x)
        de_list = de_list[::-1]

        de = [
            self.fuse_feature([de_list[idx] for idx in idxs])
            for idxs in self.fuse_layer_decoder
        ]
        side = en[0].shape[2]

        de = [
            d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous()
            for d in de
        ]
        return en, de

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)


class ViTillCat(nn.Module):
    def __init__(
        self,
        encoder,
        bottleneck,
        decoder,
        target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
        fuse_layer_encoder=[1, 3, 5, 7],
        mask_neighbor_size=0,
        remove_class_token=False,
        encoder_require_grad_layer=[],
    ) -> None:
        super(ViTillCat, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer

        if not hasattr(self.encoder, "num_register_tokens"):
            self.encoder.num_register_tokens = 0
        self.mask_neighbor_size = mask_neighbor_size

    def forward(self, x):
        x = self.encoder.prepare_tokens(x)
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                if i in self.encoder_require_grad_layer:
                    x = blk(x)
                else:
                    with torch.no_grad():
                        x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)
        side = int(
            math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens)
        )

        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens :, :] for e in en_list]

        x = self.fuse_feature(en_list)
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        for i, blk in enumerate(self.decoder):
            x = blk(x)

        en = [torch.cat([en_list[idx] for idx in self.fuse_layer_encoder], dim=2)]
        de = [x]

        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + self.encoder.num_register_tokens :, :] for e in en]
            de = [d[:, 1 + self.encoder.num_register_tokens :, :] for d in de]

        en = [
            e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous()
            for e in en
        ]
        de = [
            d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous()
            for d in de
        ]
        return en, de

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)


class ViTAD(nn.Module):
    def __init__(
        self,
        encoder,
        bottleneck,
        decoder,
        target_layers=[2, 5, 8, 11],
        fuse_layer_encoder=[0, 1, 2],
        fuse_layer_decoder=[2, 5, 8],
        mask_neighbor_size=0,
        remove_class_token=False,
    ) -> None:
        super(ViTAD, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token

        if not hasattr(self.encoder, "num_register_tokens"):
            self.encoder.num_register_tokens = 0
        self.mask_neighbor_size = mask_neighbor_size

    def forward(self, x):
        x = self.encoder.prepare_tokens(x)
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                with torch.no_grad():
                    x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)
        side = int(
            math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens)
        )

        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens :, :] for e in en_list]
            x = x[:, 1 + self.encoder.num_register_tokens :, :]

        # x = torch.cat(en_list, dim=2)
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        if self.mask_neighbor_size > 0:
            attn_mask = self.generate_mask(side, x.device)
        else:
            attn_mask = None

        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x, attn_mask=attn_mask)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [en_list[idx] for idx in self.fuse_layer_encoder]
        de = [de_list[idx] for idx in self.fuse_layer_decoder]

        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + self.encoder.num_register_tokens :, :] for e in en]
            de = [d[:, 1 + self.encoder.num_register_tokens :, :] for d in de]

        en = [
            e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous()
            for e in en
        ]
        de = [
            d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous()
            for d in de
        ]
        return en, de


class ViTillv2(nn.Module):
    def __init__(
        self, encoder, bottleneck, decoder, target_layers=[2, 3, 4, 5, 6, 7]
    ) -> None:
        super(ViTillv2, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        if not hasattr(self.encoder, "num_register_tokens"):
            self.encoder.num_register_tokens = 0

    def forward(self, x):
        x = self.encoder.prepare_tokens(x)
        en = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                with torch.no_grad():
                    x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en.append(x)

        x = self.fuse_feature(en)
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        de = []
        for i, blk in enumerate(self.decoder):
            x = blk(x)
            de.append(x)

        side = int(math.sqrt(x.shape[1]))

        en = [e[:, self.encoder.num_register_tokens + 1 :, :] for e in en]
        de = [d[:, self.encoder.num_register_tokens + 1 :, :] for d in de]

        en = [
            e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous()
            for e in en
        ]
        de = [
            d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous()
            for d in de
        ]

        return en[::-1], de

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)


class ViTillv3(nn.Module):
    def __init__(
        self,
        teacher,
        student,
        target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
        fuse_dropout=0.0,
    ) -> None:
        super(ViTillv3, self).__init__()
        self.teacher = teacher
        self.student = student
        if fuse_dropout > 0:
            self.fuse_dropout = nn.Dropout(fuse_dropout)
        else:
            self.fuse_dropout = nn.Identity()
        self.target_layers = target_layers
        if not hasattr(self.teacher, "num_register_tokens"):
            self.teacher.num_register_tokens = 0

    def forward(self, x):
        with torch.no_grad():
            patch = self.teacher.prepare_tokens(x)
            x = patch
            en = []
            for i, blk in enumerate(self.teacher.blocks):
                if i <= self.target_layers[-1]:
                    x = blk(x)
                else:
                    continue
                if i in self.target_layers:
                    en.append(x)
            en = self.fuse_feature(en, fuse_dropout=False)

        x = patch
        de = []
        for i, blk in enumerate(self.student):
            x = blk(x)
            if i in self.target_layers:
                de.append(x)
        de = self.fuse_feature(de, fuse_dropout=False)

        en = en[:, 1 + self.teacher.num_register_tokens :, :]
        de = de[:, 1 + self.teacher.num_register_tokens :, :]
        side = int(math.sqrt(en.shape[1]))

        en = en.permute(0, 2, 1).reshape([x.shape[0], -1, side, side])
        de = de.permute(0, 2, 1).reshape([x.shape[0], -1, side, side])
        return [en.contiguous()], [de.contiguous()]

    def fuse_feature(self, feat_list, fuse_dropout=False):
        if fuse_dropout:
            feat = torch.stack(feat_list, dim=1)
            feat = self.fuse_dropout(feat).mean(dim=1)
            return feat
        else:
            return torch.stack(feat_list, dim=1).mean(dim=1)


class ReContrast(nn.Module):
    def __init__(
        self,
        encoder,
        encoder_freeze,
        bottleneck,
        decoder,
    ) -> None:
        super(ReContrast, self).__init__()
        self.encoder = encoder
        self.encoder.layer4 = None
        self.encoder.fc = None

        self.encoder_freeze = encoder_freeze
        self.encoder_freeze.layer4 = None
        self.encoder_freeze.fc = None

        self.bottleneck = bottleneck
        self.decoder = decoder

    def forward(self, x):
        en = self.encoder(x)
        with torch.no_grad():
            en_freeze = self.encoder_freeze(x)
        en_2 = [torch.cat([a, b], dim=0) for a, b in zip(en, en_freeze)]
        de = self.decoder(self.bottleneck(en_2))
        de = [a.chunk(dim=0, chunks=2) for a in de]
        de = [de[0][0], de[1][0], de[2][0], de[3][1], de[4][1], de[5][1]]
        return en_freeze + en, de

    def train(self, mode=True, encoder_bn_train=True):
        self.training = mode
        if mode is True:
            if encoder_bn_train:
                self.encoder.train(True)
            else:
                self.encoder.train(False)
            self.encoder_freeze.train(False)  # the frozen encoder is eval()
            self.bottleneck.train(True)
            self.decoder.train(True)
        else:
            self.encoder.train(False)
            self.encoder_freeze.train(False)
            self.bottleneck.train(False)
            self.decoder.train(False)
        return self


def update_moving_average(ma_model, current_model, momentum=0.99):
    for current_params, ma_params in zip(
        current_model.parameters(), ma_model.parameters()
    ):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = update_average(old_weight, up_weight)

    for current_buffers, ma_buffers in zip(current_model.buffers(), ma_model.buffers()):
        old_buffer, up_buffer = ma_buffers.data, current_buffers.data
        ma_buffers.data = update_average(old_buffer, up_buffer, momentum)


def update_average(old, new, momentum=0.99):
    if old is None:
        return new
    return old * momentum + (1 - momentum) * new


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)
