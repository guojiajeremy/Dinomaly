import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.multi_view.encoder.clip import ClipExtractor


def bchw_to_patch_tokens(feat: torch.Tensor) -> torch.Tensor:
    """(B,C,H,W) -> (B, H*W, C)"""
    b, c, h, w = feat.shape
    return feat.reshape(b, c, h * w).transpose(1, 2).contiguous()


@torch.no_grad()
def manual_feats_all_layers(visual, x: torch.Tensor) -> list[torch.Tensor]:
    """Manual forward that mirrors OpenCLIP VisionTransformer tokenization + per-block outputs.

    Returns list of (B,C,H,W) per block, patches only (no CLS).
    """
    patch = visual.conv1(x)  # (B,C,H',W')
    patch_hw = (int(patch.shape[2]), int(patch.shape[3]))

    B, C, H, W = patch.shape
    tokens = patch.reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)

    cls = visual.class_embedding.to(tokens.dtype)
    cls = cls + torch.zeros(B, 1, C, dtype=tokens.dtype, device=tokens.device)
    tokens = torch.cat([cls, tokens], dim=1)  # (B, 1+HW, C)

    pos = visual.positional_embedding.to(tokens.dtype)  # (N,C)
    if pos.shape[0] != tokens.shape[1]:
        raise RuntimeError(f"positional_embedding length mismatch: expected {tokens.shape[1]}, got {pos.shape[0]}")

    tokens = visual.ln_pre(tokens + pos)

    # transformer blocks expect (N,B,C)
    tokens = tokens.permute(1, 0, 2)

    outs: list[torch.Tensor] = []
    for blk in visual.transformer.resblocks:
        tokens = blk(tokens)
        out_bnc = tokens.permute(1, 0, 2).contiguous()  # (B,N,C)
        patch_tokens = out_bnc[:, 1:, :]
        ph, pw = patch_hw
        out_bchw = patch_tokens.transpose(1, 2).reshape(B, C, ph, pw).contiguous()
        outs.append(out_bchw)

    return outs


def main() -> None:
    import open_clip

    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ViT-B-16")
    parser.add_argument(
        "--pretrained",
        default="openai",
        help="Pretrained tag for open_clip (e.g., openai, laion2b_s34b_b88k).",
    )
    parser.add_argument(
        "--force-quick-gelu",
        action="store_true",
        help="Force QuickGELU in the model config (recommended for --pretrained openai).",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(REPO_ROOT / "backbones" / "weights" / "open_clip"),
        help="Local cache dir for pretrained weights (avoids re-downloading).",
    )
    args = parser.parse_args()

    torch.manual_seed(0)

    # Load pretrained weights (will download if not cached).
    # For OpenAI pretrained weights, QuickGELU is the expected activation.
    force_quick_gelu = args.force_quick_gelu or (args.pretrained == "openai")

    os.makedirs(args.cache_dir, exist_ok=True)
    model, _, _ = open_clip.create_model_and_transforms(
        args.model,
        pretrained=args.pretrained,
        force_quick_gelu=force_quick_gelu,
        cache_dir=args.cache_dir,
    )
    visual = model.visual
    visual.eval()

    image_size = getattr(visual, "image_size", (518,518))
    if isinstance(image_size, int):
        h = w = int(image_size)
    else:
        h, w = int(image_size[0]), int(image_size[1])

    x = torch.randn(1, 3, h, w, dtype=torch.float32)

    n_layers = len(visual.transformer.resblocks)
    extractor = ClipExtractor(visual, n=n_layers, norm=False).eval()

    with torch.no_grad():
        feats_list = extractor(x)
        manual_list = manual_feats_all_layers(visual, x)

    if len(feats_list) != len(manual_list):
        raise AssertionError(f"Length mismatch: extractor={len(feats_list)} manual={len(manual_list)}")

    print(
        f"open_clip={getattr(open_clip, '__version__', 'unknown')} model={args.model} pretrained={args.pretrained} "
        f"force_quick_gelu={force_quick_gelu} cache_dir={args.cache_dir}"
    )
    print(f"input={tuple(x.shape)} layers={n_layers}")

    tol = 1e-6

    for i, (feat, ref) in enumerate(zip(feats_list, manual_list)):
        if feat.shape != ref.shape:
            raise AssertionError(f"Shape mismatch at layer[{i}]: extractor={tuple(feat.shape)} manual={tuple(ref.shape)}")

        diff = bchw_to_patch_tokens(ref) - bchw_to_patch_tokens(feat)
        diff_sum = diff.sum().item()
        diff_abs_sum = diff.abs().sum().item()
        diff_abs_max = diff.abs().max().item()

        ok = (abs(diff_sum) <= tol) and (diff_abs_sum <= tol) and (diff_abs_max <= tol)
        print(
            f"layer[{i}]: shape={tuple(feat.shape)} diff_sum={diff_sum:.6g} diff_abs_sum={diff_abs_sum:.6g} diff_abs_max={diff_abs_max:.6g} ok={ok}"
        )
        if not ok:
            raise SystemExit(1)

    # Basic numeric sanity
    stacked = torch.stack([f.float().mean() for f in feats_list])
    if not torch.isfinite(stacked).all():
        raise AssertionError("Non-finite values detected in extractor outputs")


if __name__ == "__main__":
    main()
