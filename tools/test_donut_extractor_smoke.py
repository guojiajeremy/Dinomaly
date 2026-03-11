import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.multi_view.encoder.donut import DonutExtractor


def main() -> None:
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="naver-clova-ix/donut-base")
    parser.add_argument("--h", type=int, default=512)
    parser.add_argument("--w", type=int, default=512)
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Override encoder.config.image_size (e.g. 518 518).",
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Disable any resizing inside DonutExtractor.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(REPO_ROOT / "backbones" / "weights" / "hf"),
        help="HuggingFace cache dir for model weights",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--tol", type=float, default=1e-6)
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    torch.manual_seed(0)

    extractor = DonutExtractor(
        model_name=args.model_name,
        trainable=False,
        do_resize=False,
        image_size=tuple(args.image_size) if args.image_size is not None else None,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    ).eval()

    pixel_values = torch.randn(1, 3, args.h, args.w, dtype=torch.float32)

    with torch.no_grad():
        feats = extractor(pixel_values)

        # Reference: direct encoder call
        pixel_values_ref = extractor._resize_if_needed(pixel_values)
        ref_out = extractor.encoder(
            pixel_values=pixel_values_ref,
            output_hidden_states=True,
            return_dict=True,
        )
        reshaped = getattr(ref_out, "reshaped_hidden_states", None)
        if reshaped is None or len(reshaped) == 0:
            raise AssertionError("encoder did not return reshaped_hidden_states")
        ref_feats = list(reshaped)[1:]

    if not isinstance(feats, list) or len(feats) == 0:
        raise AssertionError("Extractor returned empty outputs")

    if len(feats) != len(ref_feats):
        raise AssertionError(f"Length mismatch: extractor={len(feats)} ref={len(ref_feats)}")

    print(f"model={args.model_name}")
    print(f"input={tuple(pixel_values.shape)}")
    print(f"resized_input={tuple(pixel_values_ref.shape)}")
    print(f"do_resize={getattr(extractor, 'do_resize', None)}")
    print(f"encoder.config.image_size={getattr(extractor.encoder.config, 'image_size', None)}")
    num_layers = getattr(extractor.encoder.config, "num_layers", None)
    print(f"encoder.config.num_layers={num_layers} outputs={len(feats)}")

    prev_hw = None
    for i, (f, r) in enumerate(zip(feats, ref_feats)):
        if f.dim() != 4:
            raise AssertionError(f"Output at idx[{i}] is not BCHW: shape={tuple(f.shape)}")
        if not torch.isfinite(f).all():
            raise AssertionError(f"Non-finite values in output idx[{i}]")

        if r.shape != f.shape:
            raise AssertionError(
                f"Shape mismatch at idx[{i}]: extractor={tuple(f.shape)} ref={tuple(r.shape)}"
            )

        diff = (r - f)
        diff_sum = diff.sum().item()
        diff_abs_sum = diff.abs().sum().item()
        diff_abs_max = diff.abs().max().item()

        ok = (abs(diff_sum) <= args.tol) and (diff_abs_sum <= args.tol) and (diff_abs_max <= args.tol)

        b, c, h, w = f.shape
        print(
            f"out[{i}]: shape={tuple(f.shape)} mean={f.float().mean().item():.6g} std={f.float().std().item():.6g} "
            f"diff_sum={diff_sum:.6g} diff_abs_sum={diff_abs_sum:.6g} diff_abs_max={diff_abs_max:.6g} ok={ok}"
        )

        if not ok:
            raise SystemExit(1)

        if prev_hw is not None:
            # stage features usually downsample, so spatial dims should be non-increasing
            if h > prev_hw[0] or w > prev_hw[1]:
                print(f"[WARN] spatial size increased from {prev_hw} to {(h, w)}")
        prev_hw = (h, w)


if __name__ == "__main__":
    main()
