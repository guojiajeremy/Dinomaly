import argparse
import sys
from pathlib import Path

import torch

# Ensure repo root is on sys.path so `import models...` works regardless of CWD.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.vit_encoder import load as load_vit_encoder
from models.multi_view.encoder.dino import DinoExtractor


def feats_bchw_to_patch_tokens(feat: torch.Tensor) -> torch.Tensor:
    """(B,C,H,W) -> (B, H*W, C)"""
    if feat.dim() != 4:
        raise ValueError(f"Expected (B,C,H,W), got {tuple(feat.shape)}")
    b, c, h, w = feat.shape
    return feat.reshape(b, c, h * w).transpose(1, 2).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="dinov2reg_vit_base_14")
    parser.add_argument("--h", type=int, default=518)
    parser.add_argument("--w", type=int, default=518)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--tol", type=float, default=1e-6, help="Tolerance for |diff| sum and max.")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")

    device = torch.device(args.device)

    torch.manual_seed(0)

    model = load_vit_encoder(args.encoder)
    model.eval().to(device)

    # deterministic input; ensure divisible by patch size (14 for vitb14)
    x = torch.randn(1, 3, args.h, args.w, device=device, dtype=torch.float32)

    # 1) 拿到两个 list
    # NOTE: This assumes model.blocks is not chunked (true for this project's default DINOv2 configs).
    n_layers = len(getattr(model, "blocks", []))
    extractor = DinoExtractor(model, n=n_layers, norm=True).eval().to(device)

    with torch.no_grad():
        feats_list = extractor(x)
        inter_list = model.get_intermediate_layers(x, n=n_layers)

    if len(feats_list) != len(inter_list):
        raise AssertionError(f"Length mismatch: extractor={len(feats_list)} get_intermediate_layers={len(inter_list)}")

    # dinov2_vitb14_reg4: CLS(1) + register(4) => prefix=5
    num_prefix_tokens = 1 + int(getattr(model, "num_register_tokens", 0))
    if num_prefix_tokens != 5:
        print(f"[WARN] num_prefix_tokens={num_prefix_tokens}, expected 5 for reg4 models")

    print(f"Encoder: {args.encoder}")
    print(f"Input: {tuple(x.shape)} device={device} layers={len(inter_list)} prefix={num_prefix_tokens}")

    # 2) 逐个 list 比较：对 get_intermediate_layers 去掉前 5 个 token；展平 extractor 输出
    # 3) 二者作差：检查差的 sum 是否接近 0
    for i, (feat_bchw, tokens_bnc) in enumerate(zip(feats_list, inter_list)):
        patch_tokens_ref = tokens_bnc[:, num_prefix_tokens:, :].contiguous()
        patch_tokens_ext = feats_bchw_to_patch_tokens(feat_bchw)

        if patch_tokens_ref.shape != patch_tokens_ext.shape:
            raise AssertionError(
                f"Shape mismatch at layer[{i}]: ref={tuple(patch_tokens_ref.shape)} ext={tuple(patch_tokens_ext.shape)}"
            )

        diff = patch_tokens_ref - patch_tokens_ext
        diff_sum = diff.sum().item()
        diff_abs_sum = diff.abs().sum().item()
        diff_abs_max = diff.abs().max().item()

        ok = (abs(diff_sum) <= args.tol) and (diff_abs_sum <= args.tol) and (diff_abs_max <= args.tol)

        print(
            f"layer[{i}]: diff_sum={diff_sum:.6g} diff_abs_sum={diff_abs_sum:.6g} diff_abs_max={diff_abs_max:.6g} ok={ok}"
        )

        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
