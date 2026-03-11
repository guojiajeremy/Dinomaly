import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.multi_view.encoder.resnet import ResNetExtractor


def forward_stem(model, x: torch.Tensor) -> torch.Tensor:
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    return x


def main() -> None:
    import argparse

    from torchvision.models import wide_resnet50_2

    try:
        # torchvision >= 0.13
        from torchvision.models import Wide_ResNet50_2_Weights

        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
        model = wide_resnet50_2(weights=weights)
        weights_name = str(weights)
    except Exception:
        # fallback for older torchvision
        model = wide_resnet50_2(pretrained=True)
        weights_name = "pretrained=True"

    parser = argparse.ArgumentParser()
    parser.add_argument("--h", type=int, default=224)
    parser.add_argument("--w", type=int, default=224)
    parser.add_argument("--tol", type=float, default=1e-6)
    args = parser.parse_args()

    torch.manual_seed(0)

    model.eval()

    extractor = ResNetExtractor(model, n=4).eval()

    x = torch.randn(1, 3, args.h, args.w, dtype=torch.float32)

    with torch.no_grad():
        feats = extractor(x)

        y = forward_stem(model, x)
        ref1 = model.layer1(y)
        ref2 = model.layer2(ref1)
        ref3 = model.layer3(ref2)
        ref4 = model.layer4(ref3)
        refs = [ref1, ref2, ref3, ref4]

    if len(feats) != 4:
        raise AssertionError(f"Expected 4 outputs, got {len(feats)}")

    print(f"model=wide_resnet50_2 weights={weights_name}")
    print(f"input={tuple(x.shape)}")

    for i, (out, ref) in enumerate(zip(feats, refs)):
        if out.shape != ref.shape:
            raise AssertionError(f"Shape mismatch at stage[{i}]: extractor={tuple(out.shape)} ref={tuple(ref.shape)}")

        diff = ref - out
        diff_sum = diff.sum().item()
        diff_abs_sum = diff.abs().sum().item()
        diff_abs_max = diff.abs().max().item()

        ok = (abs(diff_sum) <= args.tol) and (diff_abs_sum <= args.tol) and (diff_abs_max <= args.tol)
        print(
            f"stage[{i}]: shape={tuple(out.shape)} diff_sum={diff_sum:.6g} diff_abs_sum={diff_abs_sum:.6g} diff_abs_max={diff_abs_max:.6g} ok={ok}"
        )
        if not ok:
            raise SystemExit(1)

        if not torch.isfinite(out).all():
            raise AssertionError(f"Non-finite values in extractor output at stage[{i}]")


if __name__ == "__main__":
    main()
