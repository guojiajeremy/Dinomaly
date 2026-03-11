import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.multi_view.decoder.decoder_adapters import DecoderAdapter


def main() -> None:
    torch.manual_seed(0)

    b = 2
    c = 768
    num_register = 4
    h = w = 37
    n = 1 + num_register + h * w

    adapter = DecoderAdapter(num_register_tokens=num_register, num_conv=2).eval()

    x = torch.randn(b, n, c, dtype=torch.float32)

    with torch.no_grad():
        y1 = adapter(x)
        y2 = adapter(x)

    assert y1.shape == (b, c, h, w), f"Unexpected output shape: {tuple(y1.shape)}"

    # Should be identical across repeated calls (conv built once).
    max_diff_repeat = (y1 - y2).abs().max().item()
    assert max_diff_repeat == 0.0, f"Repeat call mismatch: max_diff={max_diff_repeat}"

    # Prefix tokens should be ignored. If patch tokens are all zeros, output must be all zeros
    # (conv has no bias; GELU(0)=0).
    x_prefix_only = torch.zeros(b, n, c, dtype=torch.float32)
    x_prefix_only[:, : 1 + num_register, :] = torch.randn(b, 1 + num_register, c)

    with torch.no_grad():
        y_prefix_only = adapter(x_prefix_only)

    max_abs = y_prefix_only.abs().max().item()
    assert max_abs == 0.0, f"Prefix leakage detected: max_abs={max_abs}"

    print("OK")
    print(f"input_tokens: (B,N,C)=({b},{n},{c}) = cls(1)+reg({num_register})+patch({h}*{w})")
    print(f"output: {tuple(y1.shape)}")


if __name__ == "__main__":
    main()
