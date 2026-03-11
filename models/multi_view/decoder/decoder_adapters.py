import math
import torch.nn as nn


class DecoderAdapter(nn.Module):
    def __init__(self, num_register_tokens=0, num_conv=2):
        super().__init__()
        self.prefix = 1 + num_register_tokens
        self.num_conv = num_conv
        self.conv = None

    def _build(self, c):
        layers = []
        for _ in range(self.num_conv):
            layers += [
                nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False),
                nn.Conv2d(c, c, 1, bias=False),
                nn.GELU(),
            ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        b, n, c = x.shape
        x = x[:, self.prefix:, :]
        p = x.shape[1]
        h = int(math.isqrt(p))
        if h * h != p:
            raise RuntimeError(f"num_patches={p} not square")

        x = x.transpose(1, 2).reshape(b, c, h, h)

        if self.conv is None:
            self._build(c)

        return self.conv(x)