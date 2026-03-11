import math
import torch.nn as nn


class DecoderAdapter(nn.Module):
    def __init__(self, embed_dim, num_conv=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_conv = num_conv
        conv_list = []
        for i in range(num_conv):
            if i == num_conv - 1:
                conv_list.append(
                    nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding="same")
                )
            else:
                conv_list.append(
                    nn.Sequential(
                        nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding="same"),
                        nn.GELU(),
                    )
                )
        self.conv = nn.Sequential(
            *conv_list
        )
        


    def forward(self, x):
        b, n, c = x.shape
        h = w = int(math.sqrt(n))
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        return x