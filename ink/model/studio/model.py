import math

import torch
from torch import nn
from torch.nn import functional as F


def norm(channels):
    return nn.GroupNorm(math.gcd(32, channels), channels)


class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout):
        super().__init__()
        self.branch = nn.Sequential(
            norm(channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            norm(channels),
            nn.SiLU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.branch(x)


class DeepResidualUNet(nn.Module):
    def __init__(self):
        super().__init__()
        widths = [48, 96, 192, 384, 640]
        counts = [2, 3, 4, 5, 6]
        drops = [0.02, 0.05, 0.10, 0.15, 0.20]
        decoder_counts = [3, 3, 2, 2]
        self.stem = nn.Conv3d(1, widths[0], 3, padding=1, bias=False)
        self.enc = nn.ModuleList(
            [
                nn.Sequential(*[ResidualBlock(c, p) for _ in range(n)])
                for c, n, p in zip(widths, counts, drops)
            ]
        )
        self.down = nn.ModuleList(
            [
                nn.Sequential(
                    norm(widths[i]),
                    nn.SiLU(),
                    nn.Conv3d(
                        widths[i],
                        widths[i + 1],
                        3,
                        stride=(1 if i == 0 else 2, 2, 2),
                        padding=1,
                        bias=False,
                    ),
                )
                for i in range(4)
            ]
        )
        self.up = nn.ModuleList(
            [nn.Conv3d(widths[i + 1], widths[i], 1, bias=False) for i in range(3, -1, -1)]
        )
        self.fuse = nn.ModuleList(
            [
                nn.Sequential(
                    norm(widths[i] * 2),
                    nn.SiLU(),
                    nn.Conv3d(widths[i] * 2, widths[i], 1, bias=False),
                )
                for i in range(3, -1, -1)
            ]
        )
        self.dec = nn.ModuleList(
            [
                nn.Sequential(*[ResidualBlock(widths[i], drops[i]) for _ in range(n)])
                for i, n in zip(range(3, -1, -1), decoder_counts)
            ]
        )
        self.head = nn.Sequential(norm(widths[0]), nn.SiLU(), nn.Conv3d(widths[0], 1, 1))
        self.depth_logits = nn.Parameter(torch.zeros(16))

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for i, stage in enumerate(self.enc):
            x = stage(x)
            if i < 4:
                skips.append(x)
                x = self.down[i](x)
        for up, fuse, stage, skip in zip(self.up, self.fuse, self.dec, reversed(skips)):
            x = up(F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False))
            x = stage(fuse(torch.cat([x, skip], dim=1)))
        weights = self.depth_logits.softmax(0)[None, None, :, None, None]
        return (self.head(x) * weights).sum(2)
