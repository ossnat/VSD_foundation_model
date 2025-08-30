# ==================================
# File: src/models/cnn3d.py
# ==================================

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=(3,3,3), s=(1,2,2), p=(1,1,1)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(c_in, c_out, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm3d(c_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)


class Simple3DEncoder(nn.Module):
    """Tiny 3D CNN encoder that outputs a spatiotemporal feature map and a pooled embedding."""
    def __init__(self, in_channels=1, dim=256):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, s=(1,2,2)),    # T,H/2,W/2
            ConvBlock(32, 64, s=(2,2,2)),             # T/2,H/4,W/4
            ConvBlock(64, 128, s=(2,2,2)),            # T/4,H/8,W/8
        )
        self.proj = nn.Conv3d(128, dim, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool3d((1,1,1))

    def forward(self, x):
        # x: (B,C,T,H,W)
        f = self.features(x)
        f = self.proj(f)
        pooled = self.pool(f).flatten(1)  # (B, dim)
        return f, pooled
