# src/models/backbone/mae_backbone_2d_CNN.py
import torch
import torch.nn as nn


class MAEShallowCNNBackbone(nn.Module):
    """
    Shallow 2D CNN encoder for Masked Autoencoder (MAE).
    Five Conv2d blocks (each with BatchNorm + ReLU, stride 2) produce
    feature maps with the same channel dimension as ResNet18 (512),
    making it a drop-in replacement for MAEResNet18Backbone.
    """
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.feature_dim = 512
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Masked image tensor of shape (B, C, H, W)
        Returns:
            Feature maps of shape (B, 512, H/32, W/32)
        """
        return self.encoder(x)
