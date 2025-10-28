# src/models/backbone/mae_backbone_3d.py
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class MAER3D18Backbone(nn.Module):
    """
    3D R3D-18 encoder for Video Masked Autoencoder (VideoMAE).
    Returns spatiotemporal feature maps for reconstruction.
    """
    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = R3D_18_Weights.DEFAULT if pretrained else None
        model = r3d_18(weights=weights)
        
        # Remove global pooling and classification head
        # Keep convolutional feature extraction only
        self.encoder = nn.Sequential(*list(model.children())[:-2])  # up to layer4, before avgpool
        
        # Feature map will be (B, 512, T', H', W') for R3D-18
        self.feature_dim = 512
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Masked video tensor of shape (B, C, T, H, W)
        Returns:
            Spatiotemporal feature maps of shape (B, feature_dim, T', H', W')
            where T', H', W' are downsampled dimensions
        """
        features = self.encoder(x)  # (B, 512, T/8, H/32, W/32) approximately
        return features
