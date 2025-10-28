# src/models/backbone/backbone_3d.py
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class R3D18Backbone(nn.Module):
    """
    3D R3D-18 feature extractor for spatiotemporal video encoding.
    Provides (B, D) embeddings suitable for DINO-3D or MAE-3D.
    """
    def __init__(self, pretrained: bool = True, projection_dim: int = None):
        super().__init__()
        weights = R3D_18_Weights.DEFAULT if pretrained else None
        model = r3d_18(weights=weights)

        # Remove classifier head (fc)
        self.encoder = nn.Sequential(*list(model.children())[:-1])  # up to last pooling
        self.in_dim = model.fc.in_features

        # Optional projection MLP
        if projection_dim:
            self.projection = nn.Linear(self.in_dim, projection_dim)
        else:
            self.projection = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C, T, H, W)
        Returns:
            Feature embeddings of shape (B, D)
        """
        feats = self.encoder(x)             # (B, 512, 1, 1, 1)
        feats = feats.flatten(1)            # (B, 512)
        if self.projection:
            feats = self.projection(feats)
        return feats
