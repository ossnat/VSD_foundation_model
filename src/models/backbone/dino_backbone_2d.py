# src/models/backbone/backbone_2d.py
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Backbone(nn.Module):
    """
    2D ResNet18 feature extractor for image-based foundation models (e.g., DINO-2D).
    Returns flattened feature vectors instead of classification logits.
    """
    def __init__(self, pretrained: bool = True, projection_dim: int = None):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)

        # Remove the classification head (fc layer)
        self.encoder = nn.Sequential(*list(model.children())[:-1])  # up to global avgpool

        # Optional projection layer (for embedding dimension reduction)
        in_dim = model.fc.in_features
        if projection_dim:
            self.projection = nn.Linear(in_dim, projection_dim)
        else:
            self.projection = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C, H, W)
        Returns:
            Feature embeddings of shape (B, D)
        """
        features = self.encoder(x)          # (B, 512, 1, 1)
        features = features.flatten(1)      # (B, 512)
        if self.projection:
            features = self.projection(features)
        return features
