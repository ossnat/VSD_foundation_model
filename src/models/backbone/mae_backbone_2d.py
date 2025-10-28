# src/models/backbone/mae_backbone_2d.py
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class MAEResNet18Backbone(nn.Module):
    """
    2D ResNet18 encoder for Masked Autoencoder (MAE).
    Returns spatial feature maps instead of flattened vectors,
    suitable for reconstruction via decoder.
    """
    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        
        # Remove global pooling and classification head
        # Keep feature extraction layers only
        self.encoder = nn.Sequential(*list(model.children())[:-2])  # up to layer4, before avgpool
        
        # Feature map will be (B, 512, H/32, W/32) for standard ResNet18
        self.feature_dim = 512
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Masked image tensor of shape (B, C, H, W)
        Returns:
            Feature maps of shape (B, feature_dim, H', W')
            where H' = H/32, W' = W/32 for ResNet18
        """
        features = self.encoder(x)  # (B, 512, H/32, W/32)
        return features
