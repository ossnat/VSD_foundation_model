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
    def __init__(self, pretrained: bool = False, in_channels: int = 1):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        
        # Handle single-channel input (e.g., grayscale images)
        if in_channels == 1:
            # Replace first conv layer to accept 1 channel instead of 3
            original_conv = model.conv1
            self.input_conv = nn.Conv2d(
                in_channels=1,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            # Initialize weights: average the RGB channels if pretrained
            if pretrained and original_conv.weight is not None:
                with torch.no_grad():
                    self.input_conv.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
                    if self.input_conv.bias is not None and original_conv.bias is not None:
                        self.input_conv.bias.data = original_conv.bias.data.clone()
        else:
            self.input_conv = None
        
        # Remove global pooling and classification head
        # Keep feature extraction layers only (skip first conv if we replaced it)
        if in_channels == 1:
            encoder_layers = list(model.children())[1:-2]  # Skip first conv, before avgpool
        else:
            encoder_layers = list(model.children())[:-2]  # up to layer4, before avgpool
        self.encoder = nn.Sequential(*encoder_layers)
        
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
        if self.input_conv is not None:
            x = self.input_conv(x)
        features = self.encoder(x)  # (B, 512, H/32, W/32)
        return features
