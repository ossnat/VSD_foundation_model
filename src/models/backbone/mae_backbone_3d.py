# src/models/backbone/mae_backbone_3d.py
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class MAER3D18Backbone(nn.Module):
    """
    3D R3D-18 encoder for Video Masked Autoencoder (VideoMAE).
    Returns spatiotemporal feature maps for reconstruction.
    """
    def __init__(self, pretrained: bool = False, in_channels: int = 1):
        super().__init__()
        weights = R3D_18_Weights.DEFAULT if pretrained else None
        model = r3d_18(weights=weights)
        
        # Handle single-channel input (e.g., grayscale videos)
        # Replace first conv layer to accept 1 channel instead of 3
        if in_channels == 1:
            # Create new first conv layer for single channel
            original_stem = list(model.children())[0]
            # Extract the actual conv layer from BasicStem
            if hasattr(original_stem, 'conv'):
                original_conv = original_stem.conv
            else:
                # If it's a direct Conv3d, use it directly
                original_conv = original_stem
            
            out_channels = original_conv.out_channels if hasattr(original_conv, 'out_channels') else 64
            
            self.input_conv = nn.Conv3d(
                in_channels=1,
                out_channels=out_channels,
                kernel_size=original_conv.kernel_size if hasattr(original_conv, 'kernel_size') else (3, 7, 7),
                stride=original_conv.stride if hasattr(original_conv, 'stride') else (1, 2, 2),
                padding=original_conv.padding if hasattr(original_conv, 'padding') else (1, 3, 3),
                bias=original_conv.bias is not None if hasattr(original_conv, 'bias') else True
            )
            # Initialize weights: average the RGB channels if pretrained
            if pretrained and hasattr(original_conv, 'weight') and original_conv.weight is not None:
                with torch.no_grad():
                    self.input_conv.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
                    if self.input_conv.bias is not None and hasattr(original_conv, 'bias'):
                        self.input_conv.bias.data = original_conv.bias.data.clone()
        else:
            self.input_conv = None
        
        # Remove global pooling and classification head
        # Keep convolutional feature extraction only
        encoder_layers = list(model.children())[1:-2]  # Skip first conv and last two layers
        self.encoder = nn.Sequential(*encoder_layers)
        
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
        if self.input_conv is not None:
            x = self.input_conv(x)
        features = self.encoder(x)  # (B, 512, T/8, H/32, W/32) approximately
        return features
