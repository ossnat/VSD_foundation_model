# src/models/heads/mae_decoder_2d.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MAEDecoder2D(nn.Module):
    """
    2D Decoder for Masked Autoencoder.
    Reconstructs masked 2D images from encoded feature maps.
    
    Architecture: 5 stages of transposed 2D convolutions
    - 512 → 256 → 128 → 64 → 32 → 1 channels
    - Each stage: ConvTranspose2d (stride=2) + BatchNorm + ReLU
    - Total upsampling: 32× (2× per stage × 5 stages)
    """
    def __init__(self, in_channels=512, out_channels=1, hidden_dim=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Channel progression: 512 → 256 → 128 → 64 → 32 → 1
        channels = [in_channels, 256, 128, 64, 32, out_channels]
        
        layers = []
        for i in range(len(channels) - 1):
            layers.append(
                nn.ConvTranspose2d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                )
            )
            if i < len(channels) - 2:  # No BatchNorm/ReLU after final layer
                layers.append(nn.BatchNorm2d(channels[i + 1]))
                layers.append(nn.ReLU(inplace=True))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor, target_size=None) -> torch.Tensor:
        """
        Args:
            features: Encoded feature maps of shape (B, 512, H/32, W/32)
            target_size: Optional (H, W) target size for exact size matching
        
        Returns:
            Reconstructed image of shape (B, 1, H, W)
        """
        reconstruction = self.decoder(features)  # (B, 1, H', W')
        
        # If target_size is provided, interpolate to exact size
        if target_size is not None:
            H, W = target_size
            reconstruction = F.interpolate(reconstruction, size=(H, W), mode='bilinear', align_corners=False)
        
        return reconstruction
