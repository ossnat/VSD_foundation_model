# src/models/heads/mae_decoder_3d.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MAEDecoder3D(nn.Module):
    """
    3D Decoder for Video Masked Autoencoder.
    Reconstructs masked 3D videos from encoded spatiotemporal feature maps.
    
    Architecture: 5 stages of transposed 3D convolutions
    - 512 → 256 → 128 → 64 → 32 → 1 channels
    - Upsampling: 8× temporal, 32× spatial
    """
    def __init__(self, in_channels=512, out_channels=1, hidden_dim=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Channel progression: 512 → 256 → 128 → 64 → 32 → 1
        channels = [in_channels, 256, 128, 64, 32, out_channels]
        
        layers = []
        for i in range(len(channels) - 1):
            # For temporal upsampling, we need to handle it carefully
            # First few layers upsample temporally, later ones focus on spatial
            if i < 2:  # First 2 stages: upsample both temporal and spatial
                kernel_size = (4, 4, 4)
                stride = (2, 2, 2)
                padding = (1, 1, 1)
            else:  # Remaining stages: mainly spatial upsampling
                kernel_size = (3, 4, 4)
                stride = (1, 2, 2)  # Keep temporal same or slightly upsample
                padding = (1, 1, 1)
            
            layers.append(
                nn.ConvTranspose3d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False
                )
            )
            if i < len(channels) - 2:  # No BatchNorm/ReLU after final layer
                layers.append(nn.BatchNorm3d(channels[i + 1]))
                layers.append(nn.ReLU(inplace=True))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor, target_size=None) -> torch.Tensor:
        """
        Args:
            features: Encoded spatiotemporal feature maps of shape (B, 512, T', H', W')
            target_size: Optional (T, H, W) target size for exact size matching
        
        Returns:
            Reconstructed video of shape (B, 1, T, H, W)
        """
        reconstruction = self.decoder(features)  # (B, 1, T', H', W')
        
        # If target_size is provided, interpolate to exact size
        if target_size is not None:
            T, H, W = target_size
            reconstruction = F.interpolate(reconstruction, size=(T, H, W), mode='trilinear', align_corners=False)
        
        return reconstruction
