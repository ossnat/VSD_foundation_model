# src/models/heads/mae_decoder_2d.py
import torch
import torch.nn as nn


class MAEDecoder2D(nn.Module):
    """
    Lightweight decoder for 2D Masked Autoencoder.
    Upsamples feature maps from encoder back to original image resolution.
    """
    def __init__(self, 
                 in_channels: int = 512,      # from ResNet18 encoder
                 out_channels: int = 1,       # VSD is single channel
                 hidden_dim: int = 256):
        super().__init__()
        
        # Decoder: upsample from (B, 512, H/32, W/32) → (B, 1, H, W)
        self.decoder = nn.Sequential(
            # Stage 1: 512 → 256, upsample 2x
            nn.ConvTranspose2d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            
            # Stage 2: 256 → 128, upsample 2x
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            
            # Stage 3: 128 → 64, upsample 2x
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            
            # Stage 4: 64 → 32, upsample 2x
            nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 8),
            nn.ReLU(inplace=True),
            
            # Stage 5: 32 → 1, upsample 2x (total 32x upsampling)
            nn.ConvTranspose2d(hidden_dim // 8, out_channels, kernel_size=4, stride=2, padding=1),
        )
    
    def forward(self, x: torch.Tensor, target_size: tuple = None) -> torch.Tensor:
        """
        Args:
            x: Encoded feature maps (B, 512, H/32, W/32)
            target_size: Optional (H, W) tuple to resize output to match input
        Returns:
            Reconstructed image (B, 1, H, W)
        """
        out = self.decoder(x)
        if target_size is not None:
            # Interpolate to match target size if provided
            out = torch.nn.functional.interpolate(
                out, size=target_size, mode='bilinear', align_corners=False
            )
        return out
