# src/models/heads/mae_decoder_3d.py
import torch
import torch.nn as nn


class MAEDecoder3D(nn.Module):
    """
    Lightweight decoder for 3D Video Masked Autoencoder (VideoMAE).
    Upsamples spatiotemporal feature maps back to original video resolution.
    """
    def __init__(self,
                 in_channels: int = 512,      # from R3D-18 encoder
                 out_channels: int = 1,       # VSD is single channel
                 hidden_dim: int = 256):
        super().__init__()
        
        # Decoder: upsample from (B, 512, T/8, H/32, W/32) → (B, 1, T, H, W)
        self.decoder = nn.Sequential(
            # Stage 1: 512 → 256, upsample 2x spatiotemporally
            nn.ConvTranspose3d(in_channels, hidden_dim, kernel_size=(2,4,4), stride=(2,2,2), padding=(0,1,1)),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
            
            # Stage 2: 256 → 128, upsample 2x
            nn.ConvTranspose3d(hidden_dim, hidden_dim // 2, kernel_size=(2,4,4), stride=(2,2,2), padding=(0,1,1)),
            nn.BatchNorm3d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            
            # Stage 3: 128 → 64, upsample 2x
            nn.ConvTranspose3d(hidden_dim // 2, hidden_dim // 4, kernel_size=(2,4,4), stride=(2,2,2), padding=(0,1,1)),
            nn.BatchNorm3d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            
            # Stage 4: 64 → 32, upsample 2x
            nn.ConvTranspose3d(hidden_dim // 4, hidden_dim // 8, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1)),
            nn.BatchNorm3d(hidden_dim // 8),
            nn.ReLU(inplace=True),
            
            # Stage 5: 32 → 1, final upsample
            nn.ConvTranspose3d(hidden_dim // 8, out_channels, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1)),
        )
    
    def forward(self, x: torch.Tensor, target_size: tuple = None) -> torch.Tensor:
        """
        Args:
            x: Encoded spatiotemporal feature maps (B, 512, T', H', W')
            target_size: Optional (T, H, W) tuple to resize output to match input
        Returns:
            Reconstructed video (B, 1, T, H, W)
        """
        out = self.decoder(x)
        if target_size is not None:
            # Interpolate to match target size if provided
            out = torch.nn.functional.interpolate(
                out, size=target_size, mode='trilinear', align_corners=False
            )
        return out
