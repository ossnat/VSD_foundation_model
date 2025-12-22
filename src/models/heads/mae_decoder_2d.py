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
                 hidden_dim: int = 256,
                 final_activation: str = "tanh",  # "tanh" (default, helps prevent NaN), "relu", "sigmoid", or "none"
                 norm_type: str = "batch"):  # "batch", "layer", or "none"
        super().__init__()
        
        # Helper function to create normalization layer
        def make_norm(channels, norm_type):
            if norm_type == "batch":
                return nn.BatchNorm2d(channels)
            elif norm_type == "layer":
                # LayerNorm for 2D: normalize over (C, H, W) dimensions
                # We'll use GroupNorm with 1 group as an equivalent to LayerNorm for 2D
                return nn.GroupNorm(1, channels)  # 1 group = LayerNorm equivalent
            elif norm_type == "none":
                return nn.Identity()
            else:
                raise ValueError(f"Unknown norm_type: {norm_type}. Use 'batch', 'layer', or 'none'")
        
        # Decoder: upsample from (B, 512, H/32, W/32) → (B, 1, H, W)
        decoder_layers = [
            # Stage 1: 512 → 256, upsample 2x
            nn.ConvTranspose2d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            make_norm(hidden_dim, norm_type),
            nn.ReLU(inplace=True),
            
            # Stage 2: 256 → 128, upsample 2x
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            make_norm(hidden_dim // 2, norm_type),
            nn.ReLU(inplace=True),
            
            # Stage 3: 128 → 64, upsample 2x
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1),
            make_norm(hidden_dim // 4, norm_type),
            nn.ReLU(inplace=True),
            
            # Stage 4: 64 → 32, upsample 2x
            nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 8, kernel_size=4, stride=2, padding=1),
            make_norm(hidden_dim // 8, norm_type),
            nn.ReLU(inplace=True),
            
            # Stage 5: 32 → 1, upsample 2x (total 32x upsampling)
            nn.ConvTranspose2d(hidden_dim // 8, out_channels, kernel_size=4, stride=2, padding=1),
        ]
        
        # Add final activation if specified
        if final_activation is not None and final_activation.lower() != "none":
            if final_activation.lower() == "tanh":
                decoder_layers.append(nn.Tanh())
            elif final_activation.lower() == "relu":
                decoder_layers.append(nn.ReLU(inplace=True))
            elif final_activation.lower() == "sigmoid":
                decoder_layers.append(nn.Sigmoid())
            else:
                raise ValueError(f"Unknown final_activation: {final_activation}. Use 'tanh', 'relu', 'sigmoid', or None")
        
        self.decoder = nn.Sequential(*decoder_layers)
        self.final_activation = final_activation
        self.norm_type = norm_type
        
        # Initialize all ConvTranspose2d layers with He (Kaiming) initialization
        # This ensures consistent initialization: std = sqrt(2 / n_in)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize all ConvTranspose2d layers with He (Kaiming) initialization."""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                # He initialization for ConvTranspose2d: std = sqrt(2 / (fan_in))
                # fan_in = kernel_size * kernel_size * in_channels
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
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
