# src/models/systems/mae_system.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_system import BaseSystem


class MAELoss(nn.Module):
    """
    Reconstruction loss for Masked Autoencoders.
    Computes MSE only on masked patches.
    """
    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
    
    def forward(self, reconstruction: torch.Tensor, 
                target: torch.Tensor, 
                mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            reconstruction: Model output (B, C, T?, H, W)
            target: Ground truth unmasked video (B, C, T?, H, W)
            mask: Binary mask (B, 1, T?, H, W) where 1=visible, 0=masked
        
        Returns:
            MSE loss computed only on masked patches
        """
        # Compute per-element loss
        loss_per_element = F.mse_loss(reconstruction, target, reduction='none')  # (B, C, T?, H, W)
        
        # Apply mask: keep only masked regions (mask == 0)
        masked_loss = loss_per_element * (1 - mask)
        
        # Average over masked patches
        loss = masked_loss.sum() / ((1 - mask).sum() + 1e-8)
        
        return loss


class MAESystem(BaseSystem):
    """
    Masked Autoencoder system combining encoder + decoder.
    Supports both 2D images and 3D videos through config.
    Inherits from BaseSystem for consistent interface.
    """
    
    def __init__(self, encoder, decoder, config):
        """
        Args:
            encoder: Backbone (MAEResNet18Backbone or MAER3D18Backbone)
            decoder: Head (MAEDecoder2D or MAEDecoder3D)
            config: Configuration dict with training hyperparameters
        """
        super().__init__(config)
        
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = MAELoss(normalize=config.get("loss", {}).get("normalize", True))
        
        # Training config
        self.lr = config.get("training", {}).get("lr", 1e-4)
        self.weight_decay = config.get("training", {}).get("weight_decay", 0.05)
    
    def forward(self, batch):
        """
        Forward pass for MAE training.
        
        Args:
            batch: Dict containing:
                - "video_masked": Masked input (B, C, T?, H, W)
                - "video_target": Original unmasked video (B, C, T?, H, W)
                - "mask": Binary mask (B, 1, T?, H, W)
        
        Returns:
            Dict with:
                - "loss": Scalar reconstruction loss
                - "metrics": Dict of metrics (MSE, etc.)
        """
        video_masked = batch["video_masked"]
        video_target = batch["video_target"]
        mask = batch["mask"]
        
        # Handle dimension differences: 2D images may come as (B, C, T=1, H, W) or (B, C, H, W)
        # Check if temporal dimension exists and is 1 (2D case)
        is_2d = False
        if len(video_target.shape) == 5 and video_target.shape[2] == 1:
            # Squeeze temporal dimension for 2D: (B, C, 1, H, W) -> (B, C, H, W)
            is_2d = True
            video_masked = video_masked.squeeze(2)  # (B, C, H, W)
            video_target = video_target.squeeze(2)  # (B, C, H, W)
            # Mask from dataset may be (B, 1, nPT, nPH, nPW) or (B, 1, nPH, nPW)
            if len(mask.shape) == 5:
                mask = mask.squeeze(2)  # Remove temporal patch dimension if present
            if len(mask.shape) == 4 and mask.shape[1] == 1:
                # Expand mask to match video: (B, 1, nPH, nPW) -> (B, 1, H, W)
                H, W = video_target.shape[2], video_target.shape[3]
                # Use interpolate to expand mask to full resolution
                mask = F.interpolate(mask, size=(H, W), mode='nearest')
        
        # Encode masked input
        features = self.encoder(video_masked)
        
        # Decode to reconstruct - pass target size to ensure correct output dimensions
        if is_2d or len(video_target.shape) == 4:  # 2D: (B, C, H, W)
            target_size = (video_target.shape[2], video_target.shape[3])
            reconstruction = self.decoder(features, target_size=target_size)
        else:  # 3D: (B, C, T, H, W)
            target_size = (video_target.shape[2], video_target.shape[3], video_target.shape[4])
            reconstruction = self.decoder(features, target_size=target_size)
        
        # Compute loss only on masked patches
        loss = self.loss_fn(reconstruction, video_target, mask)
        
        # Compute additional metrics for logging
        with torch.no_grad():
            # Overall MSE
            mse_overall = F.mse_loss(reconstruction, video_target)
            
            # MSE on masked regions only
            mse_masked = F.mse_loss(reconstruction * (1 - mask), 
                                     video_target * (1 - mask))
            
            # MSE on visible regions (should be low if model doesn't overwrite)
            mse_visible = F.mse_loss(reconstruction * mask, 
                                      video_target * mask)
        
        metrics = {
            "mse_overall": mse_overall.item(),
            "mse_masked": mse_masked.item(),
            "mse_visible": mse_visible.item(),
        }
        
        return {
            "loss": loss,
            "metrics": metrics
        }
    
    def get_optimizer(self, lr=None, weight_decay=None):
        """
        Creates optimizer for MAE training.
        """
        lr = lr or self.lr
        wd = weight_decay or self.weight_decay
        
        optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=lr,
            weight_decay=wd
        )
        return optimizer
