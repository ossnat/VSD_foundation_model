# src/models/systems/mae_system.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_system import BaseSystem


class MAELoss(nn.Module):
    """
    Reconstruction loss for Masked Autoencoders.
    Computes MSE only on masked patches, optionally restricted to a
    centered spatial region (circle or square).
    """
    def __init__(self, normalize: bool = True,
                 crop_loss: str = None,
                 crop_loss_radius: int = 30):
        super().__init__()
        self.normalize = normalize
        self.crop_loss = crop_loss            # None | "circle" | "square"
        self.crop_loss_radius = crop_loss_radius
        self._region_cache: dict = {}

    def _build_region_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Return a (1, 1, H, W) float mask that is 1 inside the region, 0 outside."""
        key = (H, W, self.crop_loss, self.crop_loss_radius, device)
        if key in self._region_cache:
            return self._region_cache[key]

        cy, cx = H / 2.0, W / 2.0
        r = self.crop_loss_radius

        ys = torch.arange(H, device=device).float()
        xs = torch.arange(W, device=device).float()
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        if self.crop_loss == "circle":
            region = ((yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2).float()
        elif self.crop_loss == "square":
            region = ((yy - cy).abs() <= r).float() * ((xx - cx).abs() <= r).float()
        else:
            region = torch.ones(H, W, device=device)

        region = region.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        self._region_cache[key] = region
        return region

    def forward(self, reconstruction: torch.Tensor, 
                target: torch.Tensor, 
                mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            reconstruction: Model output (B, C, T?, H, W)
            target: Ground truth unmasked video (B, C, T?, H, W)
            mask: Binary mask (B, 1, T?, H, W) where 1=visible, 0=masked
        
        Returns:
            MSE loss computed only on masked patches (within the crop region
            when crop_loss is set).
        """
        loss_per_element = F.mse_loss(reconstruction, target, reduction='none')

        # weight = 1 where the patch was masked (mask==0 means masked)
        weight = 1 - mask

        if self.crop_loss is not None:
            H, W = target.shape[-2], target.shape[-1]
            region = self._build_region_mask(H, W, target.device)
            weight = weight * region

        masked_loss = loss_per_element * weight
        loss = masked_loss.sum() / (weight.sum() + 1e-8)
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
        loss_cfg = config.get("loss", {})
        self.loss_fn = MAELoss(
            normalize=loss_cfg.get("normalize", True),
            crop_loss=loss_cfg.get("crop_loss", None),
            crop_loss_radius=loss_cfg.get("crop_loss_radius", 30),
        )
        
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
            # Ensure 3D mask matches full (T, H, W) resolution: (B, 1, nPT, nPH, nPW) -> (B, 1, T, H, W)
            if len(mask.shape) == 5 and mask.shape[1] == 1:
                T, H, W = video_target.shape[2], video_target.shape[3], video_target.shape[4]
                mask = F.interpolate(mask, size=(T, H, W), mode='nearest')
        
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
