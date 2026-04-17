# src/models/systems/mae_system.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_system import BaseSystem
from src.training.mae_masked_metrics import aggregate_batch_metrics, masked_ssim_per_sample


class MAELoss(nn.Module):
    """
    Reconstruction loss for Masked Autoencoders.
    Computes reconstruction losses on masked patches only, optionally restricted
    to a centered spatial region (circle or square).

    Supported loss types:
      - mse:          masked MSE
      - l1:           masked L1
      - l1_mse:       alpha * masked_L1 + (1-alpha) * masked_MSE
      - l1_ssim:      alpha * (1 - masked_ssim) + (1-alpha) * masked_L1
      - mse_ssim:     alpha * (1 - masked_ssim) + (1-alpha) * masked_MSE
    """
    def __init__(self, normalize: bool = True,
                 crop_loss: str = None,
                 crop_loss_radius: int = 30,
                 loss_type: str = "mse",
                 alpha: float = 0.84,
                 ssim_window_size: int = 11,
                 ssim_sigma: float = 1.5):
        super().__init__()
        self.normalize = normalize
        self.crop_loss = crop_loss            # None | "circle" | "square"
        self.crop_loss_radius = crop_loss_radius
        self.loss_type = str(loss_type).lower().strip()
        self.alpha = float(alpha)
        self.ssim_window_size = int(ssim_window_size)
        self.ssim_sigma = float(ssim_sigma)
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
        eps = 1e-8
        if self.loss_type not in ("mse", "l1", "l1_mse", "l1_ssim", "mse_ssim"):
            raise ValueError(
                f"Unknown loss_type={self.loss_type!r}. "
                "Expected one of: mse, l1, l1_mse, l1_ssim, mse_ssim."
            )

        # weight = 1 where the patch was masked (mask==0 means masked)
        weight = 1.0 - mask

        if self.crop_loss is not None:
            H, W = target.shape[-2], target.shape[-1]
            region = self._build_region_mask(H, W, target.device)
            weight = weight * region

        # Pixel fidelity terms
        if self.loss_type in ("mse", "mse_ssim", "l1_mse"):
            mse_per_element = F.mse_loss(reconstruction, target, reduction="none")
            mse_loss = (mse_per_element * weight).sum() / (weight.sum() + eps)
        else:
            mse_loss = None
        if self.loss_type in ("l1", "l1_ssim", "l1_mse"):
            l1_per_element = F.l1_loss(reconstruction, target, reduction="none")
            l1_loss = (l1_per_element * weight).sum() / (weight.sum() + eps)
        else:
            l1_loss = None

        if self.loss_type == "mse":
            return mse_loss
        if self.loss_type == "l1":
            return l1_loss
        if self.loss_type == "l1_mse":
            alpha = max(0.0, min(1.0, self.alpha))
            return alpha * l1_loss + (1.0 - alpha) * mse_loss

        if self.loss_type == "mse_ssim":
            pixel_loss = mse_loss
        elif self.loss_type == "l1_ssim":
            pixel_loss = l1_loss
        else:
            # Defensive: should be unreachable due to loss_type check above.
            raise RuntimeError(f"Unhandled loss_type={self.loss_type!r}")

        # SSIM term (only for *_ssim variants)
        # masked_ssim_per_sample expects weight where >0 includes pixels to consider.
        # It returns SSIM per-sample averaged over masked pixels only.
        ssim_ps = masked_ssim_per_sample(
            recon_norm=reconstruction,
            target_norm=target,
            weight=weight,
            window_size=self.ssim_window_size,
            sigma=self.ssim_sigma,
        )  # (B,)
        ssim_loss = (1.0 - ssim_ps).mean()

        # Zhao et al. style mixing: alpha*(1-SSIM) + (1-alpha)*pixel_loss
        alpha = max(0.0, min(1.0, self.alpha))
        loss = alpha * ssim_loss + (1.0 - alpha) * pixel_loss
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
            loss_type=loss_cfg.get("loss_type", "mse"),
            alpha=loss_cfg.get("alpha", 0.84),
            ssim_window_size=loss_cfg.get("ssim_window_size", 11),
            ssim_sigma=loss_cfg.get("ssim_sigma", 1.5),
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
        
        # Compute additional metrics for logging (z-score recon and target so MSE/PSNR are scale-invariant)
        mse_per_sample = None
        r2_per_sample = None
        ss_tot_per_sample = None
        ssim_per_sample = None
        with torch.no_grad():
            eps = 1e-8
            # Z-score reconstruction and target per batch so we compare two normalized populations
            r_mean, r_std = reconstruction.mean(), reconstruction.std() + eps
            t_mean, t_std = video_target.mean(), video_target.std() + eps
            recon_norm = (reconstruction - r_mean) / r_std
            target_norm = (video_target - t_mean) / t_std

            # Overall MSE (on normalized values; good → near 0)
            mse_overall = F.mse_loss(recon_norm, target_norm)

            # MSE on masked regions only
            loss_per_elem = (recon_norm - target_norm).pow(2)
            weight = 1.0 - mask
            if self.loss_fn.crop_loss is not None:
                H, W = video_target.shape[-2], video_target.shape[-1]
                region = self.loss_fn._build_region_mask(H, W, video_target.device)
                if weight.dim() == 5:
                    region = region.unsqueeze(2).expand_as(weight)
                weight = weight * region

            mse_masked = (loss_per_elem * weight).sum() / (weight.sum() + eps)
            mse_visible = (loss_per_elem * mask).sum() / (mask.sum() + eps)

            agg = None
            if batch.get("_return_per_sample_metrics") or batch.get("_extended_test_metrics"):
                agg = aggregate_batch_metrics(recon_norm, target_norm, weight)
                metrics_extra = {
                    "r2_masked": agg["r2_masked"].item(),
                    "ssim_masked": agg["ssim_masked"].item(),
                }
            else:
                metrics_extra = {}

            if batch.get("_return_per_sample_metrics") and agg is not None:
                mse_per_sample = agg["mse_masked_per_sample"]
                r2_per_sample = agg["r2_masked_per_sample"]
                ss_tot_per_sample = agg.get("ss_tot_masked_per_sample", None)
                ssim_per_sample = agg["ssim_masked_per_sample"]

        metrics = {
            "mse_overall": mse_overall.item(),
            "mse_masked": mse_masked.item(),
            "mse_visible": mse_visible.item(),
        }
        metrics.update(metrics_extra)

        out = {
            "loss": loss,
            "metrics": metrics
        }
        if mse_per_sample is not None:
            out["mse_per_sample"] = mse_per_sample
        if r2_per_sample is not None:
            out["r2_per_sample"] = r2_per_sample
        if ss_tot_per_sample is not None:
            out["ss_tot_per_sample"] = ss_tot_per_sample
        if ssim_per_sample is not None:
            out["ssim_per_sample"] = ssim_per_sample
        return out
    
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
