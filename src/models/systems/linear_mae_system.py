# src/models/systems/linear_mae_system.py
"""
Global linear masked reconstruction baseline: concat(masked frame, mask) -> Linear -> full frame.

Ridge-style L2 is applied via Trainer AdamW `weight_decay`. Optional L1 on weights
implements an elastic-net-style penalty: loss = MAELoss + linear_l1_penalty * ||W||_1.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_system import BaseSystem
from .mae_system import MAELoss
from src.training.mae_masked_metrics import aggregate_batch_metrics, flatten_pearson_r_per_sample


def _prepare_mae_tensors(
    video_masked: torch.Tensor,
    video_target: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Match MAESystem 2D/5D squeezing and mask resize."""
    is_2d = False
    if len(video_target.shape) == 5 and video_target.shape[2] == 1:
        is_2d = True
        video_masked = video_masked.squeeze(2)
        video_target = video_target.squeeze(2)
        if len(mask.shape) == 5:
            mask = mask.squeeze(2)
        if len(mask.shape) == 4 and mask.shape[1] == 1:
            H, W = video_target.shape[2], video_target.shape[3]
            mask = F.interpolate(mask, size=(H, W), mode="nearest")
    elif len(video_target.shape) == 5:
        if len(mask.shape) == 5 and mask.shape[1] == 1:
            T, H, W = video_target.shape[2], video_target.shape[3], video_target.shape[4]
            mask = F.interpolate(mask, size=(T, H, W), mode="nearest")
    return video_masked, video_target, mask, is_2d


class LinearMAESystem(BaseSystem):
    """
    Single Linear map: vec([video_masked, mask_broadcast_to_C]) -> vec(reconstruction).

    For 2D MAE baseline, use with clip_length=1 and same batches as MAESystem.
    """

    def __init__(self, num_channels: int, height: int, width: int, config: Dict[str, Any]):
        super().__init__(config)
        self.num_channels = int(num_channels)
        self.height = int(height)
        self.width = int(width)
        in_features = (self.num_channels + 1) * self.height * self.width
        out_features = self.num_channels * self.height * self.width
        self.proj = nn.Linear(in_features, out_features)

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
        self.linear_l1_penalty = float(loss_cfg.get("linear_l1_penalty", 0.0))

        self.lr = config.get("training", {}).get("lr", 1e-3)
        self.weight_decay = config.get("training", {}).get("weight_decay", 1e-2)

    def _features(self, video_masked: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Concat channels: (B, C, H, W) + (B, 1, H, W) -> (B, (C+1)*H*W)."""
        x = torch.cat([video_masked, mask], dim=1)
        return x.reshape(x.shape[0], -1)

    def _project(self, video_masked: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, C, H, W = video_masked.shape
        if (H, W) != (self.height, self.width):
            raise ValueError(
                f"LinearMAESystem expected spatial ({self.height}, {self.width}), "
                f"got ({H}, {W}). Set linear_spatial_hw in config to match trimmed video."
            )
        if C != self.num_channels:
            raise ValueError(
                f"LinearMAESystem expected C={self.num_channels}, got {C}. "
                "Fix `channels` / linear_spatial_hw in config."
            )
        feat = self._features(video_masked, mask)
        out = self.proj(feat)
        return out.view(B, C, H, W)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        video_masked = batch["video_masked"]
        video_target = batch["video_target"]
        mask = batch["mask"]

        video_masked, video_target, mask, is_2d = _prepare_mae_tensors(
            video_masked, video_target, mask
        )

        if not is_2d and len(video_target.shape) != 4:
            raise NotImplementedError(
                "LinearMAESystem currently supports 2D (B, C, H, W) or (B, C, 1, H, W) only."
            )

        reconstruction = self._project(video_masked, mask)
        base_loss = self.loss_fn(reconstruction, video_target, mask)
        if self.linear_l1_penalty > 0.0:
            l1w = self.proj.weight.abs().sum()
            loss = base_loss + self.linear_l1_penalty * l1w
        else:
            loss = base_loss

        mse_per_sample = None
        r2_per_sample = None
        ss_tot_per_sample = None
        ssim_per_sample = None
        pearson_per_sample = None
        with torch.no_grad():
            eps = 1e-8
            r_mean, r_std = reconstruction.mean(), reconstruction.std() + eps
            t_mean, t_std = video_target.mean(), video_target.std() + eps
            recon_norm = (reconstruction - r_mean) / r_std
            target_norm = (video_target - t_mean) / t_std

            mse_overall = F.mse_loss(recon_norm, target_norm)
            loss_per_elem = (recon_norm - target_norm).pow(2)
            weight = 1.0 - mask
            if self.loss_fn.crop_loss is not None:
                H, W = video_target.shape[-2], video_target.shape[-1]
                region = self.loss_fn._build_region_mask(H, W, video_target.device)
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
                pearson_per_sample = flatten_pearson_r_per_sample(reconstruction, video_target)

        metrics = {
            "mse_overall": mse_overall.item(),
            "mse_masked": mse_masked.item(),
            "mse_visible": mse_visible.item(),
        }
        metrics.update(metrics_extra)

        out: Dict[str, Any] = {"loss": loss, "metrics": metrics}
        if mse_per_sample is not None:
            out["mse_per_sample"] = mse_per_sample
        if r2_per_sample is not None:
            out["r2_per_sample"] = r2_per_sample
        if ss_tot_per_sample is not None:
            out["ss_tot_per_sample"] = ss_tot_per_sample
        if ssim_per_sample is not None:
            out["ssim_per_sample"] = ssim_per_sample
        if pearson_per_sample is not None:
            out["pearson_per_sample"] = pearson_per_sample
        return out

    def reconstruct_for_vis(
        self, batch: Dict[str, Any], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Used by vis_test_reconstruction when present."""
        video_masked = batch["video_masked"].to(device)
        video_target = batch["video_target"].to(device)
        mask = batch["mask"].to(device)
        video_masked, video_target, mask, _ = _prepare_mae_tensors(
            video_masked, video_target, mask
        )
        with torch.no_grad():
            reconstruction = self._project(video_masked, mask)
        return reconstruction, video_target, video_masked, mask

    def get_optimizer(self, lr=None, weight_decay=None):
        import torch.optim as optim

        lr = lr or self.lr
        wd = weight_decay or self.weight_decay
        return optim.AdamW(self.proj.parameters(), lr=lr, weight_decay=wd)
