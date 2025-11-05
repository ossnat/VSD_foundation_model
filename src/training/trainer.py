# ==================================
# File: src/training/trainer.py
# ==================================

import os
import torch
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm


class Trainer:
    def __init__(self, model, logger, cfg, device):
        self.model = model
        self.logger = logger
        self.cfg = cfg
        self.device = device
        self.opt = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        self.scaler = GradScaler()
        os.makedirs(cfg.get("ckpt_dir", "checkpoints"), exist_ok=True)

    def fit(self, train_loader, val_loader):
        global_step = 0
        for epoch in range(self.cfg["epochs"]):
            self.model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.cfg['epochs']}")
            for batch in pbar:
                self.opt.zero_grad()
                with autocast():
                    loss = self._forward_and_loss(batch)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()

                pbar.set_postfix({"loss": float(loss.item())})
                self.logger.log_scalar("train/loss", float(loss.item()), global_step)
                global_step += 1

            # Validation (optional)
            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    val_losses = []
                    for batch in val_loader:
                        val_loss = self._forward_and_loss(batch)
                        val_losses.append(float(val_loss.item()))
                    if val_losses:
                        self.logger.log_scalar("val/loss", sum(val_losses)/len(val_losses), epoch)

            # Save checkpoint each epoch
            ckpt_path = os.path.join(self.cfg.get("ckpt_dir","checkpoints"), f"epoch_{epoch+1}.pt")
            torch.save(self.model.state_dict(), ckpt_path)

    def _forward_and_loss(self, batch):
        """Route batch to model depending on task/system.
        Supports:
          - MAE (2D/3D): dict with keys {video_masked, video_target, mask}
          - Generic video: dict with key {video}
          - DINO: list/tuple of crops (tensors)
        Returns a scalar loss tensor on self.device.
        """
        # DINO style: list/tuple of crops
        if isinstance(batch, (list, tuple)):
            crops = [b.to(self.device) for b in batch]
            out = self.model(crops)
            # DINOSystem returns a scalar loss
            return out if torch.is_tensor(out) else out["loss"]

        # Dict-style batches
        if isinstance(batch, dict):
            # MAE style batch
            if "video_masked" in batch and "video_target" in batch and "mask" in batch:
                mae_batch = {k: v.to(self.device) for k, v in batch.items()}
                out = self.model(mae_batch)
                # MAESystem returns dict with loss
                return out["loss"] if isinstance(out, dict) and "loss" in out else out

            # Generic video batch
            if "video" in batch:
                vid = batch["video"].to(self.device)
                out = self.model(vid)
                return out["loss"] if isinstance(out, dict) and "loss" in out else out

        # Fallback: assume tensor
        if torch.is_tensor(batch):
            out = self.model(batch.to(self.device))
            return out["loss"] if isinstance(out, dict) and "loss" in out else out

        raise ValueError("Unsupported batch format for Trainer._forward_and_loss")
