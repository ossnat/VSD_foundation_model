# ==================================
# File: src/training/trainer.py
# ==================================

import os
import torch
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from src.utils.visualization import save_reconstruction_grid


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
                vid = batch["video"].to(self.device)  # (B,C,T,H,W)
                self.opt.zero_grad()
                with autocast():
                    out = self.model(vid)
                    loss = out["loss"]
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()

                pbar.set_postfix({"loss": float(loss.item())})
                self.logger.log_scalar("train/loss", float(loss.item()), global_step)
                global_step += 1

            # Validation (reconstruction preview)
            self.model.eval()
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    vid = batch["video"].to(self.device)
                    out = self.model(vid)
                    if i == 0:
                        # Save a small grid: original, masked, recon
                        recon = self.model.reconstruct_video(out["recon_patches"], out["shape"]).float().cpu()
                        target = self.model.reconstruct_video(out["target_patches"], out["shape"]).float().cpu()
                        mask = out["mask"].float().cpu()
                        save_reconstruction_grid(target, recon, mask,
                                                 fname=os.path.join(self.cfg.get("log_dir","logs"), f"recon_epoch_{epoch+1}.png"))
                        break

            # Save checkpoint each epoch
            ckpt_path = os.path.join(self.cfg.get("ckpt_dir","checkpoints"), f"epoch_{epoch+1}.pt")
            torch.save(self.model.state_dict(), ckpt_path)
