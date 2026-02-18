# ==================================
# File: src/training/trainer.py
# ==================================

import os
import torch
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
except Exception:  # matplotlib might not be available in some environments
    plt = None


class Trainer:
    def __init__(self, model, logger, cfg, device, plot_loss: bool = False, debug_train: bool = False):
        self.model = model
        self.logger = logger
        self.cfg = cfg
        self.device = device
        self.plot_loss = plot_loss
        self.debug_train = debug_train
        self.opt = AdamW(
            model.parameters(),
            lr=cfg.get("lr", 1e-4),
            weight_decay=cfg.get("weight_decay", 0.05),
        )
        self.scaler = GradScaler()
        os.makedirs(cfg.get("ckpt_dir", "checkpoints"), exist_ok=True)

    def fit(self, train_loader, val_loader):
        global_step = 0
        loss_buffer = []
        plot_steps = []
        plot_losses = []
        fig = None
        ax = None
        line = None

        if self.plot_loss and plt is None:
            print("Plotting disabled: matplotlib is not available.")
            self.plot_loss = False
        epochs = self.cfg.get("epochs", 10)
        for epoch in range(epochs):
            self.model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                self.opt.zero_grad()
                with autocast():
                    if self.debug_train:
                        loss = self._forward_and_loss_debug(batch)
                    else:
                        loss = self._forward_and_loss(batch)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()

                pbar.set_postfix({"loss": float(loss.item())})
                self.logger.log_scalar("train/loss", float(loss.item()), global_step)
                loss_buffer.append(float(loss.item()))
                if self.plot_loss and (global_step + 1) % 10 == 0:
                    recent_mean = sum(loss_buffer[-10:]) / 10.0
                    plot_steps.append(global_step + 1)
                    plot_losses.append(recent_mean)
                    fig, ax, line = self._update_loss_plot(plot_steps, plot_losses, fig, ax, line)
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
            # DINO style (dict with crops + metadata)
            if "crops" in batch:
                crops = [b.to(self.device) for b in batch["crops"]]
                out = self.model(crops)
                return out if torch.is_tensor(out) else out["loss"]

            # MAE style batch
            if "video_masked" in batch and "video_target" in batch and "mask" in batch:
                mae_batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                             for k, v in batch.items()}
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

    def _forward_and_loss_debug(self, batch):
        """Forward + loss with NaN/Inf checks on inputs/outputs of all layers."""
        handles = []
        error_message = {"msg": None, "sample": None}

        def format_sample(b_idx):
            if not isinstance(batch, dict) or b_idx is None:
                return None
            try:
                monkey = batch.get("monkey", None)
                date = batch.get("date", None)
                condition = batch.get("condition", None)
                if monkey is None or date is None or condition is None:
                    return None
                return f"Sample {b_idx} {monkey[b_idx]} {date[b_idx]} {condition[b_idx]}"
            except Exception:
                return None

        def find_bad_sample(tensor):
            if tensor.dim() == 0:
                return None
            # Assume batch dimension is 0
            finite = torch.isfinite(tensor)
            if finite.all():
                return None
            # Identify first batch index with any non-finite
            bad_per_sample = ~finite.view(finite.shape[0], -1).all(dim=1)
            if bad_per_sample.any():
                return int(bad_per_sample.nonzero(as_tuple=False)[0].item())
            return None

        def check_tensor(tensor, label):
            if torch.is_tensor(tensor) and not torch.isfinite(tensor).all():
                error_message["msg"] = f"Non-finite values in {label}"
                b_idx = find_bad_sample(tensor)
                if b_idx is not None:
                    error_message["sample"] = format_sample(b_idx)
                return True
            return False

        def hook_fn(module, inputs, output):
            name = module.__class__.__name__
            # Check inputs
            if isinstance(inputs, (list, tuple)):
                for i, inp in enumerate(inputs):
                    if check_tensor(inp, f"{name} input[{i}]"):
                        raise RuntimeError(error_message["msg"])
            else:
                if check_tensor(inputs, f"{name} input"):
                    raise RuntimeError(error_message["msg"])
            # Check outputs
            if isinstance(output, (list, tuple)):
                for i, out in enumerate(output):
                    if check_tensor(out, f"{name} output[{i}]"):
                        raise RuntimeError(error_message["msg"])
            else:
                if check_tensor(output, f"{name} output"):
                    raise RuntimeError(error_message["msg"])

        for module in self.model.modules():
            handles.append(module.register_forward_hook(hook_fn))

        try:
            # Check initial input before forward
            if isinstance(batch, dict) and "video_masked" in batch:
                _ = check_tensor(batch["video_masked"], "input video_masked")
                if error_message["msg"]:
                    raise RuntimeError(error_message["msg"])
            elif torch.is_tensor(batch):
                _ = check_tensor(batch, "input batch")
                if error_message["msg"]:
                    raise RuntimeError(error_message["msg"])

            return self._forward_and_loss(batch)
        except RuntimeError as exc:
            if error_message["msg"] is None:
                error_message["msg"] = str(exc)
            if error_message["sample"] is not None:
                print(f"Stopping training: {error_message['msg']} ({error_message['sample']})")
            else:
                print(f"Stopping training: {error_message['msg']}")
            raise
        finally:
            for h in handles:
                h.remove()

    def _update_loss_plot(self, steps, losses, fig, ax, line):
        if plt is None:
            return fig, ax, line
        if fig is None or ax is None or line is None:
            fig, ax = plt.subplots(figsize=(8, 4))
            line, = ax.plot(steps, losses, marker="o")
            ax.set_title("Train Loss (mean of every 10 iterations)")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)
        else:
            line.set_data(steps, losses)
            ax.relim()
            ax.autoscale_view()
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        return fig, ax, line
