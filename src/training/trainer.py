# ==================================
# File: src/training/trainer.py
# ==================================

import json
import os
import math
from collections import defaultdict

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

        # Validation schedule: "epoch" (default) or "step"
        val_mode = self.cfg.get("val_mode", "epoch")
        val_every = self.cfg.get("val_every", 1)

        if self.plot_loss and plt is None:
            print("Plotting disabled: matplotlib is not available.")
            self.plot_loss = False
        epochs = self.cfg.get("epochs", 10)
        for epoch in range(epochs):
            self.model.train()
            train_losses_epoch = []
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

                loss_val = float(loss.item())
                train_losses_epoch.append(loss_val)
                pbar.set_postfix({"loss": loss_val})
                self.logger.log_scalar("train/loss", loss_val, global_step)
                loss_buffer.append(loss_val)
                if self.plot_loss and (global_step + 1) % 10 == 0:
                    recent_mean = sum(loss_buffer[-10:]) / 10.0
                    plot_steps.append(global_step + 1)
                    plot_losses.append(recent_mean)
                    fig, ax, line = self._update_loss_plot(plot_steps, plot_losses, fig, ax, line)
                global_step += 1

            mean_train_loss = sum(train_losses_epoch) / len(train_losses_epoch) if train_losses_epoch else 0.0
            self.logger.log_scalar("train/loss_epoch", mean_train_loss, epoch)
            print(f"  train/loss: {mean_train_loss:.4f}")

            # Validation (optional)
            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    val_losses = []
                    for batch in val_loader:
                        val_loss = self._forward_and_loss(batch)
                        val_losses.append(float(val_loss.item()))
                    if val_losses:
                        mean_val_loss = sum(val_losses) / len(val_losses)
                        self.logger.log_scalar("val/loss", mean_val_loss, epoch)
                        print(f"  val/loss: {mean_val_loss:.4f}")

            # Save checkpoint each epoch
            ckpt_path = os.path.join(self.cfg.get("ckpt_dir","checkpoints"), f"epoch_{epoch+1}.pt")
            torch.save(self.model.state_dict(), ckpt_path)

    def evaluate_metrics(self, loader, split_name: str = "train") -> dict:
        """
        Run the trained model on a dataloader and compute performance metrics.
        Model-agnostic: collects loss and any metrics returned by the model
        (e.g. MAE returns mse_overall, mse_masked; DINO could return different metrics).
        Adds common derived metrics (e.g. PSNR from MSE when available).
        Prints all metrics to screen and returns them as a dict.
        """
        self.model.eval()
        all_losses = []
        all_metrics = []  # list of dicts per batch

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating ({split_name})"):
                out = self._model_forward_for_metrics(batch)
                if out is None:
                    continue
                if torch.is_tensor(out):
                    all_losses.append(float(out.item()))
                    continue
                if isinstance(out, dict):
                    if "loss" in out:
                        all_losses.append(float(out["loss"].item()))
                    if "metrics" in out and isinstance(out["metrics"], dict):
                        all_metrics.append({k: float(v) if torch.is_tensor(v) else v for k, v in out["metrics"].items()})

        # Aggregate
        result = {}
        if all_losses:
            result["loss"] = sum(all_losses) / len(all_losses)
        if all_metrics:
            keys = all_metrics[0].keys()
            for k in keys:
                vals = [m[k] for m in all_metrics if k in m]
                if vals:
                    result[k] = sum(vals) / len(vals)

        # Derived metrics (modular: add MAE-specific or others here)
        if "mse_overall" in result and result["mse_overall"] > 0:
            # PSNR = 10 * log10(MAX^2 / MSE); assume signal in [0,1] or normalized so MAX^2=1
            result["psnr_db"] = 10.0 * math.log10(1.0 / result["mse_overall"] + 1e-10)
        if "mse_masked" in result and result["mse_masked"] > 0:
            result["psnr_masked_db"] = 10.0 * math.log10(1.0 / result["mse_masked"] + 1e-10)

        # Print to screen
        print(f"\n--- Metrics ({split_name}) ---")
        for k, v in sorted(result.items()):
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")
        print("---\n")
        return result

    def evaluate_metrics_over_time(self, loader, split_name: str = "test", save_dir: str = None) -> dict:
        """
        Evaluate reconstruction quality (MSE) over time on the test set.
        Aggregates per-clip MSE by clip start frame and returns mean (and optionally std)
        so you can see how performance varies from early to late clips (e.g. frames 30-34 vs 95-99).

        If save_dir is set (or cfg has results_dir), saves a plot and a JSON file of the metrics.
        Requires MAE-style batches with "video_masked", "video_target", "mask", and "start_frame".
        Use with shuffle=False so results are reproducible. Does not change training or existing eval.
        """
        self.model.eval()
        all_start_frames = []
        all_mse = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Eval over time ({split_name})"):
                if not isinstance(batch, dict) or "video_masked" not in batch or "video_target" not in batch:
                    continue
                if "start_frame" not in batch:
                    print("evaluate_metrics_over_time: batch missing 'start_frame'; skipping.")
                    continue
                batch_with_flag = {**batch, "_return_per_sample_metrics": True}
                out = self._model_forward_for_metrics(batch_with_flag)
                if out is None or not isinstance(out, dict) or "mse_per_sample" not in out:
                    continue
                start_frames = batch["start_frame"]
                if torch.is_tensor(start_frames):
                    start_frames = start_frames.cpu().tolist()
                else:
                    start_frames = list(start_frames)
                mse_list = out["mse_per_sample"].cpu().tolist()
                all_start_frames.extend(start_frames)
                all_mse.extend(mse_list)

        if not all_start_frames:
            print(f"evaluate_metrics_over_time: no samples collected for {split_name}.")
            return {}

        # Aggregate by start_frame
        by_start = defaultdict(list)
        for s, m in zip(all_start_frames, all_mse):
            by_start[s].append(m)

        result = {}
        for start_frame in sorted(by_start.keys()):
            vals = by_start[start_frame]
            mean_mse = sum(vals) / len(vals)
            variance = sum((x - mean_mse) ** 2 for x in vals) / len(vals) if len(vals) > 1 else 0.0
            std_mse = math.sqrt(variance)
            result[start_frame] = {
                "mean_mse": mean_mse,
                "std_mse": std_mse,
                "mean_rmse": math.sqrt(mean_mse),
                "count": len(vals),
            }

        # Print table
        print(f"\n--- Metrics over time ({split_name}) ---")
        print(f"{'start_frame':<12} {'mean_mse':<12} {'mean_rmse':<12} {'std_mse':<10} {'count':<8}")
        print("-" * 54)
        for start_frame in sorted(result.keys()):
            r = result[start_frame]
            print(f"{start_frame:<12} {r['mean_mse']:<12.6f} {r['mean_rmse']:<12.6f} {r['std_mse']:<10.6f} {r['count']:<8}")
        print("---\n")

        # Save to results dir: JSON (metrics) + PNG (plot)
        out_dir = save_dir or self.cfg.get("results_dir") or self.cfg.get("ckpt_dir", "checkpoints")
        out_dir = os.path.join(out_dir, "temporal_eval")
        os.makedirs(out_dir, exist_ok=True)
        base_name = f"temporal_metrics_{split_name}"
        json_path = os.path.join(out_dir, f"{base_name}.json")
        plot_path = os.path.join(out_dir, f"{base_name}.png")
        # JSON: list of records for easy reading
        records = [{"start_frame": sf, **r} for sf, r in sorted(result.items())]
        with open(json_path, "w") as f:
            json.dump({"split": split_name, "metrics": records}, f, indent=2)
        print(f"Saved temporal metrics to {json_path}")

        if plt is not None:
            fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
            x = sorted(result.keys())
            mean_mse = [result[sf]["mean_mse"] for sf in x]
            std_mse = [result[sf]["std_mse"] for sf in x]
            mean_rmse = [result[sf]["mean_rmse"] for sf in x]
            axes[0].errorbar(x, mean_mse, yerr=std_mse, capsize=3, marker="o", linestyle="-")
            axes[0].set_ylabel("MSE")
            axes[0].set_title(f"Reconstruction MSE over time ({split_name})")
            axes[0].grid(True, alpha=0.3)
            axes[1].plot(x, mean_rmse, marker="o", linestyle="-")
            axes[1].set_xlabel("Clip start frame")
            axes[1].set_ylabel("RMSE")
            axes[1].set_title(f"Reconstruction RMSE over time ({split_name})")
            axes[1].grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved temporal metrics plot to {plot_path}")
        else:
            print("Matplotlib not available; skipping plot.")

        return result

    def _model_forward_for_metrics(self, batch):
        """Run model on batch and return full output (dict or tensor). Used by evaluate_metrics."""
        # Same batch routing as _forward_and_loss, but return full output
        if isinstance(batch, (list, tuple)):
            crops = [b.to(self.device) for b in batch]
            return self.model(crops)
        if isinstance(batch, dict):
            if "crops" in batch:
                crops = [b.to(self.device) for b in batch["crops"]]
                return self.model(crops)
            if "video_masked" in batch and "video_target" in batch and "mask" in batch:
                mae_batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                return self.model(mae_batch)
            if "video" in batch:
                return self.model(batch["video"].to(self.device))
        if torch.is_tensor(batch):
            return self.model(batch.to(self.device))
        return None

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
