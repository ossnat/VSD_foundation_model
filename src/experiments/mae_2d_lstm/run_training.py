import json
import os
from typing import Dict, Any, Tuple

import torch

from src.training.trainer import Trainer
from src.utils.logger import TBLogger, set_seed
from src.experiments.mae_2d_lstm.vis_test_reconstruction import save_test_reconstruction_figure


def run_training_and_temporal_eval(
    cfg: Dict[str, Any],
    model,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
    """
    Train the given MAE 2D+LSTM model with the Trainer, then evaluate
    reconstruction quality over time on the test set.

    Returns:
        (final_eval_metrics, temporal_metrics)
    """
    set_seed(cfg.get("seed", 42))

    ckpt_dir = cfg.get("ckpt_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    analysis_dir = os.path.join(ckpt_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    # Snapshot of the resolved config used for this run
    config_snapshot_path = os.path.join(analysis_dir, "config_used.json")
    try:
        with open(config_snapshot_path, "w") as f:
            json.dump(cfg, f, indent=2, default=str)
        print(f"Saved config snapshot to {config_snapshot_path}")
    except Exception as e:
        print(f"Warning: failed to save config snapshot ({e}).")

    logger = TBLogger(log_dir=cfg.get("log_dir", "logs"))
    trainer = Trainer(model=model, logger=logger, cfg=cfg, device=device)

    # Train
    history = trainer.fit(train_loader, val_loader)

    # Save train/val loss-vs-epoch plot (if history is available)
    train_loss_epoch = (history or {}).get("train_loss_epoch", []) if isinstance(history, dict) else []
    val_loss_epoch = (history or {}).get("val_loss_epoch", []) if isinstance(history, dict) else []
    if train_loss_epoch:
        loss_history_path = os.path.join(analysis_dir, "loss_history_epoch.json")
        try:
            with open(loss_history_path, "w") as f:
                json.dump(
                    {"train_loss_epoch": train_loss_epoch, "val_loss_epoch": val_loss_epoch},
                    f,
                    indent=2,
                )
            print(f"Saved loss history to {loss_history_path}")
        except Exception as e:
            print(f"Warning: failed to save loss history ({e}).")

        try:
            import matplotlib.pyplot as plt

            epochs = list(range(1, len(train_loss_epoch) + 1))
            plt.figure(figsize=(8, 5))
            plt.plot(epochs, train_loss_epoch, marker="o", label="train/loss_epoch")
            if val_loss_epoch and any(v is not None for v in val_loss_epoch):
                plt.plot(
                    epochs,
                    [v if v is not None else float("nan") for v in val_loss_epoch],
                    marker="o",
                    label="val/loss_epoch",
                )
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.title("Training loss (epoch-level)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()

            loss_plot_path = os.path.join(analysis_dir, "train_val_loss_vs_epoch.png")
            plt.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved loss-vs-epoch plot to {loss_plot_path}")
        except Exception as e:
            print(f"Warning: failed to save loss plot ({e}).")

    # Save final encoder checkpoint
    enc_path = os.path.join(ckpt_dir, "encoder_final.pt")
    torch.save(model.encoder.state_dict(), enc_path)
    print(f"Saved encoder to {enc_path}")

    # Optional: save full model state dict for convenience
    model_final_path = os.path.join(ckpt_dir, "model_final.pt")
    try:
        torch.save(model.state_dict(), model_final_path)
        print(f"Saved full model to {model_final_path}")
    except Exception as e:
        print(f"Warning: failed to save full model ({e}).")

    # Standard metrics on validation and test sets
    metrics_dir = os.path.join(analysis_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    val_metrics = trainer.evaluate_metrics(val_loader, split_name="val")
    test_metrics = trainer.evaluate_metrics(test_loader, split_name="test")

    try:
        with open(os.path.join(metrics_dir, "metrics_val.json"), "w") as f:
            json.dump(val_metrics, f, indent=2, default=str)
        with open(os.path.join(metrics_dir, "metrics_test.json"), "w") as f:
            json.dump(test_metrics, f, indent=2, default=str)
        print(f"Saved metrics to {metrics_dir}")
    except Exception as e:
        print(f"Warning: failed to save metrics JSON ({e}).")

    # Standard metrics on test set (optional, but useful)
    eval_metrics = test_metrics

    # Temporal evaluation: MSE/RMSE/R²/SSIM over time, plus plot + JSON paths are printed by Trainer
    _ = trainer.evaluate_metrics_over_time(val_loader, split_name="val")
    temporal_metrics = trainer.evaluate_metrics_over_time(test_loader, split_name="test")

    # Reconstruction visualization:
    # - original | reconstructed | |diff|
    # - original | masked_input | reconstructed | |diff|
    vis_dir = os.path.join(
        cfg.get("results_dir") or cfg.get("ckpt_dir", "checkpoints"),
        "temporal_eval",
    )

    for split, loader in (("val", val_loader), ("test", test_loader)):
        # original | reconstructed | |diff|
        save_test_reconstruction_figure(
            model,
            loader,
            device,
            out_dir=vis_dir,
            split_name=split,
            num_batches=1,
            max_frames_per_clip=8,
            plot_masked=False,
        )
        # original | masked_input | reconstructed | |diff|
        save_test_reconstruction_figure(
            model,
            loader,
            device,
            out_dir=vis_dir,
            split_name=split,
            num_batches=1,
            max_frames_per_clip=8,
            plot_masked=True,
        )

    return eval_metrics, temporal_metrics

