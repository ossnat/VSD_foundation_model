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

    logger = TBLogger(log_dir=cfg.get("log_dir", "logs"))
    trainer = Trainer(model=model, logger=logger, cfg=cfg, device=device)

    # Train
    trainer.fit(train_loader, val_loader)

    # Save final encoder checkpoint
    ckpt_dir = cfg.get("ckpt_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    enc_path = os.path.join(ckpt_dir, "encoder_final.pt")
    torch.save(model.encoder.state_dict(), enc_path)
    print(f"Saved encoder to {enc_path}")

    # Standard metrics on test set (optional, but useful)
    eval_metrics = trainer.evaluate_metrics(test_loader, split_name="test")

    # Temporal evaluation: MSE/RMSE/R²/SSIM over time, plus plot + JSON paths are printed by Trainer
    temporal_metrics = trainer.evaluate_metrics_over_time(test_loader, split_name="test")

    # Test-only visualization: original | reconstructed | |diff|
    vis_dir = os.path.join(
        cfg.get("results_dir") or cfg.get("ckpt_dir", "checkpoints"),
        "temporal_eval",
    )
    save_test_reconstruction_figure(
        model,
        test_loader,
        device,
        out_dir=vis_dir,
        split_name="test",
        num_batches=1,
        max_frames_per_clip=8,
    )

    return eval_metrics, temporal_metrics

