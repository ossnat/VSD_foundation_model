#!/usr/bin/env python3
"""
Train MAE 2D+LSTM with config overrides. Uses MAE_2D_LSTM_full.yaml as base;
edit CONFIG_OVERRIDES and LOADER_OPTIONS below to change data, training, and loader settings.
"""
import os
import sys
import yaml
import torch
from pathlib import Path

# Add project root so imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_dataset
from src.models import build_ssl_model
from src.training.trainer import Trainer
from src.utils.logger import TBLogger, set_seed


# =============================================================================
# Edit these to change run settings (overrides applied on top of base config)
# =============================================================================

CONFIG_OVERRIDES = {
    # Data / split
    "monkeys": ["frodo"],
    # "split_csv_path": "Data/FoundationData/ProcessedData/splits/split_v2_seed17_strat_monkey.csv",
    # "stats_json_path": "Data/FoundationData/ProcessedData/splits/baseline_stats_v2_seed17_strat_monkey.json",
    "split_csv_path": 'Data/FoundationData/ProcessedData/splits/split_v2_seed17_session_split.csv',
    "stats_json_path": 'Data/FoundationData/ProcessedData/splits/baseline_stats_v2_seed17_session_split.json',    # MAE masking (dataset)
    "mask_ratio": 0.85,
    # Training
    "epochs": 16,
    "batch_size": 32,  # only used if LOADER_OPTIONS["batch_size"] is None
    "clip_length": 5,
}

# Loader options (None = use cfg["batch_size"])
LOADER_OPTIONS = {
    "batch_size": 512,
    "train_num_workers": 4,
    "val_num_workers": 0,
    "test_num_workers": 0,
    "train_shuffle": True,
    "val_shuffle": False,
    "test_shuffle": False,
}

# Base config file (relative to project root or absolute)
BASE_CONFIG_PATH = "configs/MAE_2D_LSTM_full.yaml"


def resolve_paths(cfg: dict, base_dir: Path) -> None:
    """Resolve relative data paths against base_dir (project root)."""
    for key in ("split_csv_path", "stats_json_path", "processed_root"):
        value = cfg.get(key)
        if value is None:
            continue
        value_path = Path(value)
        if not value_path.is_absolute():
            cfg[key] = str((base_dir / value_path).resolve())


def main():
    base_dir = PROJECT_ROOT
    config_path = base_dir / BASE_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Base config not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Apply overrides (user-editable)
    for k, v in CONFIG_OVERRIDES.items():
        cfg[k] = v

    resolve_paths(cfg, base_dir)

    # Batch size for loaders: explicit LOADER_OPTIONS["batch_size"] or cfg
    batch_size = LOADER_OPTIONS.get("batch_size") or cfg.get("batch_size", 32)
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loaders
    train_loader = load_dataset(
        cfg,
        split="train",
        batch_size=batch_size,
        num_workers=LOADER_OPTIONS.get("train_num_workers", 4),
        shuffle=LOADER_OPTIONS.get("train_shuffle", True),
    )
    val_loader = load_dataset(
        cfg,
        split="val",
        batch_size=batch_size,
        num_workers=LOADER_OPTIONS.get("val_num_workers", 0),
        shuffle=LOADER_OPTIONS.get("val_shuffle", False),
    )
    test_loader = load_dataset(
        cfg,
        split="test",
        batch_size=batch_size,
        num_workers=LOADER_OPTIONS.get("test_num_workers", 0),
        shuffle=LOADER_OPTIONS.get("test_shuffle", False),
    )

    # Model (2D+LSTM MAE from config)
    model = build_ssl_model(cfg).to(device)
    logger = TBLogger(log_dir=cfg.get("log_dir", "logs"))
    trainer = Trainer(model=model, logger=logger, cfg=cfg, device=device)

    # Train
    trainer.fit(train_loader, val_loader)

    # Save final encoder
    ckpt_dir = cfg.get("ckpt_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    enc_path = os.path.join(ckpt_dir, "encoder_final.pt")
    torch.save(model.encoder.state_dict(), enc_path)
    print(f"Saved encoder to {enc_path}")

    # Evaluate MSE over time on test set; plot and save JSON, then print locations to screen
    temporal_metrics = trainer.evaluate_metrics_over_time(test_loader, split_name="test")
    out_dir = cfg.get("results_dir") or cfg.get("ckpt_dir", "checkpoints")
    temporal_dir = os.path.join(out_dir, "temporal_eval")
    plot_path = os.path.join(temporal_dir, "temporal_metrics_test.png")
    json_path = os.path.join(temporal_dir, "temporal_metrics_test.json")
    print("\n--- Temporal evaluation outputs (test set) ---")
    print(f"  Plot:  {os.path.abspath(plot_path)}")
    print(f"  JSON:  {os.path.abspath(json_path)}")
    print("---\n")


if __name__ == "__main__":
    main()
