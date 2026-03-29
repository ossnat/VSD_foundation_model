#!/usr/bin/env python3
"""
Train the MAE 2D+LSTM model using the standard config and data pipeline.

This script is designed to roughly mimic the Colab notebook
MAE_3D_VideoMAE_2D_CNN_V2.ipynb, but with cleaner structure:

- Load a base config (e.g. configs/MAE_2D_LSTM_full.yaml)
- Optionally apply flat overrides via CLI
- Resolve data paths against a local Data/ directory
- Build train/val/test loaders (skipping rows whose H5 files are missing)
- Build the MAE 2D+LSTM model
- Train with Trainer
- Evaluate metrics on the test set and MSE/RMSE over time (with plots)
"""

import argparse
from pathlib import Path
from typing import Dict, Any

import torch

from src.experiments.mae_2d_lstm.load_config import load_and_prepare_config
from src.experiments.mae_2d_lstm.build_dataloaders import build_dataloaders
from src.experiments.mae_2d_lstm.build_model import build_mae_2d_lstm_model
from src.experiments.mae_2d_lstm.run_training import run_training_and_temporal_eval


def _parse_overrides_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Collect simple flat config overrides from CLI args."""
    overrides: Dict[str, Any] = {}
    if args.monkeys:
        overrides["monkeys"] = args.monkeys
    if args.mask_ratio is not None:
        overrides["mask_ratio"] = args.mask_ratio
    if args.clip_length is not None:
        overrides["clip_length"] = args.clip_length
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.patch_size is not None:
        overrides["patch_size"] = list(args.patch_size)
    if args.frame_start is not None:
        overrides["frame_start"] = args.frame_start
    if args.frame_end is not None:
        overrides["frame_end"] = args.frame_end
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MAE 2D+LSTM model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/MAE_2D_LSTM_full.yaml",
        help="Path to base YAML config.",
    )
    parser.add_argument(
        "--monkeys",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of monkeys to include, e.g. --monkeys frodo.",
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=None,
        help="Override mask_ratio in config.",
    )
    parser.add_argument(
        "--clip-length",
        type=int,
        default=None,
        help="Override clip_length in config.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs in config.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch_size in config (also used for loaders if provided).",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=3,
        metavar=("T", "H", "W"),
        default=None,
        help="Override patch_size as three integers, e.g. --patch-size 4 8 8 (matches YAML [T,H,W]).",
    )
    parser.add_argument(
        "--frame-start",
        type=int,
        default=None,
        help="Override frame_start in config (inclusive frame index).",
    )
    parser.add_argument(
        "--frame-end",
        type=int,
        default=None,
        help="Override frame_end in config (exclusive end index, same semantics as YAML).",
    )
    parser.add_argument(
        "--train-num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers for train split.",
    )
    parser.add_argument(
        "--val-num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers for val split.",
    )
    parser.add_argument(
        "--test-num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers for test split.",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    overrides = _parse_overrides_from_args(args)

    cfg = load_and_prepare_config(
        base_cfg_path=args.config,
        project_root=project_root,
        data_root=None,  # assume Data/ is sibling of project root
        overrides=overrides,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_mae_2d_lstm] Using device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders(
        cfg,
        project_root=project_root,
        batch_size=args.batch_size,
        train_num_workers=args.train_num_workers,
        val_num_workers=args.val_num_workers,
        test_num_workers=args.test_num_workers,
    )

    model = build_mae_2d_lstm_model(cfg, device=device)

    eval_metrics, temporal_metrics = run_training_and_temporal_eval(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
    )

    print("[train_mae_2d_lstm] Final test metrics:", eval_metrics)
    print("[train_mae_2d_lstm] Temporal metrics keys (start_frames):", sorted(temporal_metrics.keys()))


if __name__ == "__main__":
    main()

