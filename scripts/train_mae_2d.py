#!/usr/bin/env python3
"""
Train **MAE 2D** (ResNet18 + MAEDecoder2D only — no LSTM / no temporal encoder).

Mirrors `train_mae_2d_lstm.py`: same dataloaders, Trainer, test metrics, temporal plots,
and reconstruction PNG. Default config: `configs/MAE_2D_full.yaml` (clip_length: 1).

Load order: YAML base config → apply CLI overrides (only set keys you pass).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from src.experiments.mae_2d_lstm.load_config import load_and_prepare_config
from src.experiments.mae_2d_lstm.build_dataloaders import build_dataloaders
from src.experiments.mae_2d_lstm.run_training import run_training_and_temporal_eval
from src.models import build_ssl_model


def _optional_str(s: Optional[str]) -> Optional[str]:
    if s is None or str(s).strip().lower() in ("", "none", "null"):
        return None
    return s


def _merge_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Build flat config overrides from CLI args (only non-None values)."""
    o: Dict[str, Any] = {}

    def set_if(key: str, value: Any) -> None:
        if value is not None:
            o[key] = value

    set_if("model", "mae_2d")
    set_if("monkeys", args.monkeys)
    set_if("mask_ratio", args.mask_ratio)
    set_if("clip_length", args.clip_length)
    set_if("epochs", args.epochs)
    set_if("batch_size", args.batch_size)
    set_if("patch_size", list(args.patch_size) if args.patch_size is not None else None)
    set_if("frame_start", args.frame_start)
    set_if("frame_end", args.frame_end)
    set_if("val_frame_stride", args.val_frame_stride)
    set_if("lr", args.lr)
    set_if("weight_decay", args.weight_decay)
    set_if("seed", args.seed)
    set_if("max_grad_norm", args.max_grad_norm)
    set_if("ckpt_dir", args.ckpt_dir)
    set_if("log_dir", args.log_dir)
    set_if("results_dir", args.results_dir)
    set_if("dataset_name", args.dataset_name)
    set_if("split_csv_path", args.split_csv_path)
    set_if("stats_json_path", args.stats_json_path)
    set_if("processed_root", args.processed_root)
    set_if("backbone", args.backbone)
    set_if("channels", args.channels)
    set_if("hidden_dim", args.hidden_dim)
    set_if("crop_loss_radius", getattr(args, "crop_loss_radius", None))

    if getattr(args, "crop_loss", None) is not None:
        o["crop_loss"] = _optional_str(args.crop_loss)

    if args.pretrained is not None:
        o["pretrained"] = args.pretrained
    if args.normalize_loss is not None:
        o["normalize_loss"] = args.normalize_loss
    if args.preload_into_ram is not None:
        o["preload_into_ram"] = args.preload_into_ram
    if args.pin_memory is not None:
        o["pin_memory"] = args.pin_memory

    crop_frame_arg = getattr(args, "crop_frame", None)
    cf = _optional_str(crop_frame_arg)
    if cf is not None or crop_frame_arg is not None:
        o["crop_frame"] = cf

    cr = getattr(args, "crop_radius", None)
    if cr is not None:
        o["crop_radius"] = cr

    return o


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train MAE 2D (ResNet18 + MAEDecoder2D). Overrides config when flags are set.",
    )
    p.add_argument(
        "--config",
        type=str,
        default="configs/MAE_2D_full.yaml",
        help="Base YAML path (relative to project root or absolute).",
    )

    # Data
    p.add_argument("--dataset-name", type=str, default=None, help="e.g. vsd_mae")
    p.add_argument("--split-csv-path", type=str, default=None)
    p.add_argument("--stats-json-path", type=str, default=None)
    p.add_argument("--processed-root", type=str, default=None)
    p.add_argument("--monkeys", type=str, nargs="*", default=None)
    p.add_argument("--mask-ratio", type=float, default=None)
    p.add_argument("--clip-length", type=int, default=None, help="Use 1 for single-frame 2D MAE.")
    p.add_argument("--frame-start", type=int, default=None)
    p.add_argument("--frame-end", type=int, default=None, help="Inclusive end index (dataset slice).")
    p.add_argument("--val-frame-stride", type=int, default=None)
    p.add_argument("--patch-size", type=int, nargs=3, metavar=("T", "H", "W"), default=None)
    p.add_argument("--crop-frame", type=str, default=None, help='null | "square" | "circle" — use "null" to clear.')
    p.add_argument("--crop-radius", type=int, default=None, help="Used when crop_frame is set.")
    p.add_argument(
        "--preload-into-ram",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Preload trials into RAM (default: from YAML).",
    )
    p.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="DataLoader pin_memory (default: from YAML).",
    )

    # Model
    p.add_argument("--backbone", type=str, default=None, help="resnet18 | MAEShallowCNNBackbone")
    p.add_argument("--channels", type=int, default=None)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pretrained backbone weights.",
    )
    p.add_argument(
        "--normalize-loss",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="MAE loss normalize flag (default: from YAML).",
    )
    p.add_argument(
        "--crop-loss",
        type=str,
        default=None,
        help='Loss crop: null | "square" | "circle" (matches YAML crop_loss).',
    )
    p.add_argument(
        "--crop-loss-radius",
        type=int,
        default=None,
        help="Pixel radius for crop_loss (matches YAML crop_loss_radius; default from config if omitted).",
    )

    # Training
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-grad-norm", type=float, default=None)
    p.add_argument("--ckpt-dir", type=str, default=None)
    p.add_argument("--log-dir", type=str, default=None)
    p.add_argument("--results-dir", type=str, default=None, help="Optional; else temporal_eval under ckpt_dir.")

    p.add_argument("--train-num-workers", type=int, default=4)
    p.add_argument("--val-num-workers", type=int, default=0)
    p.add_argument("--test-num-workers", type=int, default=0)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent
    overrides = _merge_cli_overrides(args)

    cfg = load_and_prepare_config(
        base_cfg_path=args.config,
        project_root=project_root,
        data_root=None,
        overrides=overrides,
    )

    if cfg.get("model") != "mae_2d":
        print(
            f"[train_mae_2d] Warning: config has model={cfg.get('model')!r}; "
            f"this script trains MAE 2D. Forcing model='mae_2d'."
        )
        cfg["model"] = "mae_2d"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_mae_2d] Using device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders(
        cfg,
        project_root=project_root,
        batch_size=args.batch_size,
        train_num_workers=args.train_num_workers,
        val_num_workers=args.val_num_workers,
        test_num_workers=args.test_num_workers,
    )

    model = build_ssl_model(cfg).to(device)

    eval_metrics, temporal_metrics = run_training_and_temporal_eval(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
    )

    print("[train_mae_2d] Final test metrics:", eval_metrics)
    print("[train_mae_2d] Temporal metrics keys (start_frames):", sorted(temporal_metrics.keys()))


if __name__ == "__main__":
    main()
