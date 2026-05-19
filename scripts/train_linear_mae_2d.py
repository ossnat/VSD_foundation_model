#!/usr/bin/env python3
"""
Train **Linear MAE 2D** baseline: single Linear(map) from [masked frame | mask] to reconstruction.

Same dataloaders, Trainer, metrics, temporal JSON/plots, and reconstruction PNGs as MAE 2D.
Default config: `configs/linear_mae_2d_full.yaml`.

For **lasso / elastic-net** behaviour, set `linear_l1_penalty` in YAML (L1 on weights) and tune
`weight_decay` (L2 / ridge via AdamW).

Load order: YAML base config → CLI overrides.
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
    o: Dict[str, Any] = {}

    def set_if(key: str, value: Any) -> None:
        if value is not None:
            o[key] = value

    set_if("model", "linear_mae_2d")
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
    set_if("linear_l1_penalty", args.linear_l1_penalty)
    set_if("seed", args.seed)
    set_if("max_grad_norm", args.max_grad_norm)
    set_if("ckpt_dir", args.ckpt_dir)
    set_if("log_dir", args.log_dir)
    set_if("results_dir", args.results_dir)
    set_if("dataset_name", args.dataset_name)
    set_if("split_csv_path", args.split_csv_path)
    set_if("stats_json_path", args.stats_json_path)
    set_if("processed_root", args.processed_root)
    set_if("channels", args.channels)
    set_if("crop_loss_radius", getattr(args, "crop_loss_radius", None))
    set_if("loss_type", args.loss_type)

    if getattr(args, "linear_spatial_hw", None) is not None and len(args.linear_spatial_hw) == 2:
        o["linear_spatial_hw"] = [int(args.linear_spatial_hw[0]), int(args.linear_spatial_hw[1])]

    if getattr(args, "crop_loss", None) is not None:
        o["crop_loss"] = _optional_str(args.crop_loss)

    if args.normalize_loss is not None:
        o["normalize_loss"] = args.normalize_loss
    if args.preload_into_ram is not None:
        o["preload_into_ram"] = args.preload_into_ram
    if args.pin_memory is not None:
        o["pin_memory"] = args.pin_memory
    if getattr(args, "use_amp", None) is not None:
        o["use_amp"] = bool(args.use_amp)

    if getattr(args, "temporal_eval_per_metric_plots", None) is not None:
        o["temporal_eval_per_metric_plots"] = bool(args.temporal_eval_per_metric_plots)
    if getattr(args, "resume", None) is not None:
        o["resume"] = bool(args.resume)
    set_if("resume_checkpoint_path", getattr(args, "resume_checkpoint_path", None))

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
        description="Train Linear MAE 2D baseline (ridge / optional L1 on weights).",
    )
    p.add_argument(
        "--config",
        type=str,
        default="configs/linear_mae_2d_full.yaml",
        help="Base YAML path (relative to project root or absolute).",
    )

    p.add_argument("--dataset-name", type=str, default=None)
    p.add_argument("--split-csv-path", type=str, default=None)
    p.add_argument("--stats-json-path", type=str, default=None)
    p.add_argument("--processed-root", type=str, default=None)
    p.add_argument("--monkeys", type=str, nargs="*", default=None)
    p.add_argument("--mask-ratio", type=float, default=None)
    p.add_argument("--clip-length", type=int, default=None, help="Use 1 for 2D linear MAE.")
    p.add_argument("--frame-start", type=int, default=None)
    p.add_argument("--frame-end", type=int, default=None)
    p.add_argument("--val-frame-stride", type=int, default=None)
    p.add_argument("--patch-size", type=int, nargs=3, metavar=("T", "H", "W"), default=None)
    p.add_argument(
        "--linear-spatial-hw",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help="Trimmed spatial size (must match dataloader output). Default: infer from stats + patch.",
    )
    p.add_argument("--crop-frame", type=str, default=None)
    p.add_argument("--crop-radius", type=int, default=None)
    p.add_argument(
        "--preload-into-ram",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p.add_argument(
        "--use-amp",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    p.add_argument("--channels", type=int, default=None)
    p.add_argument(
        "--normalize-loss",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p.add_argument(
        "--crop-loss",
        type=str,
        default=None,
        help='null | "square" | "circle"',
    )
    p.add_argument("--crop-loss-radius", type=int, default=None)
    p.add_argument(
        "--loss-type",
        type=str,
        default=None,
        help='Training loss_type for MAELoss: "mse" | "l1" | "l1_mse" | "l1_ssim" | "mse_ssim"',
    )

    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument(
        "--linear-l1-penalty",
        type=float,
        default=None,
        help="L1 penalty on linear weights (0 = ridge only via weight_decay).",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-grad-norm", type=float, default=None)
    p.add_argument("--ckpt-dir", type=str, default=None)
    p.add_argument("--log-dir", type=str, default=None)
    p.add_argument("--results-dir", type=str, default=None)
    p.add_argument(
        "--temporal-eval-per-metric-plots",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save one temporal PNG per metric (default: from YAML).",
    )
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Resume from latest checkpoint in ckpt_dir (default: false; Colab-safe).",
    )
    p.add_argument(
        "--resume-checkpoint-path",
        type=str,
        default=None,
        help="Explicit checkpoint .pt path when resuming (optional).",
    )

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

    if cfg.get("model") != "linear_mae_2d":
        print(
            f"[train_linear_mae_2d] Warning: config has model={cfg.get('model')!r}; "
            f"forcing model='linear_mae_2d'."
        )
        cfg["model"] = "linear_mae_2d"

    if int(cfg.get("clip_length", 1)) > 1:
        print(
            "[train_linear_mae_2d] Warning: clip_length > 1 is not supported for linear_mae_2d. "
            "Forcing clip_length=1."
        )
        cfg["clip_length"] = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_linear_mae_2d] Using device: {device}")

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

    print("[train_linear_mae_2d] Final test metrics:", eval_metrics)
    print(
        "[train_linear_mae_2d] Temporal test keys (start_frames):",
        sorted(temporal_metrics.keys()),
    )


if __name__ == "__main__":
    main()
