#!/usr/bin/env python3
"""
Evaluate a trained MAE model on a user-chosen subset of the **validation** split
(same idea as the Colab snippet for LSTM, works for **mae_2d** and **mae_2d_lstm**).

Outputs under:
  <checkpoint_dir>/SPECIFIC_VAL_<frame_start>_<frame_end>/plots/

Reuses:
  build_dataloaders, load_and_prepare_config, build_ssl_model, Trainer.evaluate_metrics,
  save_test_reconstruction_figure

Examples:
  PYTHONPATH=. python scripts/eval_val_specific.py \\
    --config configs/MAE_2D_full.yaml \\
    --checkpoint-dir ./checkpoints_mae2d \\
    --target-files session_270618b_condsAN.h5 \\
    --frame-start 28 --frame-end 58 \\
    --monkeys gandalf --mask-ratio 0.75

  PYTHONPATH=. python scripts/eval_val_specific.py \\
    --config configs/MAE_2D_LSTM_full.yaml \\
    --checkpoint-dir ./checkpoints \\
    --target-files session_270618b_condsAN.h5 \\
    --clip-length 5 --patch-size 4 8 8 \\
    --frame-start 28 --frame-end 58
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Subset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.experiments.mae_2d_lstm.build_dataloaders import build_dataloaders
from src.experiments.eval_core import (
    build_cfg_overrides_from_args,
    select_indices_by_target_files,
    enforce_mae_2d_clip_contract,
    resolve_and_load_checkpoint,
)
from src.experiments.mae_2d_lstm.load_config import load_and_prepare_config
from src.experiments.eval_plots import save_reconstruction_figure
from src.models import build_ssl_model
from src.training.trainer import Trainer
from src.utils.logger import TBLogger, set_seed


def _build_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    o = build_cfg_overrides_from_args(
        args,
        keys=("monkeys", "mask_ratio", "clip_length", "patch_size", "frame_start", "frame_end", "batch_size"),
    )
    if "monkeys" in o:
        o["monkeys"] = list(o["monkeys"])
    if "patch_size" in o:
        o["patch_size"] = list(o["patch_size"])
    return o


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="MAE eval on specific validation files + recon plots.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint-dir", type=str, required=True)
    p.add_argument("--checkpoint-path", type=str, default=None)
    p.add_argument("--target-files", type=str, nargs="+", required=True)
    p.add_argument("--monkeys", type=str, nargs="*", default=None)
    p.add_argument("--mask-ratio", type=float, default=None)
    p.add_argument("--clip-length", type=int, default=None)
    p.add_argument("--patch-size", type=int, nargs=3, metavar=("T", "H", "W"), default=None)
    p.add_argument("--frame-start", type=int, default=None)
    p.add_argument("--frame-end", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--train-num-workers", type=int, default=0)
    p.add_argument("--val-num-workers", type=int, default=0)
    p.add_argument("--test-num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-vis-batches", type=int, default=20, help="Max samples to plot (batch_size=1).")
    args = p.parse_args(argv)

    set_seed(args.seed)
    project_root = _PROJECT_ROOT
    overrides = _build_overrides(args)
    cfg = load_and_prepare_config(
        base_cfg_path=args.config,
        project_root=project_root,
        data_root=None,
        overrides=overrides,
    )

    # mae_2d: single-frame clips and temporal patch = 1
    cfg = enforce_mae_2d_clip_contract(cfg)

    ckpt_dir = Path(args.checkpoint_dir).resolve()
    cfg["ckpt_dir"] = str(ckpt_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval_val_specific] device={device} model={cfg.get('model')}")

    _, val_loader, _ = build_dataloaders(
        cfg,
        project_root=project_root,
        batch_size=args.batch_size or cfg.get("batch_size", 32),
        train_num_workers=args.train_num_workers,
        val_num_workers=args.val_num_workers,
        test_num_workers=args.test_num_workers,
    )

    selected, selection_meta = select_indices_by_target_files(val_loader.dataset, list(args.target_files))
    print(f"[eval_val_specific] matched val samples: {len(selected)}")
    if not selected:
        raise RuntimeError("No matching files in validation set. Check basenames and --monkeys.")

    val_subset = Subset(val_loader.dataset, selected)
    vis_loader = DataLoader(
        val_subset,
        batch_size=1,
        shuffle=False,
        num_workers=args.val_num_workers,
        pin_memory=cfg.get("pin_memory", False),
    )

    model = build_ssl_model(cfg).to(device)
    ckpt_file, _load_mode = resolve_and_load_checkpoint(model, ckpt_dir, args.checkpoint_path, device)
    model.eval()
    print(f"[eval_val_specific] loaded checkpoint: {ckpt_file}")

    logger = TBLogger(log_dir=str(ckpt_dir / "VAL_SPECIFIC_tb"))
    trainer = Trainer(model=model, logger=logger, cfg=cfg, device=device)
    metrics = trainer.evaluate_metrics(vis_loader, split_name="specific_val")
    print("[eval_val_specific] specific_val metrics:", metrics)

    sf = cfg.get("frame_start", "NA")
    ef = cfg.get("frame_end", "NA")
    out_dir = ckpt_dir / f"SPECIFIC_VAL_{sf}_{ef}" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    max_frames = 4 if cfg.get("model") == "mae_2d_lstm" else 1
    n_vis = min(len(selected), max(1, int(args.max_vis_batches)))

    save_reconstruction_figure(
        model=model,
        test_loader=vis_loader,
        device=device,
        out_dir=str(out_dir),
        split_name="specific_val",
        num_batches=n_vis,
        max_frames_per_clip=max_frames,
        plot_masked=False,
    )
    save_reconstruction_figure(
        model=model,
        test_loader=vis_loader,
        device=device,
        out_dir=str(out_dir),
        split_name="specific_val",
        num_batches=n_vis,
        max_frames_per_clip=max_frames,
        plot_masked=True,
    )

    snap = {
        "checkpoint": str(ckpt_file),
        "config": args.config,
        "target_files": list(args.target_files),
        "n_matched": len(selected),
        "selection": selection_meta,
        "metrics": metrics,
        "out_dir": str(out_dir),
    }
    with open(out_dir.parent / "val_specific_summary.json", "w") as f:
        json.dump(snap, f, indent=2, default=str)

    print(f"[eval_val_specific] saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
