#!/usr/bin/env python3
"""
Compare reconstruction quality for real vs scrambled frames.

Outputs (under --out-dir):
  - Scatter: metric X vs Y (default mse vs correlation) for real vs scrambled recons
  - Scatter: R² vs SSIM by default (--extra-scatter)
  - true_vs_scrambled_frames.png: original vs scrambled targets (--n-vis-frames)
  - test_reconstruction_orig_masked_recon_diff_scrambled.png: masked recon panels
  - points.csv with mse, correlation, r2, ssim per point

Data source:
  1) split-based eval data from config (default test split), or
  2) a specific .h5 file (optionally one trial dataset).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data import load_dataset
from src.experiments.eval_core import (
    build_cfg_overrides_from_args,
    choose_eval_indices,
    enforce_mae_2d_clip_contract,
    resolve_and_load_checkpoint,
)
from src.experiments.eval_plots import (
    save_real_vs_scrambled_scatter,
    save_reconstruction_from_batches,
    save_true_vs_scrambled_frames,
)
from src.experiments.mae_2d_lstm.build_dataloaders import _filter_split_csv_missing_files
from src.experiments.mae_2d_lstm.load_config import load_and_prepare_config
from src.experiments.recon_eval import (
    METRIC_CHOICES,
    build_single_h5_split_csv as _build_single_h5_split_csv,
    choose_h5_trials as _choose_h5_trials,
    collect_trial_datasets as _collect_trial_datasets,
    is_square_pixels_shape as _is_square_pixels_shape,
    label_for_metric as _label_for_metric,
    merge_checkpoint_config_snapshot as _merge_checkpoint_config_snapshot,
    metric_value as _metric_value,
    reconstruct_batch,
    resolve_h5_path as _resolve_h5_path,
    scramble_spatial as _scramble_spatial,
    start_frame_from_batch as _start_frame_from_batch,
    tensor_to_2d_frame as _target_frame_for_vis,
)
from src.models import build_ssl_model
from src.utils.logger import set_seed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scatter real vs scrambled recon metrics.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint-dir", type=str, required=True)
    p.add_argument("--checkpoint-path", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None)

    # Data source options
    p.add_argument("--h5-path", type=str, default=None, help="Optional local .h5 input.")
    p.add_argument("--h5-root", type=str, default=str(_PROJECT_ROOT.parent / "Data" / "FoundationData" / "ProcessedData"))
    p.add_argument("--trial-dataset", type=str, default=None, help="Optional dataset path inside H5.")
    p.add_argument("--split", type=str, choices=("train", "val", "test"), default="test")
    p.add_argument("--target-files", type=str, nargs="*", default=None, help="Optional file filter in split mode.")

    # Sampling + metrics
    p.add_argument("--n-frames", type=int, default=20)
    p.add_argument(
        "--n-vis-frames",
        type=int,
        default=3,
        help="Number of example frames for true/scrambled and recon panels.",
    )
    p.add_argument("--x-metric", type=str, default="mse", choices=METRIC_CHOICES)
    p.add_argument("--y-metric", type=str, default="correlation", choices=METRIC_CHOICES)
    p.add_argument(
        "--extra-scatter",
        type=str,
        default="r2:ssim",
        help="Optional second scatter as x:y (e.g. r2:ssim). Empty to skip.",
    )
    p.add_argument("--seed", type=int, default=17)

    # Common cfg overrides
    p.add_argument("--monkeys", type=str, nargs="*", default=None)
    p.add_argument("--mask-ratio", type=float, default=None)
    p.add_argument("--clip-length", type=int, default=None)
    p.add_argument("--patch-size", type=int, nargs=3, metavar=("T", "H", "W"), default=None)
    p.add_argument("--frame-start", type=int, default=None)
    p.add_argument("--frame-end", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def _reconstruct(model: Any, batch: Dict[str, Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    recon, target, _, _ = reconstruct_batch(model, batch, device)
    return recon, target


def _parse_extra_scatter(spec: str) -> Optional[Tuple[str, str]]:
    text = (spec or "").strip()
    if not text:
        return None
    if ":" not in text:
        raise ValueError(f"--extra-scatter must be x:y, got {spec!r}")
    x_metric, y_metric = text.split(":", 1)
    x_metric, y_metric = x_metric.strip().lower(), y_metric.strip().lower()
    if x_metric not in METRIC_CHOICES or y_metric not in METRIC_CHOICES:
        raise ValueError(f"Unknown metric in --extra-scatter {spec!r}")
    return x_metric, y_metric


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)

    ckpt_dir = Path(args.checkpoint_dir).resolve()
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"checkpoint-dir not found: {ckpt_dir}")

    project_root = _PROJECT_ROOT
    overrides = build_cfg_overrides_from_args(
        args,
        keys=("monkeys", "mask_ratio", "clip_length", "patch_size", "frame_start", "frame_end", "batch_size"),
    )
    if "monkeys" in overrides:
        overrides["monkeys"] = list(overrides["monkeys"])
    if "patch_size" in overrides:
        overrides["patch_size"] = list(overrides["patch_size"])

    cfg = load_and_prepare_config(args.config, project_root=project_root, overrides=overrides)
    cfg = _merge_checkpoint_config_snapshot(
        cfg, ckpt_dir, project_root, log_prefix="compare_real_vs_scrambled"
    )
    # CLI overrides (e.g. frame_start/end) win over checkpoint snapshot.
    cfg.update(overrides)
    cfg = enforce_mae_2d_clip_contract(cfg)
    cfg["ckpt_dir"] = str(ckpt_dir)
    cfg["auto_resume"] = False
    cfg["resume"] = False

    out_dir = Path(args.out_dir) if args.out_dir else (
        ckpt_dir / f"REAL_VS_SCRAMBLED_{args.x_metric}_VS_{args.y_metric}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.h5_path:
        h5_path = _resolve_h5_path(args.h5_path, Path(args.h5_root))
        if not h5_path.exists():
            raise FileNotFoundError(f"h5 not found: {h5_path}")
        explicit_trials = [args.trial_dataset] if args.trial_dataset else None
        trial_entries = _choose_h5_trials(h5_path, explicit_trials)
        split_csv = _build_single_h5_split_csv(out_dir / "splits", h5_path, trial_entries)
        cfg_h5 = dict(cfg)
        cfg_h5["split_csv_path"] = str(split_csv)
        cfg_h5["monkeys"] = None
        dataset_loader = load_dataset(
            cfg_h5,
            split="val",
            batch_size=1,
            num_workers=max(0, int(args.num_workers)),
            shuffle=False,
        )
        selected_idx, sel_meta = choose_eval_indices(
            dataset_loader.dataset,
            target_files=None,
            subset_size=max(1, int(args.n_frames)),
            seed=args.seed,
        )
        loader = DataLoader(
            Subset(dataset_loader.dataset, selected_idx),
            batch_size=1,
            shuffle=False,
            num_workers=max(0, int(args.num_workers)),
            pin_memory=cfg.get("pin_memory", False),
        )
        source_meta = {"mode": "h5", "h5_path": str(h5_path), "trial_dataset": args.trial_dataset, **sel_meta}
    else:
        cfg_split = dict(cfg)
        split_csv_path = cfg_split.get("split_csv_path")
        if split_csv_path:
            cfg_split["split_csv_path"] = _filter_split_csv_missing_files(split_csv_path, project_root)
        base_loader = load_dataset(
            cfg_split,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=max(0, int(args.num_workers)),
            shuffle=False,
        )
        selected_idx, sel_meta = choose_eval_indices(
            base_loader.dataset,
            target_files=list(args.target_files) if args.target_files else None,
            subset_size=max(1, int(args.n_frames)),
            seed=args.seed,
        )
        if not selected_idx:
            raise RuntimeError("No samples selected for evaluation.")
        loader = DataLoader(
            Subset(base_loader.dataset, selected_idx),
            batch_size=1,
            shuffle=False,
            num_workers=max(0, int(args.num_workers)),
            pin_memory=cfg.get("pin_memory", False),
        )
        source_meta = {"mode": "split", "split": args.split, **sel_meta}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_ssl_model(cfg).to(device)
    ckpt_file, load_mode = resolve_and_load_checkpoint(model, ckpt_dir, args.checkpoint_path, device)
    model.eval()

    gen = torch.Generator(device=device if device.type == "cuda" else "cpu")
    gen.manual_seed(args.seed + 12345)

    extra_scatter = _parse_extra_scatter(args.extra_scatter)
    extra_real: List[Tuple[float, float]] = []
    extra_scr: List[Tuple[float, float]] = []

    real_points: List[Tuple[float, float]] = []
    scr_points: List[Tuple[float, float]] = []
    rows: List[Dict[str, Any]] = []
    vis_true_scr: List[Tuple[torch.Tensor, torch.Tensor, int]] = []
    vis_batches_real: List[Dict[str, Any]] = []
    vis_batches_scr: List[Dict[str, Any]] = []
    n_vis = max(0, int(args.n_vis_frames))

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= int(args.n_frames):
                break
            if not isinstance(batch, dict) or "video_masked" not in batch or "video_target" not in batch:
                continue

            frame_no = _start_frame_from_batch(batch)

            # Real-frame reconstruction
            recon_real, target_real = _reconstruct(model, batch, device)
            xr = _metric_value(args.x_metric, recon_real, target_real)
            yr = _metric_value(args.y_metric, recon_real, target_real)
            r2_r = _metric_value("r2", recon_real, target_real)
            ssim_r = _metric_value("ssim", recon_real, target_real)
            real_points.append((xr, yr))
            rows.append(
                {
                    "idx": i,
                    "kind": "real",
                    args.x_metric: xr,
                    args.y_metric: yr,
                    "r2": r2_r,
                    "ssim": ssim_r,
                }
            )
            if extra_scatter:
                ex, ey = extra_scatter
                extra_real.append(
                    (_metric_value(ex, recon_real, target_real), _metric_value(ey, recon_real, target_real))
                )

            # Scrambled-frame reconstruction (scramble target/masked/mask consistently)
            target_orig = batch["video_target"]
            batch_scr = dict(batch)
            batch_scr["video_target"] = _scramble_spatial(target_orig.to(device), gen).cpu()
            batch_scr["video_masked"] = _scramble_spatial(batch["video_masked"].to(device), gen).cpu()
            batch_scr["mask"] = _scramble_spatial(batch["mask"].to(device), gen).cpu()
            recon_scr, target_scr = _reconstruct(model, batch_scr, device)
            xs = _metric_value(args.x_metric, recon_scr, target_scr)
            ys = _metric_value(args.y_metric, recon_scr, target_scr)
            r2_s = _metric_value("r2", recon_scr, target_scr)
            ssim_s = _metric_value("ssim", recon_scr, target_scr)
            scr_points.append((xs, ys))
            rows.append(
                {
                    "idx": i,
                    "kind": "scrambled",
                    args.x_metric: xs,
                    args.y_metric: ys,
                    "r2": r2_s,
                    "ssim": ssim_s,
                }
            )
            if extra_scatter:
                ex, ey = extra_scatter
                extra_scr.append(
                    (_metric_value(ex, recon_scr, target_scr), _metric_value(ey, recon_scr, target_scr))
                )

            if i < n_vis:
                vis_true_scr.append(
                    (_target_frame_for_vis(target_orig), _target_frame_for_vis(batch_scr["video_target"]), frame_no)
                )
                vis_batches_real.append(batch)
                vis_batches_scr.append(batch_scr)

    if not real_points:
        raise RuntimeError("No valid samples were processed.")

    plot_paths: Dict[str, str] = {}

    plot_path = out_dir / f"real_vs_scrambled_{args.x_metric}_vs_{args.y_metric}.png"
    save_real_vs_scrambled_scatter(
        real_points=real_points,
        scrambled_points=scr_points,
        out_path=str(plot_path),
        x_label=_label_for_metric(args.x_metric),
        y_label=_label_for_metric(args.y_metric),
        title=f"Real vs Scrambled reconstruction ({args.x_metric} vs {args.y_metric})",
    )
    plot_paths["primary_scatter"] = str(plot_path)

    if extra_scatter and extra_real:
        ex, ey = extra_scatter
        extra_path = out_dir / f"real_vs_scrambled_{ex}_vs_{ey}.png"
        save_real_vs_scrambled_scatter(
            real_points=extra_real,
            scrambled_points=extra_scr,
            out_path=str(extra_path),
            x_label=_label_for_metric(ex),
            y_label=_label_for_metric(ey),
            title=f"Real vs Scrambled reconstruction ({ex} vs {ey})",
        )
        plot_paths["extra_scatter"] = str(extra_path)

    if vis_true_scr:
        true_scr_path = out_dir / "true_vs_scrambled_frames.png"
        save_true_vs_scrambled_frames(vis_true_scr, str(true_scr_path))
        plot_paths["true_vs_scrambled"] = str(true_scr_path)

    if vis_batches_real:
        recon_real_path = out_dir / "test_reconstruction_orig_masked_recon_diff_real.png"
        save_reconstruction_from_batches(
            model,
            vis_batches_real,
            device,
            str(recon_real_path),
            plot_masked=True,
        )
        plot_paths["real_reconstruction"] = str(recon_real_path)

    if vis_batches_scr:
        recon_scr_path = out_dir / "test_reconstruction_orig_masked_recon_diff_scrambled.png"
        save_reconstruction_from_batches(
            model,
            vis_batches_scr,
            device,
            str(recon_scr_path),
            plot_masked=True,
        )
        plot_paths["scrambled_reconstruction"] = str(recon_scr_path)

    csv_path = out_dir / "points.csv"
    fieldnames = ["idx", "kind", args.x_metric, args.y_metric, "r2", "ssim"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary = {
        "config": args.config,
        "checkpoint_dir": str(ckpt_dir),
        "checkpoint_file": str(ckpt_file),
        "checkpoint_load_mode": load_mode,
        "source": source_meta,
        "frame_start": cfg.get("frame_start"),
        "frame_end": cfg.get("frame_end"),
        "n_points_real": len(real_points),
        "n_points_scrambled": len(scr_points),
        "n_vis_frames": len(vis_true_scr),
        "x_metric": args.x_metric,
        "y_metric": args.y_metric,
        "extra_scatter": f"{extra_scatter[0]}:{extra_scatter[1]}" if extra_scatter else None,
        "plots": plot_paths,
        "points_csv": str(csv_path),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"[compare_real_vs_scrambled] Saved plot: {plot_path}")
    print(f"[compare_real_vs_scrambled] Saved points: {csv_path}")
    for name, path in plot_paths.items():
        if name != "primary_scatter":
            print(f"[compare_real_vs_scrambled] Saved {name}: {path}")


if __name__ == "__main__":
    main()
