#!/usr/bin/env python3
"""
Reconstruct a specific local .h5 file with a trained MAE checkpoint.

This script does not depend on train/val/test membership in the original split CSV.
It creates a temporary one-row split CSV for the provided file and reuses the existing
dataset/masking/model/plot code path:
  - src.data.load_dataset (vsd_mae)
  - src.models.build_ssl_model
  - src.training.trainer.Trainer.evaluate_metrics
  - src.experiments.mae_2d_lstm.vis_test_reconstruction.save_test_reconstruction_figure

Default paths:
  - h5 root: /Users/ossnat/GondaResearch/VSD_FM/Data/FoundationData/ProcessedData
  - checkpoints root: /Users/ossnat/GondaResearch/VSD_FM/TrainedModels
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import pandas as pd
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data import load_dataset
from src.experiments.mae_2d_lstm.checkpoint_utils import resolve_checkpoint_file
from src.experiments.mae_2d_lstm.vis_test_reconstruction import save_test_reconstruction_figure
from src.models import build_ssl_model
from src.training.trainer import Trainer
from src.utils.logger import TBLogger, set_seed


_PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_H5_ROOT = (
    _PROJECT.parent / "Data" / "FoundationData" / "ProcessedData"
)
DEFAULT_CKPT_ROOT = _PROJECT / "checkpoints"


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reconstruct masked clips from one local .h5 file.")
    p.add_argument(
        "--h5-path",
        type=str,
        required=True,
        help="Path to .h5 file (absolute or relative to --h5-root).",
    )
    p.add_argument(
        "--h5-root",
        type=str,
        default=str(DEFAULT_H5_ROOT),
        help="Base directory used when --h5-path is relative.",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Checkpoint directory that contains model files and analysis/config_used.json.",
    )
    p.add_argument(
        "--checkpoints-root",
        type=str,
        default=str(DEFAULT_CKPT_ROOT),
        help="Base directory used when --checkpoint-dir is relative.",
    )
    p.add_argument(
        "--config-json",
        type=str,
        default=None,
        help="Optional explicit config JSON path. Defaults to <checkpoint-dir>/analysis/config_used.json.",
    )
    p.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Optional explicit checkpoint file. Otherwise auto-selects model_final.pt/latest epoch/encoder_final.",
    )
    p.add_argument(
        "--trial-dataset",
        type=str,
        default=None,
        help="Dataset path within .h5. If omitted, runs all compatible trial datasets in the file.",
    )
    p.add_argument("--frame-start", type=int, default=None, help="Optional override for frame_start (inclusive).")
    p.add_argument("--frame-end", type=int, default=None, help="Optional override for frame_end (inclusive).")
    p.add_argument("--clip-length", type=int, default=None, help="Optional override clip_length.")
    p.add_argument("--mask-ratio", type=float, default=None, help="Optional override mask_ratio.")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size for evaluation loader.")
    p.add_argument("--num-workers", type=int, default=0, help="Workers for evaluation loader.")
    p.add_argument("--max-vis-batches", type=int, default=20, help="Max plotted rows (samples).")
    p.add_argument("--seed", type=int, default=17)
    return p.parse_args(argv)


def _resolve_input_path(path_value: str, base_dir: Path) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _load_cfg_from_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config JSON not found: {path}")
    with open(path, "r") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config JSON must contain an object: {path}")
    return cfg


def _collect_h5_datasets(h5_path: Path) -> List[Tuple[str, Tuple[int, ...]]]:
    datasets: List[Tuple[str, Tuple[int, ...]]] = []
    with h5py.File(h5_path, "r") as h5:
        def _visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                datasets.append((name, tuple(obj.shape)))

        h5.visititems(_visitor)
    return sorted(datasets, key=lambda x: x[0])


def _is_square_pixels_shape(shape: Tuple[int, ...]) -> bool:
    if len(shape) != 2:
        return False
    n_pixels = int(shape[0])
    side = int(n_pixels ** 0.5)
    return side * side == n_pixels and shape[1] >= 1


def _select_trial_datasets(h5_path: Path, explicit: Optional[str]) -> List[Tuple[str, Tuple[int, ...]]]:
    datasets = _collect_h5_datasets(h5_path)
    if not datasets:
        raise ValueError(f"No datasets found in H5 file: {h5_path}")

    if explicit:
        for name, shape in datasets:
            if name == explicit:
                if not _is_square_pixels_shape(shape):
                    raise ValueError(
                        f"--trial-dataset={explicit!r} has shape {shape}. "
                        "Expected (n_pixels, n_frames) with n_pixels a perfect square."
                    )
                return [(name, shape)]
        raise ValueError(f"Dataset {explicit!r} not found in {h5_path}.")

    candidates = [(name, shape) for name, shape in datasets if _is_square_pixels_shape(shape)]
    if not candidates:
        preview = ", ".join(f"{n}:{s}" for n, s in datasets[:8])
        raise ValueError(
            "Could not auto-select a trial dataset with shape (n_pixels, n_frames). "
            f"Found: {preview}"
        )
    print(f"[reconstruct_h5_local] Found {len(candidates)} compatible trial datasets in file.")
    return candidates


def _sanitize_tag(value: str) -> str:
    return re.sub(r"[^\w.-]+", "_", value).strip("_") or "sample"


def _build_single_file_split_csv(
    out_dir: Path,
    h5_path: Path,
    trial_dataset: str,
    shape: Tuple[int, ...],
    monkey: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"single_file_split_{_sanitize_tag(trial_dataset)}.csv"
    row = {
        "target_file": str(h5_path),
        "trial_dataset": trial_dataset,
        "shape": str(tuple(int(x) for x in shape)),
        "split": "val",
        # Placeholder metadata fields expected by existing dataset/eval code.
        "monkey": monkey,
        "date": "local",
        "condition": "local",
    }
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    return csv_path


def _evaluate_one_trial_dataset(
    model: torch.nn.Module,
    trainer: Trainer,
    cfg_base: Dict[str, Any],
    args: argparse.Namespace,
    h5_path: Path,
    trial_dataset: str,
    shape: Tuple[int, ...],
    out_root: Path,
) -> Dict[str, Any]:
    trial_tag = _sanitize_tag(trial_dataset)
    split_csv = _build_single_file_split_csv(
        out_root / "splits",
        h5_path,
        trial_dataset,
        shape,
        monkey=h5_path.parent.name,
    )

    cfg_eval = dict(cfg_base)
    cfg_eval["split_csv_path"] = str(split_csv)

    eval_loader = load_dataset(
        cfg_eval,
        split="val",
        batch_size=max(1, int(args.batch_size)),
        num_workers=max(0, int(args.num_workers)),
        shuffle=False,
    )

    metrics = trainer.evaluate_metrics(eval_loader, split_name=f"local_file_{trial_tag}")
    max_frames = 4 if cfg_eval.get("model") == "mae_2d_lstm" else 1
    n_vis = min(len(eval_loader.dataset), max(1, int(args.max_vis_batches)))
    trial_out_dir = out_root / "plots" / trial_tag
    trial_out_dir.mkdir(parents=True, exist_ok=True)

    save_test_reconstruction_figure(
        model=model,
        test_loader=eval_loader,
        device=trainer.device,
        out_dir=str(trial_out_dir),
        split_name=f"local_file_{trial_tag}",
        num_batches=n_vis,
        max_frames_per_clip=max_frames,
        plot_masked=False,
    )
    save_test_reconstruction_figure(
        model=model,
        test_loader=eval_loader,
        device=trainer.device,
        out_dir=str(trial_out_dir),
        split_name=f"local_file_{trial_tag}",
        num_batches=n_vis,
        max_frames_per_clip=max_frames,
        plot_masked=True,
    )

    return {
        "trial_dataset": trial_dataset,
        "shape": list(shape),
        "n_samples": len(eval_loader.dataset),
        "metrics": metrics,
        "trial_out_dir": str(trial_out_dir),
        "single_file_split_csv": str(split_csv),
    }


def _apply_runtime_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(cfg)
    # Local single-file inference should not be filtered by train-time monkey subset.
    cfg["monkeys"] = None
    if args.frame_start is not None:
        cfg["frame_start"] = int(args.frame_start)
    if args.frame_end is not None:
        cfg["frame_end"] = int(args.frame_end)
    if args.clip_length is not None:
        cfg["clip_length"] = int(args.clip_length)
    if args.mask_ratio is not None:
        cfg["mask_ratio"] = float(args.mask_ratio)

    # Keep 2D MAE input contract safe.
    if cfg.get("model") == "mae_2d":
        cfg["clip_length"] = 1
        ps = cfg.get("patch_size")
        if isinstance(ps, list) and len(ps) == 3:
            cfg["patch_size"] = [1, int(ps[1]), int(ps[2])]
    return cfg


def _remap_data_path(path_value: str, data_root: Path) -> str:
    """
    If path_value doesn't exist and contains a '/Data/' segment (e.g. Colab '/content/Data/...'),
    remap it to local data_root while preserving the subpath after 'Data/'.
    """
    p = Path(path_value)
    if p.exists():
        return str(p)

    text = str(p).replace("\\", "/")
    marker = "/Data/"
    idx = text.find(marker)
    if idx >= 0:
        suffix = text[idx + len(marker):]
        candidate = (data_root / suffix).resolve()
        if candidate.exists():
            return str(candidate)
    return str(p)


def _remap_cfg_data_paths(cfg: Dict[str, Any], h5_root: Path) -> Dict[str, Any]:
    cfg = dict(cfg)
    # h5_root is .../Data/FoundationData/ProcessedData; local Data root is parent of FoundationData
    data_root = h5_root.parents[1]
    for key in ("split_csv_path", "stats_json_path", "processed_root"):
        v = cfg.get(key)
        if isinstance(v, str) and v:
            cfg[key] = _remap_data_path(v, data_root=data_root)
    return cfg


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    set_seed(args.seed)

    h5_path = _resolve_input_path(args.h5_path, Path(args.h5_root))
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    ckpt_dir = _resolve_input_path(args.checkpoint_dir, Path(args.checkpoints_root))
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

    cfg_json = Path(args.config_json) if args.config_json else (ckpt_dir / "analysis" / "config_used.json")
    cfg = _load_cfg_from_json(cfg_json)
    cfg = _apply_runtime_overrides(cfg, args)
    cfg = _remap_cfg_data_paths(cfg, Path(args.h5_root))
    cfg["ckpt_dir"] = str(ckpt_dir)

    selected_trials = _select_trial_datasets(h5_path, args.trial_dataset)
    print(f"[reconstruct_h5_local] selected {len(selected_trials)} trial dataset(s)")

    fs = cfg.get("frame_start", "NA")
    fe = cfg.get("frame_end", "NA")
    out_root = ckpt_dir / f"LOCAL_RECON_{_sanitize_tag(h5_path.stem)}_{fs}_{fe}"
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_ssl_model(cfg).to(device)
    ckpt_path = resolve_checkpoint_file(ckpt_dir, args.checkpoint_path)
    from src.experiments.mae_2d_lstm.checkpoint_utils import load_checkpoint_into_model

    try:
        load_checkpoint_into_model(model, ckpt_path, device)
    except Exception:
        load_checkpoint_into_model(model, ckpt_path, device, encoder_only=True)
    model.eval()
    print(f"[reconstruct_h5_local] loaded checkpoint: {ckpt_path}")

    logger = TBLogger(log_dir=str(ckpt_dir / "LOCAL_RECON_tb"))
    trainer = Trainer(model=model, logger=logger, cfg=cfg, device=device)

    per_trial_results: List[Dict[str, Any]] = []
    for i, (trial_dataset, shape) in enumerate(selected_trials, start=1):
        print(f"[reconstruct_h5_local] [{i}/{len(selected_trials)}] trial={trial_dataset!r} shape={shape}")
        trial_result = _evaluate_one_trial_dataset(
            model=model,
            trainer=trainer,
            cfg_base=cfg,
            args=args,
            h5_path=h5_path,
            trial_dataset=trial_dataset,
            shape=shape,
            out_root=out_root,
        )
        per_trial_results.append(trial_result)

    summary = {
        "h5_path": str(h5_path),
        "n_trial_datasets": len(per_trial_results),
        "trial_datasets": [x["trial_dataset"] for x in per_trial_results],
        "checkpoint_dir": str(ckpt_dir),
        "checkpoint_path": str(ckpt_path),
        "config_json": str(cfg_json),
        "model": cfg.get("model"),
        "frame_start": cfg.get("frame_start"),
        "frame_end": cfg.get("frame_end"),
        "clip_length": cfg.get("clip_length"),
        "mask_ratio": cfg.get("mask_ratio"),
        "out_root": str(out_root),
        "per_trial": per_trial_results,
    }
    with open(out_root / "local_recon_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"[reconstruct_h5_local] saved plots under: {out_root / 'plots'}")


if __name__ == "__main__":
    main()
