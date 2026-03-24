#!/usr/bin/env python3
"""
Evaluate a trained MAE model on a specific subset of test samples.

This script reuses existing evaluation functionality:
- Trainer.evaluate_metrics
- Trainer.evaluate_metrics_over_time
- save_test_reconstruction_figure

Outputs are saved under:
  <checkpoint_dir>/SPECIFIC_RES_<start_frame>_<end_frame>/
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Subset

from src.experiments.mae_2d_lstm.build_dataloaders import build_dataloaders
from src.experiments.mae_2d_lstm.load_config import load_and_prepare_config
from src.experiments.mae_2d_lstm.vis_test_reconstruction import save_test_reconstruction_figure
from src.models import build_ssl_model
from src.training.trainer import Trainer
from src.utils.logger import TBLogger, set_seed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained MAE model on specific test subsets.")
    p.add_argument("--config", type=str, required=True, help="Config path used to build model/data.")
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Absolute path to checkpoint dir containing model checkpoints.",
    )
    p.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Optional explicit checkpoint file path. If omitted, auto-selects best available.",
    )
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
    p.add_argument(
        "--target-files",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of target_file paths (absolute or basename) to evaluate from test set.",
    )
    p.add_argument(
        "--subset-size",
        type=int,
        default=200,
        help="Random subset size when --target-files is empty.",
    )
    p.add_argument("--seed", type=int, default=17)
    return p.parse_args()


def _build_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    o: Dict[str, Any] = {}
    if args.monkeys:
        o["monkeys"] = args.monkeys
    if args.mask_ratio is not None:
        o["mask_ratio"] = args.mask_ratio
    if args.clip_length is not None:
        o["clip_length"] = args.clip_length
    if args.patch_size is not None:
        o["patch_size"] = list(args.patch_size)
    if args.frame_start is not None:
        o["frame_start"] = args.frame_start
    if args.frame_end is not None:
        o["frame_end"] = args.frame_end
    if args.batch_size is not None:
        o["batch_size"] = args.batch_size
    if args.seed is not None:
        o["seed"] = args.seed
    return o


def _find_checkpoint_file(ckpt_dir: Path, explicit: Optional[str]) -> Path:
    if explicit is not None:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"checkpoint-path not found: {p}")
        return p

    model_final = ckpt_dir / "model_final.pt"
    if model_final.exists():
        return model_final

    epoch_files = list(ckpt_dir.glob("epoch_*.pt"))
    if epoch_files:
        def _epoch_num(path: Path) -> int:
            stem = path.stem  # epoch_10
            try:
                return int(stem.split("_")[-1])
            except Exception:
                return -1
        return sorted(epoch_files, key=_epoch_num)[-1]

    encoder_final = ckpt_dir / "encoder_final.pt"
    if encoder_final.exists():
        return encoder_final

    raise FileNotFoundError(
        f"No checkpoint file found in {ckpt_dir}. Expected model_final.pt, epoch_*.pt, or encoder_final.pt."
    )


def _select_test_indices(
    dataset: Any,
    target_files: Optional[Sequence[str]],
    subset_size: int,
    seed: int,
) -> Tuple[List[int], Dict[str, Any]]:
    total = len(dataset)
    if total == 0:
        return [], {"mode": "empty", "matched_files": 0}

    if target_files:
        wanted_abs = set()
        wanted_base = set()
        for tf in target_files:
            p = Path(tf)
            wanted_base.add(p.name)
            if p.is_absolute():
                wanted_abs.add(str(p.resolve()))

        selected: List[int] = []
        if not hasattr(dataset, "trials") or not hasattr(dataset, "data_structure"):
            raise ValueError(
                "Dataset does not expose trials/data_structure needed for --target-files filtering."
            )
        for i, (row_idx, _clip_start) in enumerate(dataset.data_structure):
            tf = str(dataset.trials.iloc[row_idx]["target_file"])
            tf_abs = str(Path(tf).resolve())
            tf_base = Path(tf).name
            if tf_abs in wanted_abs or tf_base in wanted_base:
                selected.append(i)

        meta = {
            "mode": "target_files",
            "requested_files": list(target_files),
            "matched_indices": len(selected),
            "matched_files": len({Path(str(dataset.trials.iloc[dataset.data_structure[i][0]]["target_file"])).name for i in selected}) if selected else 0,
        }
        return selected, meta

    # Random subset mode
    rng = random.Random(seed)
    k = min(max(subset_size, 1), total)
    selected = list(range(total))
    rng.shuffle(selected)
    selected = selected[:k]
    return selected, {"mode": "random_subset", "subset_size": k, "total_test_samples": total}


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)

    ckpt_dir = Path(args.checkpoint_dir).resolve()
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"checkpoint-dir not found: {ckpt_dir}")

    project_root = Path(__file__).resolve().parent.parent
    cfg = load_and_prepare_config(
        base_cfg_path=args.config,
        project_root=project_root,
        data_root=None,
        overrides=_build_overrides(args),
    )
    cfg["ckpt_dir"] = str(ckpt_dir)

    sf = cfg.get("frame_start", "NA")
    ef = cfg.get("frame_end", "NA")
    out_dir = ckpt_dir / f"SPECIFIC_RES_{sf}_{ef}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_snapshot = out_dir / "config_used.json"
    with open(cfg_snapshot, "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval_specific_subset] Using device: {device}")

    _train_loader, _val_loader, test_loader = build_dataloaders(
        cfg,
        project_root=project_root,
        batch_size=args.batch_size,
        train_num_workers=args.train_num_workers,
        val_num_workers=args.val_num_workers,
        test_num_workers=args.test_num_workers,
    )

    selected_indices, subset_meta = _select_test_indices(
        dataset=test_loader.dataset,
        target_files=args.target_files,
        subset_size=args.subset_size,
        seed=args.seed,
    )
    if not selected_indices:
        raise RuntimeError("No test samples selected for evaluation.")

    subset = Subset(test_loader.dataset, selected_indices)
    subset_loader = DataLoader(
        subset,
        batch_size=args.batch_size or cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=args.test_num_workers,
        pin_memory=cfg.get("pin_memory", False),
    )

    model = build_ssl_model(cfg).to(device)
    ckpt_file = _find_checkpoint_file(ckpt_dir, args.checkpoint_path)
    state = torch.load(ckpt_file, map_location=device)

    loaded_mode = "full_model_state_dict"
    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        # fallback for encoder-only checkpoints
        if hasattr(model, "encoder"):
            model.encoder.load_state_dict(state, strict=True)
            loaded_mode = "encoder_only_state_dict"
        else:
            raise
    print(f"[eval_specific_subset] Loaded checkpoint: {ckpt_file} ({loaded_mode})")

    logger = TBLogger(log_dir=str(out_dir / "tb_logs"))
    trainer = Trainer(model=model, logger=logger, cfg=cfg, device=device)

    split_name = "specific_test"
    eval_metrics = trainer.evaluate_metrics(subset_loader, split_name=split_name)
    temporal_metrics = trainer.evaluate_metrics_over_time(
        subset_loader,
        split_name=split_name,
        save_dir=str(out_dir),
    )

    # Save a concise analysis summary
    summary = {
        "checkpoint_file": str(ckpt_file),
        "checkpoint_load_mode": loaded_mode,
        "selection": subset_meta,
        "selected_indices": len(selected_indices),
        "eval_metrics": eval_metrics,
        "temporal_start_frames": sorted(list(temporal_metrics.keys())),
    }
    with open(out_dir / "subset_eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    model_type = cfg.get("model", "mae_2d")
    max_frames = 4 if model_type == "mae_2d_lstm" else 1
    num_batches = 10
    vis_dir = out_dir / "temporal_eval"
    save_test_reconstruction_figure(
        model=model,
        test_loader=subset_loader,
        device=device,
        out_dir=str(vis_dir),
        split_name=split_name,
        num_batches=num_batches,
        max_frames_per_clip=max_frames,
        plot_masked=False,
    )
    save_test_reconstruction_figure(
        model=model,
        test_loader=subset_loader,
        device=device,
        out_dir=str(vis_dir),
        split_name=split_name,
        num_batches=num_batches,
        max_frames_per_clip=max_frames,
        plot_masked=True,
    )

    print(f"[eval_specific_subset] Done. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()

