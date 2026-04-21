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
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Subset

from src.experiments.mae_2d_lstm.build_dataloaders import build_dataloaders
from src.experiments.mae_2d_lstm.checkpoint_utils import resolve_checkpoint_file
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
    p.add_argument(
        "--plot-retinotopic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "If true, also evaluate a retinotopic test-only subset: target_file basename "
            "matching session_YYMMDDa_condsAN.h5. Saved under a separate subdir."
        ),
    )
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
    if args.plot_retinotopic is not None:
        o["plot_retinotopic"] = args.plot_retinotopic
    return o


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


def _select_sequence_vis_indices(
    dataset: Any,
    pool_indices: Sequence[int],
    seed: int,
    n_sequences: int = 2,
) -> List[int]:
    """
    Pick contiguous-in-time visualization indices from test set.
    For each selected sequence (trial), include all available frames in pool order.
    Returns a flat ordered list of dataset indices.
    """
    if not pool_indices:
        return []
    if not hasattr(dataset, "data_structure"):
        return list(pool_indices)[:10]

    # Group by trial row index, preserve frame order by clip_start
    by_trial: Dict[int, List[Tuple[int, int]]] = {}
    for ds_idx in pool_indices:
        row_idx, clip_start = dataset.data_structure[ds_idx]
        by_trial.setdefault(int(row_idx), []).append((int(clip_start), int(ds_idx)))
    if not by_trial:
        return list(pool_indices)[:10]

    for row_idx in by_trial:
        by_trial[row_idx].sort(key=lambda x: x[0])

    trial_ids = list(by_trial.keys())
    rng = random.Random(seed)
    rng.shuffle(trial_ids)
    chosen_trials = trial_ids[: min(n_sequences, len(trial_ids))]

    vis_indices: List[int] = []
    for trial_id in chosen_trials:
        vis_indices.extend([ds_idx for _clip_start, ds_idx in by_trial[trial_id]])

    # Keep plot size manageable but still informative
    return vis_indices[: max(10, len(chosen_trials) * 4)]


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
    ckpt_file = resolve_checkpoint_file(ckpt_dir, args.checkpoint_path)
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
    # Build a dedicated visualization loader:
    # - batch_size=1 so each batch contributes exactly one plotted sample
    # - choose at least 10 samples when available (or all if fewer)
    vis_indices = _select_sequence_vis_indices(
        dataset=test_loader.dataset,
        pool_indices=selected_indices,
        seed=args.seed,
        n_sequences=2,
    )
    n_vis = len(vis_indices)
    vis_subset = Subset(test_loader.dataset, vis_indices)
    vis_loader = DataLoader(
        vis_subset,
        batch_size=1,
        shuffle=False,
        num_workers=args.test_num_workers,
        pin_memory=cfg.get("pin_memory", False),
    )

    vis_dir = out_dir / "plots"
    vis_dir.mkdir(parents=True, exist_ok=True)
    save_test_reconstruction_figure(
        model=model,
        test_loader=vis_loader,
        device=device,
        out_dir=str(vis_dir),
        split_name=split_name,
        num_batches=n_vis,
        max_frames_per_clip=max_frames,
        plot_masked=False,
    )
    masked_plot_path = save_test_reconstruction_figure(
        model=model,
        test_loader=vis_loader,
        device=device,
        out_dir=str(vis_dir),
        split_name=split_name,
        num_batches=n_vis,
        max_frames_per_clip=max_frames,
        plot_masked=True,
    )
    if masked_plot_path is None:
        print(
            "[eval_specific_subset] Warning: reconstruction plots were not generated. "
            "Check matplotlib availability in the runtime."
        )
    else:
        print(
            f"[eval_specific_subset] Saved reconstruction plots in: {vis_dir} "
            f"(sequence frames plotted: {n_vis}, max_frames_per_clip: {max_frames})"
        )

    print(f"[eval_specific_subset] Done. Results saved to: {out_dir}")

    # Optional additional retinotopic subset analysis on test set only.
    # Match files where experiment suffix ends with 'a' before "_condsAN.h5",
    # e.g. session_230909a_condsAN.h5
    if bool(cfg.get("plot_retinotopic", False)):
        if not hasattr(test_loader.dataset, "trials") or not hasattr(test_loader.dataset, "data_structure"):
            print(
                "[eval_specific_subset] plot_retinotopic=true but dataset does not expose "
                "trials/data_structure; skipping retinotopic subset."
            )
            return

        pat = re.compile(r"session_\d{6}a_condsAN\.h5$")
        retino_indices: List[int] = []
        for i, (row_idx, _clip_start) in enumerate(test_loader.dataset.data_structure):
            tf = str(test_loader.dataset.trials.iloc[row_idx]["target_file"])
            if pat.search(Path(tf).name):
                retino_indices.append(i)

        if not retino_indices:
            print(
                "[eval_specific_subset] plot_retinotopic=true but no matching test files found "
                "(session_YYMMDDa_condsAN.h5)."
            )
            return

        retino_dir = out_dir / "RETINOTOPIC_A"
        retino_dir.mkdir(parents=True, exist_ok=True)

        retino_subset = Subset(test_loader.dataset, retino_indices)
        retino_loader = DataLoader(
            retino_subset,
            batch_size=args.batch_size or cfg.get("batch_size", 32),
            shuffle=False,
            num_workers=args.test_num_workers,
            pin_memory=cfg.get("pin_memory", False),
        )

        split_name_retino = "retinotopic_a_test"
        retino_eval = trainer.evaluate_metrics(retino_loader, split_name=split_name_retino)
        retino_temporal = trainer.evaluate_metrics_over_time(
            retino_loader,
            split_name=split_name_retino,
            save_dir=str(retino_dir),
        )

        # Retinotopic sample plots (same defaults as main subset: up to 10 samples)
        vis_indices_r = _select_sequence_vis_indices(
            dataset=test_loader.dataset,
            pool_indices=retino_indices,
            seed=args.seed,
            n_sequences=2,
        )
        n_vis_r = len(vis_indices_r)
        vis_subset_r = Subset(test_loader.dataset, vis_indices_r)
        vis_loader_r = DataLoader(
            vis_subset_r,
            batch_size=1,
            shuffle=False,
            num_workers=args.test_num_workers,
            pin_memory=cfg.get("pin_memory", False),
        )
        vis_dir_r = retino_dir / "plots"
        vis_dir_r.mkdir(parents=True, exist_ok=True)
        save_test_reconstruction_figure(
            model=model,
            test_loader=vis_loader_r,
            device=device,
            out_dir=str(vis_dir_r),
            split_name=split_name_retino,
            num_batches=n_vis_r,
            max_frames_per_clip=max_frames,
            plot_masked=False,
        )
        save_test_reconstruction_figure(
            model=model,
            test_loader=vis_loader_r,
            device=device,
            out_dir=str(vis_dir_r),
            split_name=split_name_retino,
            num_batches=n_vis_r,
            max_frames_per_clip=max_frames,
            plot_masked=True,
        )

        retino_summary = {
            "pattern": r"session_\\d{6}a_condsAN\\.h5$",
            "matched_indices": len(retino_indices),
            "eval_metrics": retino_eval,
            "temporal_start_frames": sorted(list(retino_temporal.keys())),
        }
        with open(retino_dir / "retinotopic_summary.json", "w") as f:
            json.dump(retino_summary, f, indent=2, default=str)

        print(
            f"[eval_specific_subset] Retinotopic subset done. Results saved to: {retino_dir} "
            f"(matched samples: {len(retino_indices)})"
        )


if __name__ == "__main__":
    main()

