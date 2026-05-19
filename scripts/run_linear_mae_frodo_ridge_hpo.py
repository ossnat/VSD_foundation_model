#!/usr/bin/env python3
"""
Frodo-only linear MAE 2D (ridge): build split CSV from local .h5 files, estimate mean/std,
coarse HPO on val loss, then train+eval+plots into checkpoints_linear_mae2d.

Usage (from repo root):
  PYTHONPATH=. python3 scripts/run_linear_mae_frodo_ridge_hpo.py
  PYTHONPATH=. python3 scripts/run_linear_mae_frodo_ridge_hpo.py --quick   # fast HPO subset + short epochs
  PYTHONPATH=. python3 scripts/run_linear_mae_frodo_ridge_hpo.py --hpo-max-rows 200   # cap HPO data only; final uses full split

Requires: torch, h5py, pandas, tqdm, matplotlib, pyyaml, tensorboard (same stack as MAE training).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DEFAULT_FRODO_DIR = (
    _PROJECT_ROOT / "Data" / "FoundationData" / "ProcessedData" / "frodo"
)
DEFAULT_OUT_DIR = _PROJECT_ROOT / "checkpoints_linear_mae2d"
MONKEY = "frodo"

# Ridge-only (L1 on weights = 0)
RIDGE_GRID: List[Dict[str, float]] = [
    {"lr": 1e-3, "weight_decay": 1e-2},
    {"lr": 5e-3, "weight_decay": 1e-2},
    {"lr": 5e-3, "weight_decay": 5e-2},
    {"lr": 1e-2, "weight_decay": 2e-2},
]

DEFAULT_HPO_EPOCHS = 5
DEFAULT_FINAL_EPOCHS = 35
HPO_SEED = 17


def _is_square_pixels_shape(shape: Tuple[int, ...]) -> bool:
    if len(shape) != 2:
        return False
    n_pixels = int(shape[0])
    side = int(n_pixels**0.5)
    return side * side == n_pixels and shape[1] >= 1


def _collect_trial_datasets(h5_path: Path) -> List[Tuple[str, Tuple[int, ...]]]:
    import h5py

    out: List[Tuple[str, Tuple[int, ...]]] = []
    with h5py.File(h5_path, "r") as h5:

        def visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset) and re.match(r"^trial_\d+$", name.split("/")[-1]):
                out.append((name, tuple(obj.shape)))

        h5.visititems(visitor)
    return [(n.split("/")[-1], s) for n, s in out if _is_square_pixels_shape(s)]


def _discover_rows(frodo_dir: Path, monkey: str) -> pd.DataFrame:
    rows = []
    for h5_path in sorted(frodo_dir.glob("*.h5")):
        if not h5_path.is_file():
            continue
        stem = h5_path.stem.lower()
        if stem.startswith("vsd_"):
            continue
        try:
            trials = _collect_trial_datasets(h5_path)
        except Exception as e:
            print(f"[skip] {h5_path.name}: {e}")
            continue
        for trial_ds, shape in trials:
            rows.append(
                {
                    "target_file": str(h5_path.resolve()),
                    "trial_dataset": trial_ds,
                    "shape": str((int(shape[0]), int(shape[1]))),
                    "monkey": monkey,
                    "date": h5_path.stem,
                    "condition": "condsAN",
                }
            )
    if not rows:
        raise RuntimeError(f"No trial_* datasets found under {frodo_dir}")
    return pd.DataFrame(rows)


def _assign_splits(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
    df = df.copy()
    df["split"] = splits
    return df


def _subset_for_hpo(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    """Stratified downsample so HPO epochs finish quickly; final train uses the full CSV."""
    if max_rows <= 0 or len(df) <= max_rows:
        return df
    rng = np.random.default_rng(seed)
    parts: List[pd.DataFrame] = []
    total = len(df)
    for sp, g in df.groupby("split"):
        n_take = max(1, int(round(max_rows * len(g) / total)))
        n_take = min(n_take, len(g))
        idx = rng.choice(len(g), size=n_take, replace=False)
        parts.append(g.iloc[idx])
    out = pd.concat(parts, ignore_index=True)
    if len(out) > max_rows:
        idx = rng.choice(len(out), size=max_rows, replace=False)
        out = out.iloc[idx].reset_index(drop=True)
    return out


def _spatial_mean_std(
    df_train: pd.DataFrame, frame_start: int, frame_end: int
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    import h5py

    sum_img: np.ndarray | None = None
    sumsq_img: np.ndarray | None = None
    n = 0
    H = W = 0
    for _, row in df_train.iterrows():
        with h5py.File(row["target_file"], "r") as f:
            arr = f[row["trial_dataset"]][...]
        n_pix, n_frames = arr.shape
        H = W = int(math.sqrt(n_pix))
        fe = min(frame_end, n_frames - 1)
        fs = min(frame_start, fe)
        for t in range(fs, fe + 1):
            img = arr[:, t].reshape(H, W).astype(np.float64)
            if sum_img is None:
                sum_img = np.zeros((H, W), dtype=np.float64)
                sumsq_img = np.zeros((H, W), dtype=np.float64)
            sum_img += img
            sumsq_img += img * img
            n += 1
    if n == 0 or sum_img is None:
        raise RuntimeError("No frames accumulated for mean/std; check frame_start/frame_end.")
    mean = (sum_img / n).astype(np.float32)
    ex2 = (sumsq_img / n).astype(np.float64)
    var = np.maximum(ex2 - mean.astype(np.float64) ** 2, 1e-12)
    std = np.sqrt(var).astype(np.float32)
    mean_b = mean[np.newaxis, np.newaxis, :, :]
    std_b = std[np.newaxis, np.newaxis, :, :]
    return mean_b, std_b, H, W


def _write_stats(stats_h5: Path, mean_b: np.ndarray, std_b: np.ndarray) -> None:
    import h5py

    stats_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(stats_h5, "w") as f:
        f.create_dataset("mean", data=mean_b, compression="gzip")
        f.create_dataset("std", data=std_b, compression="gzip")


def _patch_trim_hw(h: int, w: int, patch_size: Tuple[int, int, int]) -> Tuple[int, int]:
    p_h, p_w = int(patch_size[1]), int(patch_size[2])
    return (h // p_h) * p_h, (w // p_w) * p_w


def _base_cfg(
    split_csv: Path,
    stats_json: Path,
    ckpt_dir: Path,
    log_dir: Path,
    results_dir: Path,
    linear_hw: Tuple[int, int],
    artifact_root: Path,
    monkey: str,
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    patch_size = [1, 8, 8]
    cfg: Dict[str, Any] = {
        "model": "linear_mae_2d",
        "channels": 1,
        "linear_spatial_hw": [linear_hw[0], linear_hw[1]],
        "monkeys": [monkey],
        "dataset_name": "vsd_mae",
        "split_csv_path": str(split_csv.resolve()),
        "stats_json_path": str(stats_json.resolve()),
        "processed_root": str(artifact_root / "analysis" / "processed_stub"),
        "frame_start": 30,
        "frame_end": 100,
        "clip_length": 1,
        "mask_ratio": 0.5,
        "patch_size": patch_size,
        "crop_frame": None,
        "crop_radius": None,
        "val_frame_stride": 1,
        "batch_size": 32,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": 2,
        "normalize_loss": True,
        "crop_loss": None,
        "crop_loss_radius": 30,
        "loss_type": "mse",
        "alpha": 0.84,
        "ssim_window_size": 11,
        "ssim_sigma": 1.5,
        "linear_l1_penalty": 0.0,
        "epochs": DEFAULT_FINAL_EPOCHS,
        "max_grad_norm": 1.0,
        "use_amp": False,
        "ckpt_dir": str(ckpt_dir.resolve()),
        "log_dir": str(log_dir.resolve()),
        "results_dir": str(results_dir.resolve()),
        "seed": HPO_SEED,
        "preload_into_ram": False,
        "plot_retinotopic": False,
        "temporal_eval_per_metric_plots": True,
    }
    cfg.update(overrides)
    return cfg


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Frodo local H5 linear MAE ridge + final train.")
    p.add_argument(
        "--frodo-dir",
        type=str,
        default=str(DEFAULT_FRODO_DIR),
        help="Directory containing session_*.h5 Frodo files.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Final checkpoint + analysis root (default: ./checkpoints_linear_mae2d).",
    )
    p.add_argument("--hpo-epochs", type=int, default=DEFAULT_HPO_EPOCHS)
    p.add_argument("--final-epochs", type=int, default=DEFAULT_FINAL_EPOCHS)
    p.add_argument(
        "--quick",
        action="store_true",
        help="Shorter HPO/final epochs (2 + 4) and default --hpo-max-rows 220 if that flag is 0.",
    )
    p.add_argument(
        "--hpo-max-rows",
        type=int,
        default=0,
        help="Max CSV rows for HPO phase only (0 = use full split). Recommended 150–300 on CPU.",
    )
    p.add_argument(
        "--monkey",
        type=str,
        default=MONKEY,
        help="Value written in CSV 'monkey' column and used for cfg['monkeys'] filter.",
    )
    return p.parse_args()


def main() -> None:
    import torch

    args = _parse_args()
    monkey = str(args.monkey).strip() or MONKEY
    hpo_epochs = 2 if args.quick else int(args.hpo_epochs)
    # Full Frodo train is large on CPU; keep final epochs low in --quick smoke mode.
    final_epochs = 2 if args.quick else int(args.final_epochs)
    hpo_max_rows = int(args.hpo_max_rows)
    if args.quick and hpo_max_rows == 0:
        hpo_max_rows = 220

    from src.experiments.mae_2d_lstm.build_dataloaders import build_dataloaders
    from src.experiments.mae_2d_lstm.run_training import run_training_and_temporal_eval
    from src.models import build_ssl_model
    from src.training.trainer import Trainer
    from src.utils.logger import TBLogger, set_seed

    frodo_dir = Path(args.frodo_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not frodo_dir.is_dir():
        raise FileNotFoundError(f"Frodo directory not found: {frodo_dir}")

    analysis = out_dir / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    (analysis / "processed_stub").mkdir(parents=True, exist_ok=True)
    hpo_root = out_dir / "hpo_sweep"
    hpo_root.mkdir(parents=True, exist_ok=True)

    print("[1/5] Discovering trials in", frodo_dir, "monkey=", monkey)
    df_all = _discover_rows(frodo_dir, monkey=monkey)
    df_all = _assign_splits(df_all, HPO_SEED)
    split_csv = analysis / "frodo_local_split.csv"
    df_all.to_csv(split_csv, index=False)
    print(f"  Wrote {len(df_all)} rows -> {split_csv}")

    hpo_split_csv = split_csv
    if hpo_max_rows > 0:
        df_hpo = _subset_for_hpo(df_all, hpo_max_rows, HPO_SEED)
        hpo_split_csv = analysis / "frodo_hpo_subset.csv"
        df_hpo.to_csv(hpo_split_csv, index=False)
        print(f"  HPO will use {len(df_hpo)} rows (subset) -> {hpo_split_csv}")

    df_train = df_all[df_all["split"] == "train"]
    frame_start, frame_end = 30, 100
    print("[2/5] Estimating spatial mean/std on train clips (this can take a minute)...")
    mean_b, std_b, H, W = _spatial_mean_std(df_train, frame_start, frame_end)
    patch_size = (1, 8, 8)
    th, tw = _patch_trim_hw(H, W, patch_size)
    print(f"  Raw spatial {H}x{W} -> patch-trimmed {th}x{tw} for linear layer")

    stats_h5 = analysis / "frodo_train_mean_std.h5"
    _write_stats(stats_h5, mean_b, std_b)
    stats_json = analysis / "frodo_train_stats.json"
    with open(stats_json, "w") as f:
        json.dump(
            {
                "mean_shape": [1, 1, int(mean_b.shape[2]), int(mean_b.shape[3])],
                "stats_h5_path": str(stats_h5.resolve()),
            },
            f,
            indent=2,
        )
    print(f"  Stats -> {stats_h5}, {stats_json}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        "[3/5] Ridge HPO on val loss (epochs=%d, hpo_csv_rows=%s) device=%s"
        % (hpo_epochs, hpo_split_csv.name, device)
    )

    best: Dict[str, Any] | None = None
    best_score = float("inf")

    for i, h in enumerate(RIDGE_GRID):
        set_seed(HPO_SEED)
        run_dir = hpo_root / f"run_{i:02d}_lr{h['lr']}_wd{h['weight_decay']}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = _base_cfg(
            hpo_split_csv,
            stats_json,
            ckpt_dir=run_dir / "ckpt",
            log_dir=run_dir / "logs",
            results_dir=run_dir / "results",
            linear_hw=(th, tw),
            artifact_root=out_dir,
            monkey=monkey,
            overrides={
                "lr": h["lr"],
                "weight_decay": h["weight_decay"],
                "linear_l1_penalty": 0.0,
                "epochs": hpo_epochs,
                "val_frame_stride": 2 if args.quick else 1,
            },
        )
        train_loader, val_loader, test_loader = build_dataloaders(
            cfg,
            project_root=_PROJECT_ROOT,
            batch_size=cfg["batch_size"],
            train_num_workers=0,
            val_num_workers=0,
            test_num_workers=0,
        )
        model = build_ssl_model(cfg).to(device)
        logger = TBLogger(log_dir=cfg["log_dir"])
        trainer = Trainer(model=model, logger=logger, cfg=cfg, device=device)
        trainer.fit(train_loader, val_loader)
        val_metrics = trainer.evaluate_metrics(val_loader, split_name="val")
        score = float(val_metrics.get("loss", val_metrics.get("mse_masked", 1e9)))
        rec = {"run_index": i, "hparams": h, "val_score": score, "metrics_val": val_metrics}
        with open(run_dir / "hpo_trial.json", "w") as f:
            json.dump(rec, f, indent=2, default=str)
        print(f"  trial {i}: lr={h['lr']} wd={h['weight_decay']} val_loss={score:.6f}")
        if score < best_score:
            best_score = score
            best = rec

    assert best is not None
    best_path = analysis / "best_ridge_hpo.json"
    with open(best_path, "w") as f:
        json.dump({"best_val_loss": best_score, "best": best}, f, indent=2, default=str)
    print("[4/5] Best HPO:", json.dumps(best["hparams"]), "val_loss=", best_score)

    bh = best["hparams"]
    best_yaml = analysis / "best_linear_mae_frodo_ridge.yaml"
    yaml_lines = [
        "# Auto-written by run_linear_mae_frodo_ridge_hpo.py — merge into linear_mae_2d_full.yaml if desired.",
        f"lr: {bh['lr']}",
        f"weight_decay: {bh['weight_decay']}",
        "linear_l1_penalty: 0.0",
        f"linear_spatial_hw: [{th}, {tw}]",
        f"monkeys: ['{monkey}']",
    ]
    best_yaml.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    print(f"  Wrote {best_path} and {best_yaml}")

    print("[5/5] Final training + temporal plots + recon figures ->", out_dir)
    set_seed(HPO_SEED)
    final_cfg = _base_cfg(
        split_csv,
        stats_json,
        ckpt_dir=out_dir,
        log_dir=out_dir / "logs_final",
        results_dir=out_dir,
        linear_hw=(th, tw),
        artifact_root=out_dir,
        monkey=monkey,
        overrides={
            "lr": bh["lr"],
            "weight_decay": bh["weight_decay"],
            "linear_l1_penalty": 0.0,
            "epochs": final_epochs,
            "val_frame_stride": 2 if args.quick else 1,
        },
    )
    with open(analysis / "config_final_frodo_linear.json", "w") as f:
        json.dump(final_cfg, f, indent=2, default=str)

    train_loader, val_loader, test_loader = build_dataloaders(
        final_cfg,
        project_root=_PROJECT_ROOT,
        batch_size=final_cfg["batch_size"],
        train_num_workers=0,
        val_num_workers=0,
        test_num_workers=0,
    )
    model = build_ssl_model(final_cfg).to(device)
    run_training_and_temporal_eval(
        cfg=final_cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
    )
    print("Done. Artifacts under:", out_dir.resolve())


if __name__ == "__main__":
    main()
