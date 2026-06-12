"""Write v3 split CSV and baseline stats artifacts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import h5py
import pandas as pd

from data_pp_and_split.src.index_io import condition_to_int


def merge_shutter_off(df: pd.DataFrame, legacy_split_csv: Path | None) -> pd.DataFrame:
    """Copy ``shutter_off`` from a legacy split CSV by ``trial_global_id``."""
    out = df.copy()
    if legacy_split_csv is None or not legacy_split_csv.exists():
        out["shutter_off"] = float("nan")
        return out

    legacy = pd.read_csv(legacy_split_csv, usecols=["trial_global_id", "shutter_off"])
    out = out.merge(legacy, on="trial_global_id", how="left", suffixes=("", "_legacy"))
    if "shutter_off_legacy" in out.columns:
        out["shutter_off"] = out["shutter_off"].fillna(out["shutter_off_legacy"])
        out = out.drop(columns=["shutter_off_legacy"])
    return out


def build_split_export_df(df: pd.DataFrame, legacy_split_csv: Path | None) -> pd.DataFrame:
    """Format dataframe to match v2 split CSV schema."""
    out = df.copy()
    out = merge_shutter_off(out, legacy_split_csv)
    out["condition"] = out["condition"].map(condition_to_int)
    cols = [
        "trial_global_id",
        "monkey",
        "date",
        "condition",
        "trial_index_in_condition",
        "trial_dataset",
        "shutter_off",
        "shape",
        "target_file",
        "split",
    ]
    return out[cols].sort_values("trial_global_id").reset_index(drop=True)


def artifact_basename(version: str, seed: int, stratify_level: str) -> str:
    return f"{version}_seed{seed}_{stratify_level}"


def save_split_csv(df: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def save_baseline_stats_h5(stats: dict, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("mean", data=stats["mean"], compression="gzip")
        f.create_dataset("std", data=stats["std"], compression="gzip")
    return out_path


def save_stats_json(
    *,
    out_path: Path,
    meta: Dict[str, Any],
    stats: dict,
    split_csv_filename: str,
    stats_h5_filename: str,
) -> Path:
    mean_arr = stats["mean"]
    std_arr = stats["std"]
    record = {
        "created": datetime.now().isoformat(),
        **meta,
        "mean_shape": list(mean_arr.shape),
        "stats_h5_path": stats_h5_filename,
        "split_csv_path": split_csv_filename,
        "mean_min": float(mean_arr.min()),
        "mean_max": float(mean_arr.max()),
        "std_min": float(std_arr.min()),
        "std_max": float(std_arr.max()),
    }
    if "n_trials_used" in stats:
        record["n_train_trials_stats"] = int(stats["n_trials_used"])
    if stats.get("n_trials_skipped"):
        record["n_train_trials_skipped_missing_h5"] = int(stats["n_trials_skipped"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    return out_path
