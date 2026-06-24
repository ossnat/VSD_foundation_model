"""Build trial index rows from per-session H5 ``trial_metadata_json``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import h5py
import pandas as pd

from data_pp_and_split.src.paths import normalize_target_file, resolve_h5_path, workspace_root


def portable_target_file_for_h5(h5_path: Path) -> str:
    """Prefer workspace-relative ``Data/...`` path for the file on disk."""
    h5_path = Path(h5_path).resolve()
    try:
        return normalize_target_file(h5_path.relative_to(workspace_root()))
    except ValueError:
        return normalize_target_file(h5_path)


def format_shape(shape: str | list | tuple) -> str:
    if isinstance(shape, str):
        return shape
    if isinstance(shape, (list, tuple)) and len(shape) == 2:
        return f"({int(shape[0])}, {int(shape[1])})"
    raise ValueError(f"Cannot format shape: {shape!r}")


def rows_from_h5(h5_path: Path) -> list[dict]:
    """
    One index row per physical trial, using H5 dataset order and metadata condition.

    ``trial_{i:06d}`` must match the *i*-th entry in ``trial_metadata_json``.
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        if "trial_metadata_json" not in f.attrs:
            raise ValueError(f"Missing trial_metadata_json attribute: {h5_path}")

        meta = json.loads(f.attrs["trial_metadata_json"])
        keys = list(f.keys())
        if len(keys) != len(meta):
            raise ValueError(
                f"{h5_path.name}: {len(keys)} datasets but {len(meta)} metadata entries"
            )

        rows: list[dict] = []
        for i, (key, entry) in enumerate(zip(keys, meta)):
            expected = f"trial_{i:06d}"
            if key != expected:
                raise ValueError(f"{h5_path.name}: expected {expected}, found {key}")

            target = portable_target_file_for_h5(h5_path)
            rows.append(
                {
                    "trial_global_id": int(entry["trial_global_id"]),
                    "monkey": str(entry["monkey"]).strip().lower(),
                    "date": str(entry["date"]),
                    "condition": str(entry["condition"]),
                    "source_file": str(entry.get("source_file", "")),
                    "target_file": target,
                    "trial_index_in_condition": int(entry["trial_index_in_condition"]),
                    "shape": format_shape(entry["shape"]),
                    "trial_dataset": expected,
                }
            )
    return rows


def discover_h5_in_directory(h5_dir: Path | str) -> list[Path]:
    """All ``*.h5`` session files in a monkey/processed directory."""
    root = Path(h5_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"H5 directory not found: {root}")
    return sorted(p.resolve() for p in root.glob("*.h5") if p.is_file())


def rebuild_index_from_h5_paths(h5_paths: Iterable[Path | str]) -> pd.DataFrame:
    """Build index from explicit H5 paths (e.g. all files in ``ProcessedData/gandalf/``)."""
    rows: list[dict] = []
    paths = [Path(p) for p in h5_paths]
    if not paths:
        raise RuntimeError("No H5 files provided for index rebuild")

    for h5_path in paths:
        if not h5_path.exists():
            raise FileNotFoundError(f"HDF5 not found: {h5_path}")
        rows.extend(rows_from_h5(h5_path))

    df = pd.DataFrame(rows)
    df = df.sort_values("trial_global_id").reset_index(drop=True)
    validate_index(df, label="rebuilt index")
    print(f"Rebuilt index from {len(paths)} H5 files -> {len(df)} trials")
    return df


def discover_h5_target_files(index_path: Path | None = None) -> list[str]:
    """Unique portable ``target_file`` paths from an existing index CSV."""
    if index_path is None:
        from data_pp_and_split.src.paths import splits_dir

        index_path = splits_dir() / "all_trials_index.csv"

    df = pd.read_csv(index_path)
    return sorted(df["target_file"].astype(str).map(normalize_target_file).unique())


def rebuild_index_dataframe(
    target_files: Iterable[str],
    *,
    allow_missing_h5: bool = False,
) -> pd.DataFrame:
    rows: list[dict] = []
    missing: list[str] = []
    rebuilt = 0

    for target_file in target_files:
        h5_path = resolve_h5_path(target_file)
        if not h5_path.exists():
            if allow_missing_h5:
                missing.append(target_file)
                continue
            raise FileNotFoundError(f"HDF5 not found for index rebuild: {target_file} -> {h5_path}")

        rows.extend(rows_from_h5(h5_path))
        rebuilt += 1

    if not rows:
        msg = "No index rows built"
        if missing:
            msg += f" ({len(missing)} H5 files missing)"
        raise RuntimeError(msg)

    df = pd.DataFrame(rows)
    df = df.sort_values("trial_global_id").reset_index(drop=True)
    validate_index(df, label="rebuilt index")

    if missing:
        print(f"Warning: skipped {len(missing)} sessions with missing H5 files")

    print(f"Rebuilt index from {rebuilt} H5 files -> {len(df)} trials")
    return df


def validate_index(df: pd.DataFrame, *, label: str = "index") -> None:
    """Ensure one row per physical trial and consistent condition metadata."""
    required = {
        "trial_global_id",
        "monkey",
        "date",
        "condition",
        "target_file",
        "trial_dataset",
        "shape",
        "trial_index_in_condition",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")

    dup_physical = df.duplicated(subset=["target_file", "trial_dataset"], keep=False)
    if dup_physical.any():
        sample = (
            df.loc[dup_physical, ["target_file", "trial_dataset", "condition"]]
            .drop_duplicates()
            .head(5)
        )
        raise ValueError(
            f"{label}: duplicate physical trials (target_file, trial_dataset):\n{sample}"
        )

    dup_ids = df.duplicated(subset=["trial_global_id"], keep=False)
    if dup_ids.any():
        n = int(dup_ids.sum())
        raise ValueError(f"{label}: duplicate trial_global_id values ({n} rows)")
