"""Per-pixel baseline mean/std over train split (frames 5–24 inclusive)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from data_pp_and_split.src.paths import resolve_h5_path


def parse_shape(shape_str: str) -> tuple[int, int]:
    s = str(shape_str).strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    parts = s.split(",")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse shape: {shape_str}")
    return int(parts[0]), int(parts[1])


def compute_per_pixel_baseline_stats(
    df_train: pd.DataFrame,
    baseline_frames_start: int = 5,
    baseline_frames_end: int = 25,
    *,
    allow_missing_h5: bool = False,
) -> dict:
    """
    Per-trial baseline: mean over baseline frame interval, then per-pixel mean/std across trials.

    ``baseline_frames_end`` follows the legacy notebook slice (exclusive), so frames 5–24.
    """
    baseline_frames: list[np.ndarray] = []
    skipped = 0

    grouped = df_train.groupby("target_file")
    for h5_path_str, group in grouped:
        h5_path = resolve_h5_path(h5_path_str)
        if not h5_path.exists():
            if allow_missing_h5:
                skipped += len(group)
                continue
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

        with h5py.File(h5_path, "r") as f:
            for _, row in group.iterrows():
                dset_name = row["trial_dataset"]
                n_pixels, n_frames = parse_shape(row["shape"])
                data = f[dset_name][...]
                if data.shape != (n_pixels, n_frames):
                    n_pixels, n_frames = data.shape

                baseline_vec = data[:, baseline_frames_start:baseline_frames_end].mean(axis=1)
                side = int(np.sqrt(n_pixels))
                if side * side != n_pixels:
                    raise ValueError(
                        f"Cannot reshape {n_pixels} pixels for trial {row['trial_global_id']}"
                    )
                baseline_frames.append(baseline_vec.reshape(side, side))

    if not baseline_frames:
        msg = "No baseline frames collected"
        if skipped:
            msg += f" ({skipped} train trials skipped due to missing H5)"
        raise RuntimeError(msg)

    if skipped:
        print(f"Warning: skipped {skipped} train trials with missing H5 files")

    stack = np.stack(baseline_frames, axis=0)
    mean_img = stack.mean(axis=0)
    std_img = stack.std(axis=0)
    eps = 1e-8
    std_img = np.where(std_img < eps, eps, std_img)

    return {
        "mean": mean_img[None, None, :, :].astype(np.float32),
        "std": std_img[None, None, :, :].astype(np.float32),
        "n_trials_used": len(baseline_frames),
        "n_trials_skipped": skipped,
    }
