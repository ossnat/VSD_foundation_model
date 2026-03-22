import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import pandas as pd

from src.data import load_dataset


def _filter_split_csv_missing_files(
    split_csv_path: str,
    project_root: Path,
) -> str:
    """
    Create a filtered copy of the split CSV that drops rows whose target_file
    does not exist on disk. Returns the path to the filtered CSV.

    This avoids hard failures when only a subset of the H5 files is present
    (e.g., on a laptop subset of the full dataset).
    """
    csv_path = Path(split_csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "target_file" not in df.columns:
        # Nothing to filter; return original path
        print("[build_dataloaders] 'target_file' column missing in split CSV; using original file.")
        return split_csv_path

    root_parent = project_root.parent
    exists_mask = []
    resolved_paths = []
    for tf in df["target_file"]:
        tf_path = Path(tf)
        if not tf_path.is_absolute():
            abs_path = (root_parent / tf_path).resolve()
        else:
            abs_path = tf_path
        exists_mask.append(abs_path.exists())
        resolved_paths.append(str(abs_path))

    exists_mask_series = pd.Series(exists_mask)
    kept = int(exists_mask_series.sum())
    total = len(exists_mask_series)
    if kept == total:
        print("[build_dataloaders] All target_file entries exist on disk; using original split CSV.")
        return split_csv_path

    print(
        f"[build_dataloaders] Warning: {total - kept} of {total} rows in split CSV refer to "
        f"missing H5 files. They will be skipped."
    )
    df_filtered = df[exists_mask_series.values].copy()
    # Overwrite target_file with absolute, existing paths so dataset code can open them
    df_filtered.loc[:, "target_file"] = [p for i, p in enumerate(resolved_paths) if exists_mask_series.iloc[i]]

    out_dir = project_root / "local_splits"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / csv_path.name
    df_filtered.to_csv(out_path, index=False)
    print(f"[build_dataloaders] Wrote filtered split CSV to {out_path}")
    return str(out_path)


def build_dataloaders(
    cfg: Dict[str, Any],
    project_root: Path,
    batch_size: Optional[int] = None,
    train_num_workers: int = 4,
    val_num_workers: int = 0,
    test_num_workers: int = 0,
) -> Tuple[Any, Any, Any]:
    """
    Build train/val/test DataLoaders for MAE 2D+LSTM experiments, with a
    safety mechanism to skip split rows whose H5 files are missing locally.

    Args:
        cfg: Flat config dict.
        project_root: Root of the VSD_foundation_model project.
        batch_size: Optional override; if None, uses cfg['batch_size'].
        train_num_workers, val_num_workers, test_num_workers: DataLoader workers.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Filter the split CSV once, then reuse for all splits
    split_csv_path = cfg.get("split_csv_path")
    if split_csv_path:
        filtered_split = _filter_split_csv_missing_files(split_csv_path, project_root)
        cfg = dict(cfg)  # shallow copy
        cfg["split_csv_path"] = filtered_split

    effective_batch_size = batch_size or cfg.get("batch_size", 32)

    train_loader = load_dataset(
        cfg,
        split="train",
        batch_size=effective_batch_size,
        num_workers=train_num_workers,
        shuffle=True,
    )
    val_loader = load_dataset(
        cfg,
        split="val",
        batch_size=effective_batch_size,
        num_workers=val_num_workers,
        shuffle=False,
    )
    test_loader = load_dataset(
        cfg,
        split="test",
        batch_size=effective_batch_size,
        num_workers=test_num_workers,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader

