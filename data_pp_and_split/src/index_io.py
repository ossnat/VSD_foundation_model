"""Load and normalize the all-trials index."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from data_pp_and_split.src.index_build import validate_index
from data_pp_and_split.src.paths import normalize_target_file, splits_dir

_CONDITION_RE = re.compile(r"conds?X?AN(\d+)", re.IGNORECASE)

REQUIRED_COLUMNS = {
    "trial_global_id",
    "monkey",
    "date",
    "condition",
    "target_file",
    "trial_dataset",
    "shape",
    "trial_index_in_condition",
}


def condition_to_int(condition: str | int) -> int:
    """Map ``condAN7`` -> 7 for v2-compatible CSV export."""
    if isinstance(condition, int) or (isinstance(condition, str) and condition.isdigit()):
        return int(condition)
    m = _CONDITION_RE.match(str(condition).strip())
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot map condition to int: {condition!r}")


def load_index(index_path: Path | None = None) -> pd.DataFrame:
    """Load ``all_trials_index.csv``, drop unknown monkeys, normalize paths."""
    if index_path is None:
        index_path = splits_dir() / "all_trials_index.csv"

    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")

    df = pd.read_csv(index_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Index missing required columns: {sorted(missing)}")

    df = df.copy()
    df["monkey"] = df["monkey"].astype(str).str.strip().str.lower()
    df = df[df["monkey"] != "unknown"].reset_index(drop=True)

    df["target_file"] = df["target_file"].map(normalize_target_file)
    df["group_key"] = (
        df["monkey"].astype(str)
        + "||"
        + df["date"].astype(str)
        + "||"
        + df["condition"].astype(str)
    )
    validate_index(df, label=str(index_path))
    return df
