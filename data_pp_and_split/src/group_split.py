"""Session×condition group-level train/val/test assignment."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import pandas as pd

SplitName = Literal["train", "val", "test"]

GROUP_COLS = ["monkey", "date", "condition"]


def _group_key(row: pd.Series) -> str:
    return f"{row['monkey']}||{row['date']}||{row['condition']}"


def _normalize_manual_entry(entry: Dict[str, Any]) -> Tuple[str, str, str]:
    monkey = str(entry.get("monkey", "")).strip().lower()
    date = str(entry.get("date") or entry.get("session", "")).strip()
    condition = str(entry.get("condition", "")).strip()
    if not monkey or not date or not condition:
        raise ValueError(f"Manual assignment missing fields: {entry}")
    return monkey, date, condition


def manual_group_keys(manual: Dict[str, List[Dict[str, Any]]]) -> Dict[str, SplitName]:
    """Build group_key -> split from YAML manual_assignments."""
    out: Dict[str, SplitName] = {}
    for split in ("train", "val", "test"):
        for entry in manual.get(split, []) or []:
            monkey, date, condition = _normalize_manual_entry(entry)
            key = f"{monkey}||{date}||{condition}"
            if key in out and out[key] != split:
                raise ValueError(f"Group {key} assigned to both {out[key]} and {split}")
            out[key] = split  # type: ignore[assignment]
    return out


def unique_groups(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (monkey, date, condition) with trial count."""
    return (
        df.groupby(GROUP_COLS, as_index=False)
        .agg(n_trials=("trial_global_id", "count"))
        .assign(group_key=lambda g: g.apply(_group_key, axis=1))
    )


def assign_splits(
    df: pd.DataFrame,
    *,
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    manual: Dict[str, List[Dict[str, Any]]] | None = None,
) -> pd.DataFrame:
    """
    Assign split labels at session×condition group granularity.

    Manual assignments are applied first; remaining groups get 75/15/10 (configurable).
    """
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0, got {total}")

    manual = manual or {}
    pinned = manual_group_keys(manual)

    groups = unique_groups(df)
    group_split: Dict[str, SplitName] = {}

    for _, row in groups.iterrows():
        key = row["group_key"]
        if key in pinned:
            group_split[key] = pinned[key]

    remaining = groups[~groups["group_key"].isin(group_split)].copy()
    if len(remaining) > 0:
        rng = np.random.default_rng(seed)
        keys = remaining["group_key"].tolist()
        rng.shuffle(keys)

        n = len(keys)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        n_train = n - n_val - n_test
        if n_train < 0:
            n_train = 0
            n_val = max(0, n - n_test)
        if n_train + n_val + n_test > n:
            n_test = max(0, n - n_train - n_val)

        for key in keys[:n_train]:
            group_split[key] = "train"
        for key in keys[n_train : n_train + n_val]:
            group_split[key] = "val"
        for key in keys[n_train + n_val :]:
            group_split[key] = "test"

    out = df.copy()
    out["split"] = out["group_key"].map(group_split)
    if out["split"].isna().any():
        missing = out[out["split"].isna()]["group_key"].unique()
        raise RuntimeError(f"Unassigned groups: {missing[:5]}")

    return out


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Trial counts per split and monkey."""
    return (
        df.groupby(["split", "monkey"], as_index=False)
        .agg(n_trials=("trial_global_id", "count"), n_groups=("group_key", "nunique"))
        .sort_values(["split", "monkey"])
    )


def group_leak_check(df: pd.DataFrame) -> None:
    """Ensure each group appears in exactly one split."""
    per_group = df.groupby("group_key")["split"].nunique()
    bad = per_group[per_group > 1]
    if len(bad):
        raise AssertionError(f"Groups span multiple splits: {bad.index.tolist()[:5]}")


def physical_split_leak_check(df: pd.DataFrame) -> None:
    """Ensure each (H5, trial_dataset) appears in at most one split."""
    per_trial = df.groupby(["target_file", "trial_dataset"])["split"].nunique()
    bad = per_trial[per_trial > 1]
    if len(bad):
        raise AssertionError(
            f"Physical trials span multiple splits ({len(bad)} cases), e.g. {bad.index[:3].tolist()}"
        )
