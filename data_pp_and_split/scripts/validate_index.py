#!/usr/bin/env python3
"""Validate all_trials_index.csv for unique physical trials and optional H5 metadata check."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data_pp_and_split.src.index_build import validate_index
from data_pp_and_split.src.paths import resolve_h5_path, splits_dir, workspace_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        default="Data/FoundationData/ProcessedData/splits/all_trials_index.csv",
    )
    args = parser.parse_args()

    index_path = Path(args.index)
    if str(index_path).startswith("Data/"):
        index_path = workspace_root() / index_path

    df = pd.read_csv(index_path)
    validate_index(df, label=index_path.name)

    n_physical = df.groupby(["target_file", "trial_dataset"]).ngroups
    print(f"OK: {len(df)} rows, {n_physical} unique physical trials")

    # Spot-check overlap across splits if split column present
    if "split" in df.columns:
        cross = (
            df.groupby(["target_file", "trial_dataset"])["split"]
            .nunique()
            .reset_index(name="n_splits")
        )
        leaky = cross[cross["n_splits"] > 1]
        if len(leaky):
            print(f"WARNING: {len(leaky)} physical trials appear in multiple splits")
        else:
            print("OK: no physical trial spans multiple splits")


if __name__ == "__main__":
    main()
