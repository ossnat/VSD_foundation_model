#!/usr/bin/env python3
"""Compare v3 session×condition group split to legacy v2 session split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data_pp_and_split.src.paths import workspace_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v3",
        default="Data/FoundationData/ProcessedData/splits/split_v3_seed17_session_condition_group.csv",
    )
    parser.add_argument(
        "--v2",
        default="Data/FoundationData/ProcessedData/splits/split_v2_seed17_session_split.csv",
    )
    args = parser.parse_args()

    v3_path = workspace_root() / args.v3 if args.v3.startswith("Data/") else Path(args.v3)
    v2_path = workspace_root() / args.v2 if args.v2.startswith("Data/") else Path(args.v2)

    v3 = pd.read_csv(v3_path)
    v2 = pd.read_csv(v2_path)

    merged = v3.merge(
        v2[["trial_global_id", "split"]].rename(columns={"split": "split_v2"}),
        on="trial_global_id",
        how="inner",
    )
    merged["agree"] = merged["split"] == merged["split_v2"]

    print(f"v3: {v3_path.name}  ({len(v3)} trials)")
    print(f"v2: {v2_path.name}  ({len(v2)} trials)")
    print(f"\nTrial-level agreement: {merged['agree'].mean():.1%} ({merged['agree'].sum()}/{len(merged)})")
    print("\nv3 split counts:")
    print(v3["split"].value_counts().to_string())
    print("\nv2 split counts:")
    print(v2["split"].value_counts().to_string())

    v3["group"] = (
        v3["monkey"].astype(str)
        + "||"
        + v3["date"].astype(str)
        + "||"
        + v3["condition"].astype(str)
    )
    g = v3.groupby("group")["split"].nunique()
    leaky = g[g > 1]
    print(f"\nGroup leakage in v3: {len(leaky)} groups span multiple splits")

    v2["session"] = v2["monkey"].astype(str) + "||" + v2["date"].astype(str)
    sess_splits = v2.groupby("session")["split"].nunique()
    print(f"v2 sessions spanning multiple splits: {(sess_splits > 1).sum()} / {len(sess_splits)}")


if __name__ == "__main__":
    main()
