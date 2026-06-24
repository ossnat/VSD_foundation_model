#!/usr/bin/env python3
"""Show trial_dataset mapping fix for one session (old buggy index vs rebuilt)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data_pp_and_split.src.index_io import condition_to_int
from data_pp_and_split.src.paths import splits_dir, workspace_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monkey", default="gandalf")
    parser.add_argument("--date", default="270618b")
    parser.add_argument(
        "--old-index",
        default=None,
        help="Backup index (default: latest all_trials_index_pre_metadata_fix_*.csv)",
    )
    args = parser.parse_args()

    splits = splits_dir()
    old_path = Path(args.old_index) if args.old_index else None
    if old_path is None:
        backups = sorted(splits.glob("all_trials_index_pre_metadata_fix_*.csv"))
        if not backups:
            raise SystemExit("No backup index found; pass --old-index")
        old_path = backups[-1]
    if str(old_path).startswith("Data/"):
        old_path = workspace_root() / old_path

    new_path = splits / "all_trials_index.csv"
    old = pd.read_csv(old_path)
    new = pd.read_csv(new_path)

    for name, cond in [("OLD (buggy)", "condAN2"), ("NEW (fixed)", "condAN2")]:
        df = old if name.startswith("OLD") else new
        sub = df[(df.monkey == args.monkey) & (df.date == args.date) & (df.condition == cond)]
        if sub.empty:
            print(f"{name} {cond}: no rows")
            continue
        trials = sub.sort_values("trial_index_in_condition")["trial_dataset"].tolist()
        print(f"{name} {cond}: n={len(trials)}  first={trials[:3]}  last={trials[-3:]}")

    # v3 split cond int
    v3 = pd.read_csv(splits / "split_v3_seed17_session_condition_group.csv")
    c2 = int(condition_to_int("condAN2"))
    test = v3[
        (v3.monkey == args.monkey)
        & (v3.date == args.date)
        & (v3.condition == c2)
        & (v3.split == "test")
    ]
    if len(test):
        t = test.sort_values("trial_index_in_condition")["trial_dataset"].tolist()
        print(f"v3 TEST cond{c2}: n={len(t)}  first={t[:3]}  last={t[-3:]}")
    else:
        print(f"v3 TEST cond{c2}: (no test rows — partial index or different group draw)")


if __name__ == "__main__":
    main()
