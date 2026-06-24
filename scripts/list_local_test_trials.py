#!/usr/bin/env python3
"""List v3 test trials available for local H5 files (for paper-figure frame picking)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data_pp_and_split.src.paths import workspace_root, processed_root


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--split-csv",
        default=str(
            processed_root() / "splits/split_v3_seed17_session_condition_group.csv"
        ),
    )
    p.add_argument("--monkey", nargs="*", default=None, help="Filter monkeys (default: all local)")
    p.add_argument("--h5", nargs="*", default=None, help="Filter H5 basenames or relative paths")
    args = p.parse_args()

    split_path = Path(args.split_csv)
    if not split_path.is_absolute():
        split_path = workspace_root() / args.split_csv if str(args.split_csv).startswith("Data/") else split_path
    df = pd.read_csv(split_path)
    df["h5_basename"] = df["target_file"].apply(lambda x: Path(str(x)).name)

    local_h5 = sorted(
        p for p in processed_root().rglob("*.h5") if "splits" not in p.parts
    )
    local_basenames = {p.name for p in local_h5}
    print("=== LOCAL SESSION H5 FILES ===")
    for p in local_h5:
        rel = p.relative_to(workspace_root()) if p.is_relative_to(workspace_root()) else p
        print(f"  {rel}")

    sub = df[df["h5_basename"].isin(local_basenames)].copy()
    if args.monkey:
        monkeys = {m.lower() for m in args.monkey}
        sub = sub[sub["monkey"].astype(str).str.lower().isin(monkeys)]
    if args.h5:
        wanted = {Path(h).name for h in args.h5}
        sub = sub[sub["h5_basename"].isin(wanted)]

    print(f"\n=== SPLIT COUNTS (local H5 subset, n={len(sub)}) ===")
    print(sub.groupby("split").size().to_string())

    print("\n=== TEST SESSION × CONDITION (local H5 only) ===")
    test = sub[sub["split"] == "test"]
    if test.empty:
        print("  (no test trials in this subset)")
    else:
        g = (
            test.groupby(["monkey", "date", "condition", "h5_basename"], as_index=False)
            .agg(n_trials=("trial_dataset", "count"), trial_max=("trial_dataset", "max"))
            .sort_values(["monkey", "date", "condition"])
        )
        print(g.to_string(index=False))

    print("\n=== GANDALF 2-FILE SUMMARY (if present) ===")
    gandalf = ["session_240718c_condsAN.h5", "session_270618b_condsAN.h5"]
    for h5 in gandalf:
        if h5 not in local_basenames:
            continue
        block = sub[sub["h5_basename"] == h5]
        print(f"\n{h5}:")
        print(block.groupby(["condition", "split"]).size().unstack(fill_value=0).to_string())
        tblock = block[block["split"] == "test"]
        if not tblock.empty:
            print(f"  test trials: {tblock['trial_dataset'].min()} .. {tblock['trial_dataset'].max()} (n={len(tblock)})")

    print("\nTip: trials trial_000000..trial_000020 exist in BOTH Gandalf test sessions;")
    print("     use trial_000021+ on 270618b only, or set `h5:` in figure YAML to disambiguate.")


if __name__ == "__main__":
    main()
