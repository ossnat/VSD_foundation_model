#!/usr/bin/env python3
"""Rebuild all_trials_index.csv from H5 trial_metadata_json (fixes trial_dataset mapping)."""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data_pp_and_split.src.export import save_split_csv
from data_pp_and_split.src.index_build import (
    discover_h5_in_directory,
    discover_h5_target_files,
    rebuild_index_dataframe,
    rebuild_index_from_h5_paths,
)
from data_pp_and_split.src.paths import splits_dir, workspace_root


def _resolve_out(path: str | None) -> Path:
    if path is None:
        return splits_dir() / "all_trials_index.csv"
    p = Path(path)
    if str(p).startswith("Data/"):
        return workspace_root() / p
    return p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-index",
        default=None,
        help="Existing index listing target_file paths (default: splits/all_trials_index.csv)",
    )
    parser.add_argument(
        "--h5-dir",
        default=None,
        help="Rebuild from all *.h5 in this directory (e.g. ProcessedData/gandalf)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV (default: overwrite splits/all_trials_index.csv)",
    )
    parser.add_argument(
        "--allow-missing-h5",
        action="store_true",
        help="Skip sessions whose H5 is missing (partial rebuild for local dev)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not backup the existing index before overwrite",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate rebuild only; do not write output",
    )
    args = parser.parse_args()

    out_path = _resolve_out(args.output)

    if args.h5_dir:
        h5_dir = Path(args.h5_dir)
        if str(h5_dir).startswith("Data/"):
            h5_dir = workspace_root() / h5_dir
        elif not h5_dir.is_absolute():
            h5_dir = workspace_root() / "Data/FoundationData/ProcessedData" / h5_dir
        h5_paths = discover_h5_in_directory(h5_dir)
        print(f"H5 directory: {h5_dir}")
        print(f"Sessions to rebuild: {len(h5_paths)}")
        print(f"Output: {out_path}")
        df = rebuild_index_from_h5_paths(h5_paths)
    else:
        source_index = Path(args.source_index) if args.source_index else splits_dir() / "all_trials_index.csv"
        if str(source_index).startswith("Data/"):
            source_index = workspace_root() / source_index
        target_files = discover_h5_target_files(source_index)
        print(f"Source index: {source_index}")
        print(f"Sessions to rebuild: {len(target_files)}")
        print(f"Output: {out_path}")
        df = rebuild_index_dataframe(target_files, allow_missing_h5=args.allow_missing_h5)

    if args.dry_run:
        print(f"Dry run OK — would write {len(df)} rows to {out_path}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.no_backup:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = out_path.with_name(f"{out_path.stem}_pre_metadata_fix_{ts}{out_path.suffix}")
        shutil.copy2(out_path, backup)
        print(f"Backed up existing index -> {backup}")

    save_split_csv(df, out_path)
    print(f"Wrote {out_path} ({len(df)} trials)")


if __name__ == "__main__":
    main()
