#!/usr/bin/env python3
"""Generate v3 session×condition group split and baseline stats."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Repo root on PYTHONPATH (run from VSD_foundation_model with PYTHONPATH=.)
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data_pp_and_split.src.baseline_stats import compute_per_pixel_baseline_stats
from data_pp_and_split.src.export import (
    artifact_basename,
    build_split_export_df,
    save_baseline_stats_h5,
    save_split_csv,
    save_stats_json,
)
from data_pp_and_split.src.group_split import (
    assign_splits,
    group_leak_check,
    physical_split_leak_check,
    split_summary,
    unique_groups,
)
from data_pp_and_split.src.index_io import load_index
from data_pp_and_split.src.paths import workspace_root


def _resolve_config_path(p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    if str(path).startswith("Data/"):
        return workspace_root() / path
    return _REPO / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(_REPO / "data_pp_and_split/configs/split_v3_default.yaml"),
    )
    parser.add_argument(
        "--allow-missing-h5",
        action="store_true",
        help="Skip train trials whose H5 files are missing (local partial runs)",
    )
    parser.add_argument("--skip-stats", action="store_true", help="Only write split CSV")
    parser.add_argument(
        "--index-path",
        default=None,
        help="Override index CSV (e.g. splits/all_trials_index_gandalf.csv)",
    )
    parser.add_argument(
        "--artifact-suffix",
        default=None,
        help="Append to output basename (e.g. gandalf -> split_v3_seed17_session_condition_group_gandalf.csv)",
    )
    args = parser.parse_args()

    cfg_path = _resolve_config_path(args.config)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    split_cfg = cfg["split"]
    version = split_cfg["version"]
    seed = int(split_cfg["seed"])
    stratify = split_cfg["stratify_level"]
    tag = artifact_basename(version, seed, stratify)
    if args.artifact_suffix:
        tag = f"{tag}_{args.artifact_suffix}"

    index_path = args.index_path or cfg.get("index_path")
    if index_path:
        index_path = _resolve_config_path(index_path)
    df = load_index(index_path)

    print(f"Loaded index: {len(df)} trials, {df['group_key'].nunique()} session×condition groups")
    print(f"Monkeys: {sorted(df['monkey'].unique())}")

    df = assign_splits(
        df,
        seed=seed,
        train_frac=float(split_cfg["train_frac"]),
        val_frac=float(split_cfg["val_frac"]),
        test_frac=float(split_cfg["test_frac"]),
        manual=split_cfg.get("manual_assignments"),
    )
    group_leak_check(df)

    print("\n=== SPLIT (trials) ===")
    print(df["split"].value_counts().to_string())
    print("\n=== SPLIT (groups) ===")
    g = unique_groups(df)
    print(g.merge(df[["group_key", "split"]].drop_duplicates(), on="group_key")["split"].value_counts().to_string())
    print("\n=== BY MONKEY ===")
    print(split_summary(df).to_string(index=False))

    out_splits = _resolve_config_path(cfg["output"]["splits_dir"])
    legacy_csv = _resolve_config_path(cfg["legacy_split_csv"]) if cfg.get("legacy_split_csv") else None

    export_df = build_split_export_df(df, legacy_csv)
    physical_split_leak_check(export_df)
    split_csv_path = out_splits / f"split_{tag}.csv"
    save_split_csv(export_df, split_csv_path)
    print(f"\n✓ Wrote {split_csv_path}")

    if args.skip_stats:
        return

    bl = cfg["baseline"]
    df_train = export_df[export_df["split"] == "train"].copy()

    stats = compute_per_pixel_baseline_stats(
        df_train,
        baseline_frames_start=int(bl["frames_start"]),
        baseline_frames_end=int(bl["frames_end"]),
        allow_missing_h5=args.allow_missing_h5,
    )

    stats_h5_path = out_splits / f"baseline_stats_{tag}.h5"
    save_baseline_stats_h5(stats, stats_h5_path)
    print(f"✓ Wrote {stats_h5_path}")

    meta = {
        "split_name": version,
        "random_seed": seed,
        "stratify_level": stratify,
        "train_frac": float(split_cfg["train_frac"]),
        "val_frac": float(split_cfg["val_frac"]),
        "test_frac": float(split_cfg["test_frac"]),
        "baseline_frames_start": int(bl["frames_start"]),
        "baseline_frames_end": int(bl["frames_end"]),
        "n_train_trials": int(len(df_train)),
    }
    stats_json_path = out_splits / f"baseline_stats_{tag}.json"
    save_stats_json(
        out_path=stats_json_path,
        meta=meta,
        stats=stats,
        split_csv_filename=split_csv_path.name,
        stats_h5_filename=stats_h5_path.name,
    )
    print(f"✓ Wrote {stats_json_path}")


if __name__ == "__main__":
    main()
