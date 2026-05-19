#!/usr/bin/env python3
"""
Report H5 file hierarchy and trial counts.

The report includes:
  - file/session name
  - full hierarchy tree (groups + datasets with shape/dtype)
  - summary counts (datasets, groups, trial datasets)
  - condition token inferred from filename (e.g. "condsAN")
  - number of trials for that condition (trial_* datasets in this file)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import h5py


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create an H5 hierarchy + trial-count report.")
    p.add_argument("--h5-path", type=str, required=True, help="Input .h5 file path.")
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for report outputs. Defaults to the H5 parent directory.",
    )
    p.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Optional output prefix. Defaults to h5 stem.",
    )
    return p.parse_args()


def _infer_condition_token(filename: str) -> str:
    stem = Path(filename).stem
    m = re.search(r"(conds?[A-Za-z0-9_]+)", stem, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return "unknown_condition"


def _collect_hierarchy(h5_path: Path) -> Tuple[List[str], Dict[str, int]]:
    lines: List[str] = []
    counts = {"groups": 0, "datasets": 0, "trial_datasets": 0}

    with h5py.File(h5_path, "r") as f:
        lines.append("/")

        def visitor(name: str, obj) -> None:
            depth = name.count("/")
            indent = "  " * (depth + 1)
            short_name = name.split("/")[-1]
            if isinstance(obj, h5py.Group):
                counts["groups"] += 1
                lines.append(f"{indent}{short_name}/")
            elif isinstance(obj, h5py.Dataset):
                counts["datasets"] += 1
                if re.match(r"^trial_\d+$", short_name):
                    counts["trial_datasets"] += 1
                lines.append(
                    f"{indent}{short_name}  [Dataset shape={tuple(obj.shape)}, dtype={obj.dtype}]"
                )

        f.visititems(visitor)

    return lines, counts


def main() -> None:
    args = _parse_args()
    h5_path = Path(args.h5_path).resolve()
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    out_dir = Path(args.output_dir).resolve() if args.output_dir else h5_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.output_prefix or h5_path.stem
    txt_path = out_dir / f"{prefix}_h5_structure_report.txt"
    json_path = out_dir / f"{prefix}_h5_structure_report.json"

    condition_token = _infer_condition_token(h5_path.name)
    tree_lines, counts = _collect_hierarchy(h5_path)

    summary = {
        "h5_path": str(h5_path),
        "session_name": h5_path.stem,
        "condition_token_from_filename": condition_token,
        "n_groups": counts["groups"],
        "n_datasets": counts["datasets"],
        "n_trials_for_condition": counts["trial_datasets"],
    }

    text_lines = [
        "H5 Structure Report",
        "===================",
        f"File: {h5_path}",
        f"Session: {h5_path.stem}",
        f"Condition token (from filename): {condition_token}",
        "",
        "Summary",
        "-------",
        f"Total groups: {counts['groups']}",
        f"Total datasets: {counts['datasets']}",
        f"Trials for this condition (trial_* datasets): {counts['trial_datasets']}",
        "",
        "Hierarchy",
        "---------",
    ]
    text_lines.extend(tree_lines)
    text_lines.append("")

    txt_path.write_text("\n".join(text_lines), encoding="utf-8")
    json_path.write_text(json.dumps({"summary": summary, "hierarchy": tree_lines}, indent=2), encoding="utf-8")

    print(f"Saved text report: {txt_path}")
    print(f"Saved JSON report: {json_path}")


if __name__ == "__main__":
    main()
