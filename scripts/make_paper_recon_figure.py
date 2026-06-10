#!/usr/bin/env python3
"""
Build a YAML-driven publication figure with per-frame assets and composite layout.

Example:
  PYTHONPATH=. python scripts/make_paper_recon_figure.py \\
    --figure-config configs/paper_figures/gandalf_trial154.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.experiments.paper_figure import load_figure_config, run_paper_figure_job


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compose publication reconstruction figure from YAML.")
    p.add_argument("--figure-config", type=str, required=True, help="YAML/JSON figure specification.")
    p.add_argument("--out-dir", type=str, default=None, help="Optional override for output directory.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg_path = Path(args.figure_config)
    if not cfg_path.is_absolute():
        cfg_path = (_PROJECT_ROOT / cfg_path).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Figure config not found: {cfg_path}")

    figure_cfg = load_figure_config(cfg_path)
    if args.out_dir:
        figure_cfg["out_dir"] = args.out_dir

    summary = run_paper_figure_job(
        figure_cfg,
        project_root=_PROJECT_ROOT,
        figure_config_path=cfg_path,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    from src.utils.colormap_register import register_custom_cmaps

    register_custom_cmaps()
    main()
