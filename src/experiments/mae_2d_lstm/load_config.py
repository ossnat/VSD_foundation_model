import os
from pathlib import Path
from typing import Dict, Any, Optional

import yaml


def load_and_prepare_config(
    base_cfg_path: str,
    project_root: Path,
    data_root: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load a base YAML config for MAE 2D+LSTM, apply simple overrides, and
    resolve data paths against a local Data directory.

    Args:
        base_cfg_path: Path to the base YAML config (relative to project_root or absolute).
        project_root: Root directory of the VSD_foundation_model project.
        data_root: Optional path to the Data directory. If None, assumes a sibling
                   'Data' directory next to project_root.
        overrides: Optional flat dict of config key -> new value.

    Returns:
        A config dict ready to be passed into data/model builders.
    """
    cfg_path = Path(base_cfg_path)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    # Apply simple flat overrides (e.g., monkeys, mask_ratio, clip_length, epochs)
    if overrides:
        for k, v in overrides.items():
            cfg[k] = v

    # Determine data_root: default to sibling Data/ if not provided
    if data_root is None:
        data_root = project_root.parent / "Data"

    # Resolve split_csv_path, stats_json_path, processed_root relative to data_root
    for key in ("split_csv_path", "stats_json_path", "processed_root"):
        value = cfg.get(key)
        if not value:
            continue
        value_path = Path(value)
        if not value_path.is_absolute():
            # Treat as relative to the shared Data directory
            resolved = (data_root / value_path.relative_to(value_path.parts[0])).resolve() if value_path.parts else data_root / value_path
            cfg[key] = str(resolved)

    # Ensure ckpt_dir and log_dir are at least present (relative to project root if needed)
    for key in ("ckpt_dir", "log_dir", "results_dir"):
        value = cfg.get(key)
        if value is None:
            continue
        value_path = Path(value)
        if not value_path.is_absolute():
            cfg[key] = str((project_root / value_path).resolve())

    return cfg

