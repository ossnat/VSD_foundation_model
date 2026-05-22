from pathlib import Path
from typing import Dict, Any, Optional

import yaml

from src.utils.data_paths import DATA_PREFIX, resolve_config_data_paths


def load_and_prepare_config(
    base_cfg_path: str,
    project_root: Path,
    data_root: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load a base YAML config for MAE experiments, apply overrides, and resolve paths.

    Data paths in YAML should start with ``Data/...`` and resolve under
    ``<workspace>/Data/...`` where workspace is the parent of ``project_root``.

    Args:
        base_cfg_path: Path to the base YAML config (relative to project_root or absolute).
        project_root: Root directory of the VSD_foundation_model project.
        data_root: Optional override for ``<workspace>/Data`` (default: ``project_root.parent / Data``).
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

    if overrides:
        for k, v in overrides.items():
            cfg[k] = v

    resolve_config_data_paths(cfg, project_root)
    if data_root is not None:
        dr = Path(data_root).resolve()
        for key in ("split_csv_path", "stats_json_path", "processed_root"):
            value = cfg.get(key)
            if not value:
                continue
            p = Path(value)
            if DATA_PREFIX in p.parts:
                idx = p.parts.index(DATA_PREFIX)
                rel = Path(*p.parts[idx + 1 :])
                cfg[key] = str((dr / rel).resolve())

    for key in ("ckpt_dir", "log_dir", "results_dir"):
        value = cfg.get(key)
        if value is None:
            continue
        value_path = Path(value)
        if not value_path.is_absolute():
            cfg[key] = str((project_root / value_path).resolve())

    return cfg
