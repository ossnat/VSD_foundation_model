"""
Resolve dataset paths relative to the project root.

All config paths should be written as ``Data/...`` (repo-local ``Data/`` directory).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

PathLike = Union[str, Path]

DATA_PREFIX = "Data"


def data_dir(project_root: Path) -> Path:
    """Return ``<project_root>/Data``."""
    return (project_root / DATA_PREFIX).resolve()


def resolve_repo_path(project_root: Path, path_value: PathLike) -> Path:
    """
    Resolve a config or CSV path under the repository.

    - Absolute paths are returned as-is.
    - Paths starting with ``Data/`` are resolved under ``project_root``.
    - Other relative paths are also resolved under ``project_root`` (legacy).
    """
    project_root = project_root.resolve()
    p = Path(path_value)
    if p.is_absolute():
        return p.resolve()
    return (project_root / p).resolve()


def resolve_config_data_paths(cfg: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    """Resolve split/stats/processed paths in a flat config dict."""
    for key in ("split_csv_path", "stats_json_path", "processed_root"):
        value = cfg.get(key)
        if value:
            cfg[key] = str(resolve_repo_path(project_root, value))
    return cfg


def resolve_target_h5_path(project_root: Path, target_file: PathLike) -> Path:
    """
    Resolve an H5 path from a split CSV ``target_file`` cell.

    Handles:
      - absolute paths that exist
      - ``Data/FoundationData/...`` relative to repo root
      - Colab-style ``/content/.../Data/...`` by re-rooting under ``project_root/Data/...``
    """
    project_root = project_root.resolve()
    p = Path(target_file)

    if p.is_absolute() and p.exists():
        return p.resolve()

    # Repo-relative Data/...
    candidate = resolve_repo_path(project_root, p)
    if candidate.exists():
        return candidate

    # Strip machine-specific prefix up to and including "Data"
    parts = p.parts
    if DATA_PREFIX in parts:
        idx = parts.index(DATA_PREFIX)
        rel = Path(*parts[idx:])
        candidate = project_root / rel
        if candidate.exists():
            return candidate.resolve()

    return resolve_repo_path(project_root, p)
