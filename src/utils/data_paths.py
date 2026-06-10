"""
Resolve dataset paths for the standard workspace layout::

    <workspace>/
        Data/FoundationData/...
        VSD_foundation_model/    # project_root (this repo)

Config YAML paths should start with ``Data/...`` (sibling of the repo, not inside it).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

PathLike = Union[str, Path]

DATA_PREFIX = "Data"


def workspace_root(project_root: Path) -> Path:
    """Parent of the repo; contains ``Data/`` and ``VSD_foundation_model/``."""
    return project_root.resolve().parent


def data_dir(project_root: Path) -> Path:
    """Return ``<workspace>/Data``."""
    return (workspace_root(project_root) / DATA_PREFIX).resolve()


def _starts_with_data_prefix(path_value: PathLike) -> bool:
    s = str(path_value).replace("\\", "/")
    return s == DATA_PREFIX or s.startswith(f"{DATA_PREFIX}/")


def resolve_data_path(project_root: Path, path_value: PathLike) -> Path:
    """
    Resolve a path that refers to shared data under ``Data/``.

    - Absolute paths: returned as-is.
    - ``Data/...`` or ``Data``: under ``<workspace>/Data/...``.
    """
    p = Path(path_value)
    if p.is_absolute():
        return p.resolve()
    if _starts_with_data_prefix(path_value):
        return (workspace_root(project_root) / p).resolve()
    return (project_root / p).resolve()


def resolve_project_path(project_root: Path, path_value: PathLike) -> Path:
    """Resolve paths relative to the repo (checkpoints, logs, etc.)."""
    p = Path(path_value)
    if p.is_absolute():
        return p.resolve()
    return (project_root / p).resolve()


def resolve_config_data_paths(cfg: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    """Resolve split/stats/processed paths in a flat config dict."""
    for key in ("split_csv_path", "stats_json_path", "processed_root"):
        value = cfg.get(key)
        if value:
            cfg[key] = str(resolve_data_path(project_root, value))
    return cfg


def resolve_target_h5_path(project_root: Path, target_file: PathLike) -> Path:
    """
    Resolve an H5 path from a split CSV ``target_file`` cell.

    Handles:
      - absolute paths that exist on disk
      - ``Data/FoundationData/...`` under workspace sibling ``Data/``
      - machine-specific prefixes (e.g. Colab ``/content/.../Data/...``) by
        re-rooting under ``<workspace>/Data/...``
    """
    project_root = project_root.resolve()
    p = Path(target_file)

    if p.is_absolute() and p.exists():
        return p.resolve()

    candidate = resolve_data_path(project_root, p)
    if candidate.exists():
        return candidate.resolve()

    parts = p.parts
    if DATA_PREFIX in parts:
        idx = parts.index(DATA_PREFIX)
        rel = Path(*parts[idx:])
        candidate = workspace_root(project_root) / rel
        if candidate.exists():
            return candidate.resolve()

    return candidate.resolve()


# Back-compat alias used during earlier refactor
resolve_repo_path = resolve_data_path
