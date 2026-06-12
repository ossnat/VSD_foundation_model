"""Path helpers for data_pp_and_split (workspace-relative Data/ layout)."""

from __future__ import annotations

import re
from pathlib import Path

DATA_PREFIX = "Data"


def project_root() -> Path:
    """VSD_foundation_model repo root (parent of data_pp_and_split/)."""
    return Path(__file__).resolve().parents[2]


def workspace_root() -> Path:
    """Parent of repo; contains ``Data/`` and ``VSD_foundation_model/``."""
    return project_root().parent


def data_dir() -> Path:
    return (workspace_root() / DATA_PREFIX).resolve()


def processed_root() -> Path:
    return data_dir() / "FoundationData" / "ProcessedData"


def splits_dir() -> Path:
    return processed_root() / "splits"


def resolve_h5_path(target_file: str | Path) -> Path:
    """
    Resolve an H5 path from a split/index ``target_file`` cell.

    Prefers portable ``Data/FoundationData/...`` under the workspace sibling.
    """
    p = Path(target_file)
    if p.is_absolute() and p.exists():
        return p.resolve()

    if str(p).replace("\\", "/").startswith(f"{DATA_PREFIX}/"):
        candidate = workspace_root() / p
        if candidate.exists():
            return candidate.resolve()

    parts = p.parts
    if DATA_PREFIX in parts:
        idx = parts.index(DATA_PREFIX)
        rel = Path(*parts[idx:])
        candidate = workspace_root() / rel
        if candidate.exists():
            return candidate.resolve()

    candidate = processed_root().parent.parent / p
    if candidate.exists():
        return candidate.resolve()

    return (workspace_root() / p).resolve()


_COLAB_DATA_RE = re.compile(r"/Data/(FoundationData/.*)$")


def normalize_target_file(target_file: str | Path) -> str:
    """Return portable ``Data/FoundationData/ProcessedData/...`` path."""
    s = str(target_file).replace("\\", "/")
    if s.startswith(f"{DATA_PREFIX}/"):
        return s

    m = _COLAB_DATA_RE.search(s)
    if m:
        return f"{DATA_PREFIX}/{m.group(1)}"

    p = Path(target_file)
    if p.is_absolute():
        ws = workspace_root()
        try:
            rel = p.resolve().relative_to(ws)
            return str(rel).replace("\\", "/")
        except ValueError:
            pass

    if "FoundationData/ProcessedData/" in s:
        idx = s.index("FoundationData/ProcessedData/")
        return f"{DATA_PREFIX}/{s[idx:]}"

    return s
