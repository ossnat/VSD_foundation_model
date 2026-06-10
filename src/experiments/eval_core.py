"""
Shared evaluation helpers (model-agnostic orchestration utilities).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.experiments.mae_2d_lstm.checkpoint_utils import (
    load_checkpoint_into_model,
    resolve_checkpoint_file,
)


def build_cfg_overrides_from_args(
    args: Any,
    keys: Sequence[str],
) -> Dict[str, Any]:
    """
    Build flat cfg overrides from argparse namespace fields.

    For list-like fields (e.g. patch_size/monkeys) this keeps native values and lets
    callers post-process if needed.
    """
    out: Dict[str, Any] = {}
    for key in keys:
        if not hasattr(args, key):
            continue
        value = getattr(args, key)
        if value is not None:
            out[key] = value
    return out


def enforce_mae_2d_clip_contract(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep MAE 2D inputs valid: clip_length=1 and patch_size temporal dim=1.
    """
    if cfg.get("model") != "mae_2d":
        return cfg
    cfg["clip_length"] = 1
    ps = cfg.get("patch_size")
    if isinstance(ps, list) and len(ps) == 3:
        cfg["patch_size"] = [1, int(ps[1]), int(ps[2])]
    return cfg


def resolve_and_load_checkpoint(
    model: Any,
    ckpt_dir: Path,
    checkpoint_path: Optional[str],
    device: Any,
) -> Tuple[Path, str]:
    """
    Resolve checkpoint file and load it into model.

    Returns (resolved_ckpt_file, load_mode), where load_mode is either
    "full_model_state_dict" or "encoder_only_state_dict".
    """
    ckpt_file = resolve_checkpoint_file(ckpt_dir, checkpoint_path)
    load_mode = "full_model_state_dict"
    try:
        load_checkpoint_into_model(model, ckpt_file, device)
    except Exception:
        if hasattr(model, "encoder"):
            load_checkpoint_into_model(model, ckpt_file, device, encoder_only=True)
            load_mode = "encoder_only_state_dict"
        else:
            raise
    return ckpt_file, load_mode


def select_indices_by_target_files(
    dataset: Any,
    target_files: Sequence[str],
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Select dataset indices whose underlying ``target_file`` matches requested files.

    Matching supports absolute paths and basenames.
    """
    wanted_abs = set()
    wanted_base = set()
    for tf in target_files:
        p = Path(tf)
        wanted_base.add(p.name)
        if p.is_absolute():
            wanted_abs.add(str(p.resolve()))

    if not hasattr(dataset, "trials") or not hasattr(dataset, "data_structure"):
        raise ValueError(
            "Dataset does not expose trials/data_structure needed for target-file filtering."
        )

    selected: List[int] = []
    matched_file_names = set()
    for i, (row_idx, _clip_start) in enumerate(dataset.data_structure):
        tf = str(dataset.trials.iloc[row_idx]["target_file"])
        tf_abs = str(Path(tf).resolve())
        tf_base = Path(tf).name
        if tf_abs in wanted_abs or tf_base in wanted_base:
            selected.append(i)
            matched_file_names.add(tf_base)

    meta = {
        "mode": "target_files",
        "requested_files": list(target_files),
        "matched_indices": len(selected),
        "matched_files": len(matched_file_names),
    }
    return selected, meta


def choose_eval_indices(
    dataset: Any,
    *,
    target_files: Optional[Sequence[str]] = None,
    subset_size: int = 200,
    seed: int = 17,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Shared subset selection for eval scripts:
      - target-files mode (if provided)
      - random subset mode otherwise
    """
    total = len(dataset)
    if total == 0:
        return [], {"mode": "empty", "matched_files": 0}

    if target_files:
        return select_indices_by_target_files(dataset, target_files)

    rng = random.Random(seed)
    k = min(max(int(subset_size), 1), total)
    selected = list(range(total))
    rng.shuffle(selected)
    selected = selected[:k]
    return selected, {"mode": "random_subset", "subset_size": k, "total_samples": total}
