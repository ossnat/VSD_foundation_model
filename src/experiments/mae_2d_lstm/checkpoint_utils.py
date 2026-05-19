from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def _epoch_num(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except Exception:
        return -1


def training_state_epoch(ckpt_dir: Path) -> Optional[int]:
    """Return next epoch index from training_state files, or None."""
    pt_path = ckpt_dir / "training_state.pt"
    json_path = ckpt_dir / "training_state.json"
    ts: Dict[str, Any] = {}
    if pt_path.is_file():
        try:
            ts = torch.load(pt_path, map_location="cpu")
        except Exception:
            ts = {}
    elif json_path.is_file():
        try:
            with open(json_path, "r") as f:
                ts = json.load(f)
        except Exception:
            ts = {}
    if ts and "epoch" in ts:
        return int(ts["epoch"])
    return None


def has_training_checkpoint(ckpt_dir: Path) -> bool:
    """True if a prior training run left resumable artifacts."""
    ckpt_dir = Path(ckpt_dir)
    if training_state_epoch(ckpt_dir) is not None:
        return True
    if list(ckpt_dir.glob("epoch_*.pt")):
        return True
    return (ckpt_dir / "model_final.pt").is_file()


def resolve_resume_checkpoint(ckpt_dir: Path, explicit_path: Optional[str] = None) -> Path:
    """
    Checkpoint for **training resume** (not eval-only).

    Priority:
      1) explicit_path
      2) epoch_{N}.pt matching training_state epoch
      3) latest epoch_*.pt
      4) model_final.pt
    """
    ckpt_dir = Path(ckpt_dir)
    if explicit_path:
        p = Path(explicit_path)
        if not p.exists():
            raise FileNotFoundError(f"resume_checkpoint_path not found: {p}")
        return p

    epoch = training_state_epoch(ckpt_dir)
    if epoch is not None:
        epoch_ckpt = ckpt_dir / f"epoch_{epoch}.pt"
        if epoch_ckpt.exists():
            return epoch_ckpt

    epoch_files = list(ckpt_dir.glob("epoch_*.pt"))
    if epoch_files:
        return sorted(epoch_files, key=_epoch_num)[-1]

    model_final = ckpt_dir / "model_final.pt"
    if model_final.exists():
        return model_final

    raise FileNotFoundError(
        f"No resumable checkpoint in {ckpt_dir}. "
        "Expected training_state.* and epoch_*.pt or model_final.pt."
    )


def resolve_checkpoint_file(ckpt_dir: Path, explicit_path: Optional[str] = None) -> Path:
    """
    Checkpoint for **evaluation** (prefer finished model).

    Priority:
      1) explicit_path
      2) model_final.pt
      3) latest epoch_*.pt
      4) encoder_final.pt
    """
    if explicit_path:
        p = Path(explicit_path)
        if not p.exists():
            raise FileNotFoundError(f"--checkpoint-path not found: {p}")
        return p

    ckpt_dir = Path(ckpt_dir)
    model_final = ckpt_dir / "model_final.pt"
    if model_final.exists():
        return model_final

    epoch_files = list(ckpt_dir.glob("epoch_*.pt"))
    if epoch_files:
        return sorted(epoch_files, key=_epoch_num)[-1]

    encoder_final = ckpt_dir / "encoder_final.pt"
    if encoder_final.exists():
        return encoder_final

    raise FileNotFoundError(
        f"No checkpoint in {ckpt_dir}. Expected model_final.pt, epoch_*.pt, or encoder_final.pt."
    )


def load_checkpoint_into_model(
    model,
    ckpt_path: Path,
    device,
    *,
    encoder_only: bool = False,
) -> None:
    """Load weights from epoch/model checkpoint (supports dict or raw state_dict)."""
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        weights = state["model"]
    else:
        weights = state
    if encoder_only and hasattr(model, "encoder"):
        model.encoder.load_state_dict(weights, strict=True)
    else:
        model.load_state_dict(weights, strict=True)
