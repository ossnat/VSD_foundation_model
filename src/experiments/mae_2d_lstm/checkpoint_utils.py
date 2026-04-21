from __future__ import annotations

from pathlib import Path
from typing import Optional


def resolve_checkpoint_file(ckpt_dir: Path, explicit_path: Optional[str] = None) -> Path:
    """
    Resolve checkpoint path with deterministic priority:
      1) explicit_path (if provided)
      2) model_final.pt
      3) latest epoch_*.pt
      4) encoder_final.pt
    """
    if explicit_path:
        p = Path(explicit_path)
        if not p.exists():
            raise FileNotFoundError(f"--checkpoint-path not found: {p}")
        return p

    model_final = ckpt_dir / "model_final.pt"
    if model_final.exists():
        return model_final

    epoch_files = list(ckpt_dir.glob("epoch_*.pt"))
    if epoch_files:
        def _epoch_num(path: Path) -> int:
            try:
                return int(path.stem.split("_")[-1])
            except Exception:
                return -1

        return sorted(epoch_files, key=_epoch_num)[-1]

    encoder_final = ckpt_dir / "encoder_final.pt"
    if encoder_final.exists():
        return encoder_final

    raise FileNotFoundError(
        f"No checkpoint in {ckpt_dir}. Expected model_final.pt, epoch_*.pt, or encoder_final.pt."
    )
