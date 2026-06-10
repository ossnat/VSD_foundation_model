"""Learning-rate schedulers for Trainer (optional; disabled when scheduler_type is none)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def build_lr_scheduler(
    optimizer: Optimizer,
    cfg: Dict[str, Any],
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """
    Build a PyTorch LR scheduler from flat training config keys.

    Supported ``scheduler_type``:
      - ``none`` / ``null`` / ``disabled`` / empty -> no scheduler (fixed LR)
      - ``cosine`` -> cosine decay to ``min_lr`` (optional linear warmup)
    """
    sched_type = str(cfg.get("scheduler_type", "none") or "none").strip().lower()
    if sched_type in ("none", "null", "off", "false", "disabled", ""):
        return None

    total_epochs = max(1, int(cfg.get("epochs", 10)))
    warmup_epochs = max(0, int(cfg.get("warmup_epochs", 0)))
    min_lr = float(cfg.get("min_lr", 1e-6))

    if sched_type == "cosine":
        if warmup_epochs > 0 and warmup_epochs < total_epochs:
            warmup = LinearLR(
                optimizer,
                start_factor=float(cfg.get("warmup_start_factor", 0.01)),
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=total_epochs - warmup_epochs,
                eta_min=min_lr,
            )
            return SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=min_lr)

    raise ValueError(
        f"Unknown scheduler_type={sched_type!r}. Supported: none, cosine."
    )
