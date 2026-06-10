"""LR scheduler factory."""

import torch
from torch.optim import AdamW

from src.training.lr_schedule import build_lr_scheduler


def test_no_scheduler():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = AdamW([p], lr=1e-3)
    assert build_lr_scheduler(opt, {"scheduler_type": "none"}) is None


def test_cosine_scheduler():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = AdamW([p], lr=1e-3)
    sched = build_lr_scheduler(
        opt,
        {"scheduler_type": "cosine", "epochs": 10, "warmup_epochs": 2, "min_lr": 1e-6},
    )
    assert sched is not None
    for _ in range(10):
        sched.step()
    assert opt.param_groups[0]["lr"] <= 1e-3
