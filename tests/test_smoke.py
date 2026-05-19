"""Minimal smoke test: build mae_2d and run one forward pass (no data files)."""

import torch

from src.models import build_ssl_model


def test_build_and_forward():
    cfg = {
        "model": "mae_2d",
        "backbone": "resnet18",
        "pretrained": False,
        "channels": 1,
        "hidden_dim": 64,
        "normalize_loss": True,
        "loss_type": "mse",
    }
    model = build_ssl_model(cfg)
    model.eval()
    b, c, t, h, w = 1, 1, 1, 32, 32
    video = torch.randn(b, c, t, h, w)
    mask = torch.ones(b, 1, t, h, w)
    mask[..., 8:24, 8:24] = 0.0
    batch = {
        "video_masked": video * mask,
        "video_target": video,
        "mask": mask,
    }
    out = model(batch)
    assert "loss" in out
    assert torch.isfinite(out["loss"])
