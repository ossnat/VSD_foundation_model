#!/usr/bin/env python3
"""
Quick environment smoke test for VSD_foundation_model.

Checks third-party imports, `src` package, and a tiny CPU forward pass (no data files).

Usage (from repo root):
  python scripts/check_env.py
  python scripts/check_env.py --device cuda   # also test GPU if available
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _ok(msg: str) -> None:
    print(f"  OK  {msg}")


def _fail(msg: str) -> None:
    print(f"  FAIL  {msg}", file=sys.stderr)
    raise SystemExit(1)


def _check_imports() -> None:
    print("[1/4] Third-party packages")
    packages = [
        "torch",
        "torchvision",
        "tensorboard",
        "numpy",
        "h5py",
        "pandas",
        "tqdm",
        "matplotlib",
        "einops",
        "yaml",
        "scipy",
    ]
    for name in packages:
        mod = importlib.import_module("yaml" if name == "yaml" else name)
        ver = getattr(mod, "__version__", "?")
        _ok(f"{name} ({ver})")


def _check_torch(device: str) -> "object":
    import torch

    print("[2/4] PyTorch")
    _ok(f"torch {torch.__version__}")
    cuda = torch.cuda.is_available()
    if cuda:
        _ok(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("  note  CUDA not available (CPU-only run is fine)")

    if device == "cuda" and not cuda:
        _fail("--device cuda requested but torch.cuda.is_available() is False")
    return torch.device(device if device == "cuda" and cuda else "cpu")


def _check_src() -> None:
    print("[3/4] Project package (src)")
    from src.models import build_ssl_model  # noqa: F401
    from src.training.trainer import Trainer  # noqa: F401
    from src.experiments.mae_2d_lstm.load_config import load_and_prepare_config  # noqa: F401

    _ok("import src.models.build_ssl_model, Trainer, load_config")


def _check_forward(dev: "object") -> None:
    import torch
    from src.models import build_ssl_model

    print(f"[4/4] Tiny forward pass on {dev}")
    cfg = {
        "model": "mae_2d",
        "backbone": "resnet18",
        "pretrained": False,
        "channels": 1,
        "hidden_dim": 64,
        "normalize_loss": True,
        "loss_type": "l1",
    }
    model = build_ssl_model(cfg).to(dev)
    model.eval()

    # Small spatial size to stay light on CPU / low-memory SLURM jobs
    b, c, t, h, w = 1, 1, 1, 32, 32
    video = torch.randn(b, c, t, h, w, device=dev)
    mask = torch.ones(b, 1, t, h, w, device=dev)
    mask[..., 8:24, 8:24] = 0.0
    batch = {
        "video_masked": video * mask,
        "video_target": video,
        "mask": mask,
    }
    with torch.no_grad():
        out = model(batch)
    if "loss" not in out or not torch.isfinite(out["loss"]):
        _fail(f"forward pass bad output: {out}")
    _ok(f"mae_2d loss={float(out['loss']):.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description="VSD foundation model environment check")
    p.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device for forward pass (default: cpu)",
    )
    args = p.parse_args()

    print("VSD_foundation_model environment check")
    print(f"  repo root: {_ROOT}")
    print(f"  python:    {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info < (3, 9):
        _fail(f"Python >= 3.9 required, got {sys.version_info.major}.{sys.version_info.minor}")

    _check_imports()
    dev = _check_torch(args.device)
    _check_src()
    _check_forward(dev)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
