import os
import sys
from pathlib import Path

# Add the project root to Python path to allow importing src modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import h5py
import numpy as np
import matplotlib

# Use non-interactive backend for tests
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from src.data.datasets import VsdVideoDataset
from src.data.data_loader import load_dataset


def _create_minimal_vsd_hdf5(hdf5_path: Path, *,
                              height: int = 100,
                              width: int = 100,
                              frames: int = 16,
                              trials: int = 3) -> None:
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(hdf5_path), "w") as f:
        grp = f.create_group("groupA")
        # dataset shape expected by code: (pixels, frames, trials)
        pixels = height * width
        data = np.random.randn(pixels, frames, trials).astype(np.float32)
        grp.create_dataset("dataset1", data=data)


def test_vsd_video_dataset_shapes(tmp_path: Path):
    h5_path = tmp_path / "vsd_test.hdf5"
    _create_minimal_vsd_hdf5(h5_path, frames=16, trials=5)

    ds = VsdVideoDataset(hdf5_path=str(h5_path))

    # Length equals number of trials added from data_structure
    assert len(ds) == 5

    sample = ds[0]
    assert isinstance(sample, dict)
    assert "video" in sample and "mask" in sample

    video: torch.Tensor = sample["video"]
    mask: torch.Tensor = sample["mask"]

    # Expected shapes per implementation
    assert video.ndim == 4  # (C, T, H, W)
    assert video.shape[0] == 1  # channels
    assert video.shape[1] == 16  # frames
    assert video.shape[2] == 100 and video.shape[3] == 100

    assert mask.shape == (1, 16, 100, 100)
    assert video.dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.uint8)


def test_load_dataset_and_visualize(tmp_path: Path):
    h5_path = tmp_path / "vsd_test.hdf5"
    _create_minimal_vsd_hdf5(h5_path, frames=24, trials=4)

    cfg = {
        "dataset": "vsd",
        "vsd_hdf5_path": str(h5_path),
    }

    loader = load_dataset(cfg, batch_size=2, num_workers=0, shuffle=False)
    batch = next(iter(loader))

    # Validate batch structure
    assert isinstance(batch, dict)
    assert "video" in batch and "mask" in batch

    video: torch.Tensor = batch["video"]  # (N, C, T, H, W) when collated
    mask: torch.Tensor = batch["mask"]

    # Shape checks
    assert video.shape[0] == 2  # batch size
    assert video.shape[1] == 1
    assert video.shape[2] == 24
    assert video.shape[3] == 100 and video.shape[4] == 100
    assert mask.shape == (2, 1, 24, 100, 100)

    # Minimal visualization to verify configuration
    # Take one sample and render a small grid of frames
    sample_video = video[0]
    if sample_video.shape[0] == 1:
        frames_to_plot = sample_video.squeeze(0)  # (T, H, W)
    else:
        frames_to_plot = sample_video.mean(dim=0)  # combine channels if present

    start_frame, end_frame = 0, min(9, frames_to_plot.shape[0] - 1)
    frames_sel = frames_to_plot[start_frame : end_frame + 1]

    cols = 5
    rows = int(np.ceil(frames_sel.shape[0] / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    axes = np.atleast_1d(axes).ravel()
    vmin, vmax = -0.003, 0.003

    for i in range(frames_sel.shape[0]):
        ax = axes[i]
        ax.imshow(frames_sel[i].cpu().numpy(), cmap="hot", vmin=vmin, vmax=vmax)
        ax.axis("off")
        ax.set_title(f"F{i}", fontsize=7)

    for j in range(frames_sel.shape[0], len(axes)):
        axes[j].axis("off")

    out_png = tmp_path / "vsd_sample_grid.png"
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    assert out_png.exists() and out_png.stat().st_size > 0


