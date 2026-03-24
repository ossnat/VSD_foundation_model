"""
Save test-set reconstruction grids: original | reconstructed | |diff| (3 columns per row).
Uses raw video tensors (not z-scored) for interpretability.
"""
import os
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _forward_reconstruction(model: torch.nn.Module, batch: Dict[str, Any], device: torch.device):
    """Run MAE forward path; returns reconstruction, mask, target, masked input (on device)."""
    video_masked = batch["video_masked"].to(device)
    video_target = batch["video_target"].to(device)
    mask = batch["mask"].to(device)

    is_2d = False
    if len(video_target.shape) == 5 and video_target.shape[2] == 1:
        is_2d = True
        video_masked = video_masked.squeeze(2)
        video_target = video_target.squeeze(2)
        if len(mask.shape) == 5:
            mask = mask.squeeze(2)
        if len(mask.shape) == 4 and mask.shape[1] == 1:
            H, W = video_target.shape[2], video_target.shape[3]
            mask = F.interpolate(mask, size=(H, W), mode="nearest")

    features = model.encoder(video_masked)
    if is_2d or len(video_target.shape) == 4:
        target_size = (video_target.shape[2], video_target.shape[3])
        reconstruction = model.decoder(features, target_size=target_size)
    else:
        target_size = (video_target.shape[2], video_target.shape[3], video_target.shape[4])
        reconstruction = model.decoder(features, target_size=target_size)
        if len(mask.shape) == 5 and mask.shape[1] == 1:
            T, H, W = video_target.shape[2], video_target.shape[3], video_target.shape[4]
            mask = F.interpolate(mask, size=(T, H, W), mode="nearest")

    return reconstruction, video_target, video_masked, mask


def save_test_reconstruction_figure(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    out_dir: str,
    split_name: str = "test",
    num_batches: int = 1,
    max_frames_per_clip: int = 8,
    plot_masked: bool = False,
) -> Optional[str]:
    """
    Take the first `num_batches` from test_loader, first sample in each batch,
    plot rows = time frames (3D) or one row (2D).

    If `plot_masked` is False:
      cols = original | reconstructed | |diff|
    If `plot_masked` is True:
      cols = original | masked_input | reconstructed | |diff|

    Returns path to saved PNG or None if matplotlib unavailable.
    """
    if plt is None:
        print("[vis_test_reconstruction] Matplotlib not available; skipping figure.")
        return None

    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    if plot_masked:
        out_path = os.path.join(
            out_dir, f"test_reconstruction_orig_masked_recon_diff_{split_name}.png"
        )
    else:
        out_path = os.path.join(out_dir, f"test_reconstruction_orig_recon_diff_{split_name}.png")

    rows_plotted = []
    plot_is_2d: Optional[bool] = None
    with torch.no_grad():
        for bi, batch in enumerate(test_loader):
            if bi >= num_batches:
                break
            if not isinstance(batch, dict) or "video_masked" not in batch:
                continue
            recon, target, masked, _mask = _forward_reconstruction(model, batch, device)
            start_frames = batch.get("start_frame", None)
            if torch.is_tensor(start_frames):
                start_frame_val = int(start_frames[0].item())
            elif isinstance(start_frames, (list, tuple)) and len(start_frames) > 0:
                start_frame_val = int(start_frames[0])
            elif start_frames is not None:
                try:
                    start_frame_val = int(start_frames)
                except Exception:
                    start_frame_val = 0
            else:
                start_frame_val = 0

            # First sample in batch
            if recon.dim() == 4:
                # (B,C,H,W)
                plot_is_2d = True
                o = target[0, 0].cpu().numpy()
                r = recon[0, 0].cpu().numpy()
                m = masked[0, 0].cpu().numpy()
                d = np.abs(o - r)
                rows_plotted.append((o, r, d, m, start_frame_val))
            else:
                # (B,C,T,H,W)
                plot_is_2d = False
                T = min(recon.shape[2], max_frames_per_clip)
                for t in range(T):
                    o = target[0, 0, t].cpu().numpy()
                    rr = recon[0, 0, t].cpu().numpy()
                    m = masked[0, 0, t].cpu().numpy()
                    d = np.abs(o - rr)
                    rows_plotted.append((o, rr, d, m, start_frame_val + t))

    if not rows_plotted:
        return None

    nrows = len(rows_plotted)
    ncols = 4 if plot_masked else 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 2.2 * nrows))
    if nrows == 1:
        axes = np.array([axes])

    if plot_masked:
        flat = np.concatenate([x[0].ravel() for x in rows_plotted] + [x[1].ravel() for x in rows_plotted] + [x[3].ravel() for x in rows_plotted])
    else:
        flat = np.concatenate([x[0].ravel() for x in rows_plotted] + [x[1].ravel() for x in rows_plotted])
    vmin, vmax = np.percentile(flat, 5), np.percentile(flat, 95)

    for i, (o, r, d, m, frame_no) in enumerate(rows_plotted):
        # Original
        axes[i, 0].imshow(o, cmap="hot", vmin=vmin, vmax=vmax)
        axes[i, 0].set_ylabel(f"frame={frame_no}")
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        if i == 0:
            axes[i, 0].set_title("Original")

        if plot_masked:
            # Masked input
            axes[i, 1].imshow(m, cmap="hot", vmin=vmin, vmax=vmax)
            axes[i, 1].set_xticks([])
            axes[i, 1].set_yticks([])
            if i == 0:
                axes[i, 1].set_title("Masked input")

            # Reconstructed
            axes[i, 2].imshow(r, cmap="hot", vmin=vmin, vmax=vmax)
            axes[i, 2].set_xticks([])
            axes[i, 2].set_yticks([])
            if i == 0:
                axes[i, 2].set_title("Reconstructed")

            dmax = max(d.max(), 1e-8)
            axes[i, 3].imshow(d, cmap="magma", vmin=0, vmax=dmax)
            axes[i, 3].set_xticks([])
            axes[i, 3].set_yticks([])
            if i == 0:
                axes[i, 3].set_title("|Original − Recon|")
        else:
            # Reconstructed (column 1)
            axes[i, 1].imshow(r, cmap="hot", vmin=vmin, vmax=vmax)
            axes[i, 1].set_xticks([])
            axes[i, 1].set_yticks([])
            if i == 0:
                axes[i, 1].set_title("Reconstructed")

            dmax = max(d.max(), 1e-8)
            axes[i, 2].imshow(d, cmap="magma", vmin=0, vmax=dmax)
            axes[i, 2].set_xticks([])
            axes[i, 2].set_yticks([])
            if i == 0:
                axes[i, 2].set_title("|Original − Recon|")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[vis_test_reconstruction] Saved figure to {out_path}")
    return out_path
