"""
Shared evaluation plots and recon visualizations (model-agnostic when possible).
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _forward_reconstruction(model: torch.nn.Module, batch: Dict[str, Any], device: torch.device):
    """
    Run a reconstruction forward path suitable for plotting.

    Preferred path is model.reconstruct_for_vis(batch, device). Falls back to MAE-style
    encoder/decoder when batch contains MAE keys.
    """
    recon_fn = getattr(model, "reconstruct_for_vis", None)
    if callable(recon_fn):
        return recon_fn(batch, device)

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
            h, w = video_target.shape[2], video_target.shape[3]
            mask = F.interpolate(mask, size=(h, w), mode="nearest")

    features = model.encoder(video_masked)
    if is_2d or len(video_target.shape) == 4:
        target_size = (video_target.shape[2], video_target.shape[3])
        reconstruction = model.decoder(features, target_size=target_size)
    else:
        target_size = (video_target.shape[2], video_target.shape[3], video_target.shape[4])
        reconstruction = model.decoder(features, target_size=target_size)
        if len(mask.shape) == 5 and mask.shape[1] == 1:
            t, h, w = video_target.shape[2], video_target.shape[3], video_target.shape[4]
            mask = F.interpolate(mask, size=(t, h, w), mode="nearest")

    return reconstruction, video_target, video_masked, mask


def save_reconstruction_figure(
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
    Save reconstruction grid:
      - original | reconstructed | |diff|
      - original | masked | reconstructed | |diff| (if plot_masked=True)
    """
    if plt is None:
        print("[eval_plots] Matplotlib not available; skipping reconstruction figure.")
        return None

    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    if plot_masked:
        out_path = os.path.join(out_dir, f"test_reconstruction_orig_masked_recon_diff_{split_name}.png")
    else:
        out_path = os.path.join(out_dir, f"test_reconstruction_orig_recon_diff_{split_name}.png")

    rows_plotted: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = []
    with torch.no_grad():
        for bi, batch in enumerate(test_loader):
            if bi >= num_batches:
                break
            if not isinstance(batch, dict) or "video_masked" not in batch:
                continue
            recon, target, masked, _ = _forward_reconstruction(model, batch, device)
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

            if recon.dim() == 4:
                o = target[0, 0].cpu().numpy()
                r = recon[0, 0].cpu().numpy()
                m = masked[0, 0].cpu().numpy()
                d = np.abs(o - r)
                rows_plotted.append((o, r, d, m, start_frame_val))
            else:
                t_lim = min(recon.shape[2], max_frames_per_clip)
                for t in range(t_lim):
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
        flat = np.concatenate(
            [x[0].ravel() for x in rows_plotted]
            + [x[1].ravel() for x in rows_plotted]
            + [x[3].ravel() for x in rows_plotted]
        )
    else:
        flat = np.concatenate([x[0].ravel() for x in rows_plotted] + [x[1].ravel() for x in rows_plotted])
    vmin, vmax = np.percentile(flat, 5), np.percentile(flat, 95)

    for i, (o, r, d, m, frame_no) in enumerate(rows_plotted):
        axes[i, 0].imshow(o, cmap="hot", vmin=vmin, vmax=vmax)
        axes[i, 0].set_ylabel(f"frame={frame_no}")
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        if i == 0:
            axes[i, 0].set_title("Original")

        if plot_masked:
            axes[i, 1].imshow(m, cmap="hot", vmin=vmin, vmax=vmax)
            axes[i, 1].set_xticks([])
            axes[i, 1].set_yticks([])
            if i == 0:
                axes[i, 1].set_title("Masked input")

            axes[i, 2].imshow(r, cmap="hot", vmin=vmin, vmax=vmax)
            axes[i, 2].set_xticks([])
            axes[i, 2].set_yticks([])
            if i == 0:
                axes[i, 2].set_title("Reconstructed")

            dmax = max(float(d.max()), 1e-8)
            axes[i, 3].imshow(d, cmap="magma", vmin=0, vmax=dmax)
            axes[i, 3].set_xticks([])
            axes[i, 3].set_yticks([])
            if i == 0:
                axes[i, 3].set_title("|Original - Recon|")
        else:
            axes[i, 1].imshow(r, cmap="hot", vmin=vmin, vmax=vmax)
            axes[i, 1].set_xticks([])
            axes[i, 1].set_yticks([])
            if i == 0:
                axes[i, 1].set_title("Reconstructed")

            dmax = max(float(d.max()), 1e-8)
            axes[i, 2].imshow(d, cmap="magma", vmin=0, vmax=dmax)
            axes[i, 2].set_xticks([])
            axes[i, 2].set_yticks([])
            if i == 0:
                axes[i, 2].set_title("|Original - Recon|")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval_plots] Saved figure to {out_path}")
    return out_path


def save_temporal_per_metric_figures(
    result: Dict[int, Dict[str, Any]],
    out_dir: str,
    split_name: str,
) -> Optional[List[str]]:
    """
    Save one PNG per available temporal metric (MSE/RMSE/R2/SSIM/Pearson/PSNR).
    """
    if plt is None or not result:
        return None

    os.makedirs(out_dir, exist_ok=True)
    x = sorted(result.keys())

    def vec(key: str, default: float = float("nan")) -> List[float]:
        return [
            result[sf].get(key) if result[sf].get(key) is not None else default
            for sf in x
        ]

    specs: List[Tuple[str, str, List[float], Optional[List[float]], Optional[Tuple[float, float]]]] = []
    if any("mean_mse" in result[sf] for sf in x):
        specs.append(("mse", "MSE (masked)", vec("mean_mse"), vec("std_mse", 0.0), None))
        mean_psnr = [10.0 * math.log10(1.0 / max(float(m), 1e-10)) for m in vec("mean_mse")]
        specs.append(("psnr_masked_db", "PSNR (masked, dB)", mean_psnr, None, None))
    if any("mean_rmse" in result[sf] for sf in x):
        specs.append(("rmse", "RMSE (masked)", vec("mean_rmse"), None, None))
    if any("mean_r2_masked" in result[sf] for sf in x):
        specs.append(("r2", "R2 (masked)", vec("mean_r2_masked"), vec("std_r2_masked", 0.0), None))
    if any("mean_ssim_masked" in result[sf] for sf in x):
        specs.append(("ssim", "SSIM (masked)", vec("mean_ssim_masked"), vec("std_ssim_masked", 0.0), None))
    if any("mean_pearson_flat" in result[sf] for sf in x):
        specs.append(
            (
                "pearson_flat",
                "Pearson r (flattened frame)",
                vec("mean_pearson_flat"),
                vec("std_pearson_flat", 0.0),
                (-1.05, 1.05),
            )
        )

    saved: List[str] = []
    for key, ylabel, y, yerr, ylim in specs:
        fig, ax = plt.subplots(figsize=(8, 4))
        if yerr is not None:
            ax.errorbar(x, y, yerr=yerr, capsize=3, marker="o", linestyle="-")
        else:
            ax.plot(x, y, marker="o", linestyle="-")
        ax.set_xlabel("Clip start frame (time)")
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title(f"{ylabel} over time ({split_name})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, f"temporal_metrics_{split_name}_{key}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
        print(f"[eval_plots] Saved {path}")
    return saved


def _frame_tensor_to_2d(frame: torch.Tensor) -> np.ndarray:
    """(C,H,W) or (H,W) -> 2D numpy for imshow."""
    if frame.dim() == 3:
        return frame[0].detach().cpu().numpy()
    return frame.detach().cpu().numpy()


def save_true_vs_scrambled_frames(
    rows: List[Tuple[torch.Tensor, torch.Tensor, int]],
    out_path: str,
) -> Optional[str]:
    """
    Side-by-side panels: original target vs spatially scrambled target (same permutation).
    """
    if plt is None:
        print("[eval_plots] Matplotlib not available; skipping true-vs-scrambled figure.")
        return None
    if not rows:
        return None

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    arrays = [(_frame_tensor_to_2d(o), _frame_tensor_to_2d(s)) for o, s, _ in rows]
    flat = np.concatenate([a[0].ravel() for a in arrays] + [a[1].ravel() for a in arrays])
    vmin, vmax = np.percentile(flat, 5), np.percentile(flat, 95)

    nrows = len(arrays)
    fig, axes = plt.subplots(nrows, 2, figsize=(6, 2.2 * nrows))
    if nrows == 1:
        axes = np.array([axes])

    for i, ((orig, scr), (_, _, frame_no)) in enumerate(zip(arrays, rows)):
        axes[i, 0].imshow(orig, cmap="hot", vmin=vmin, vmax=vmax)
        axes[i, 0].set_ylabel(f"frame={frame_no}")
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        if i == 0:
            axes[i, 0].set_title("Original")

        axes[i, 1].imshow(scr, cmap="hot", vmin=vmin, vmax=vmax)
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
        if i == 0:
            axes[i, 1].set_title("Scrambled")

    fig.suptitle("True vs scrambled input frames", y=1.01, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval_plots] Saved {out_path}")
    return out_path


def save_reconstruction_from_batches(
    model: torch.nn.Module,
    batches: List[Dict[str, Any]],
    device: torch.device,
    out_path: str,
    plot_masked: bool = False,
    max_frames_per_clip: int = 8,
) -> Optional[str]:
    """Reconstruction grid (orig | masked | recon | diff) from explicit batch list."""
    if plt is None:
        print("[eval_plots] Matplotlib not available; skipping reconstruction figure.")
        return None
    if not batches:
        return None

    model.eval()
    rows_plotted: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = []
    with torch.no_grad():
        for batch in batches:
            if not isinstance(batch, dict) or "video_masked" not in batch:
                continue
            recon, target, masked, _ = _forward_reconstruction(model, batch, device)
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

            if recon.dim() == 4:
                o = target[0, 0].cpu().numpy()
                r = recon[0, 0].cpu().numpy()
                m = masked[0, 0].cpu().numpy()
                d = np.abs(o - r)
                rows_plotted.append((o, r, d, m, start_frame_val))
            else:
                t_lim = min(recon.shape[2], max_frames_per_clip)
                for t in range(t_lim):
                    o = target[0, 0, t].cpu().numpy()
                    rr = recon[0, 0, t].cpu().numpy()
                    m = masked[0, 0, t].cpu().numpy()
                    d = np.abs(o - rr)
                    rows_plotted.append((o, rr, d, m, start_frame_val + t))

    if not rows_plotted:
        return None

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    nrows = len(rows_plotted)
    ncols = 4 if plot_masked else 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 2.2 * nrows))
    if nrows == 1:
        axes = np.array([axes])

    if plot_masked:
        flat = np.concatenate(
            [x[0].ravel() for x in rows_plotted]
            + [x[1].ravel() for x in rows_plotted]
            + [x[3].ravel() for x in rows_plotted]
        )
    else:
        flat = np.concatenate([x[0].ravel() for x in rows_plotted] + [x[1].ravel() for x in rows_plotted])
    vmin, vmax = np.percentile(flat, 5), np.percentile(flat, 95)

    for i, (o, r, d, m, frame_no) in enumerate(rows_plotted):
        axes[i, 0].imshow(o, cmap="hot", vmin=vmin, vmax=vmax)
        axes[i, 0].set_ylabel(f"frame={frame_no}")
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        if i == 0:
            axes[i, 0].set_title("Original")

        if plot_masked:
            axes[i, 1].imshow(m, cmap="hot", vmin=vmin, vmax=vmax)
            axes[i, 1].set_xticks([])
            axes[i, 1].set_yticks([])
            if i == 0:
                axes[i, 1].set_title("Masked input")

            axes[i, 2].imshow(r, cmap="hot", vmin=vmin, vmax=vmax)
            axes[i, 2].set_xticks([])
            axes[i, 2].set_yticks([])
            if i == 0:
                axes[i, 2].set_title("Reconstructed")

            dmax = max(float(d.max()), 1e-8)
            axes[i, 3].imshow(d, cmap="magma", vmin=0, vmax=dmax)
            axes[i, 3].set_xticks([])
            axes[i, 3].set_yticks([])
            if i == 0:
                axes[i, 3].set_title("|Original - Recon|")
        else:
            axes[i, 1].imshow(r, cmap="hot", vmin=vmin, vmax=vmax)
            axes[i, 1].set_xticks([])
            axes[i, 1].set_yticks([])
            if i == 0:
                axes[i, 1].set_title("Reconstructed")

            dmax = max(float(d.max()), 1e-8)
            axes[i, 2].imshow(d, cmap="magma", vmin=0, vmax=dmax)
            axes[i, 2].set_xticks([])
            axes[i, 2].set_yticks([])
            if i == 0:
                axes[i, 2].set_title("|Original - Recon|")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval_plots] Saved figure to {out_path}")
    return out_path


def save_real_vs_scrambled_scatter(
    real_points: List[Tuple[float, float]],
    scrambled_points: List[Tuple[float, float]],
    out_path: str,
    x_label: str,
    y_label: str,
    title: str,
) -> Optional[str]:
    """
    Save a 2-color scatter plot comparing real vs scrambled-frame recon metrics.
    """
    if plt is None:
        print("[eval_plots] Matplotlib not available; skipping scatter plot.")
        return None
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    if real_points:
        xr = [p[0] for p in real_points]
        yr = [p[1] for p in real_points]
        ax.scatter(xr, yr, s=40, alpha=0.75, label="real", color="#1f77b4", edgecolors="none")
    if scrambled_points:
        xs = [p[0] for p in scrambled_points]
        ys = [p[1] for p in scrambled_points]
        ax.scatter(xs, ys, s=40, alpha=0.75, label="scrambled", color="#d62728", edgecolors="none")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval_plots] Saved {out_path}")
    return out_path

