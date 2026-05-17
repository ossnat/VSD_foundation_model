"""
Extra temporal-eval figures: one PNG per metric (in addition to Trainer's 4-panel plot).
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def save_temporal_per_metric_figures(
    result: Dict[int, Dict[str, Any]],
    out_dir: str,
    split_name: str,
) -> Optional[list]:
    """
    Save separate line plots for masked MSE, RMSE, R², SSIM vs clip start_frame.

    Args:
        result: Trainer.evaluate_metrics_over_time return value (dict keyed by start_frame).
        out_dir: Directory for PNG files.
        split_name: e.g. "val" or "test".

    Returns:
        List of saved paths, or None if matplotlib unavailable / empty result.
    """
    if plt is None or not result:
        return None
    os.makedirs(out_dir, exist_ok=True)
    x = sorted(result.keys())
    mean_mse = [result[sf]["mean_mse"] for sf in x]
    std_mse = [result[sf]["std_mse"] for sf in x]
    mean_rmse = [result[sf]["mean_rmse"] for sf in x]
    mean_r2 = [result[sf]["mean_r2_masked"] for sf in x]
    std_r2 = [result[sf]["std_r2_masked"] for sf in x]
    mean_ssim = [result[sf]["mean_ssim_masked"] for sf in x]
    std_ssim = [result[sf]["std_ssim_masked"] for sf in x]
    mean_pearson = [
        result[sf].get("mean_pearson_flat")
        if result[sf].get("mean_pearson_flat") is not None
        else float("nan")
        for sf in x
    ]
    std_pearson = [
        result[sf].get("std_pearson_flat")
        if result[sf].get("std_pearson_flat") is not None
        else 0.0
        for sf in x
    ]

    specs = [
        ("mse", "MSE (masked)", mean_mse, std_mse),
        ("rmse", "RMSE (masked)", mean_rmse, None),
        ("r2", "R² (masked)", mean_r2, std_r2),
        ("ssim", "SSIM (masked)", mean_ssim, std_ssim),
        ("pearson_flat", "Pearson r (flattened frame)", mean_pearson, std_pearson),
    ]

    saved = []
    for key, ylabel, y, yerr in specs:
        fig, ax = plt.subplots(figsize=(8, 4))
        if yerr is not None:
            ax.errorbar(x, y, yerr=yerr, capsize=3, marker="o", linestyle="-")
        else:
            ax.plot(x, y, marker="o", linestyle="-")
        ax.set_xlabel("Clip start frame (time)")
        if key == "pearson_flat":
            ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} over time ({split_name})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, f"temporal_metrics_{split_name}_{key}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    # Optional: PSNR on masked MSE (same convention as Trainer: 10 log10(1 / mse))
    mean_psnr = [10.0 * math.log10(1.0 / max(m, 1e-10)) for m in mean_mse]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, mean_psnr, marker="o", linestyle="-")
    ax.set_xlabel("Clip start frame")
    ax.set_ylabel("PSNR (masked, dB)")
    ax.set_title(f"Masked PSNR over time ({split_name})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"temporal_metrics_{split_name}_psnr_masked_db.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    for p in saved:
        print(f"[temporal_eval_plots] Saved {p}")
    return saved
