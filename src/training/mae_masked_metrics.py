"""
Masked-region metrics for MAE evaluation (test-time): MSE, R², SSIM on masked pixels only.
Uses the same z-scored recon/target convention as MAESystem.
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _build_weight(mask: torch.Tensor, crop_region: Optional[torch.Tensor] = None) -> torch.Tensor:
    """mask: 1=visible, 0=masked -> weight on masked = 1-mask."""
    w = 1.0 - mask
    if crop_region is not None:
        w = w * crop_region
    return w


def masked_mse_r2_per_sample(
    recon_norm: torch.Tensor,
    target_norm: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-sample masked MSE and R² on z-scored tensors.

    R² = 1 - SS_res / SS_tot with SS_res = sum(w * (y - pred)²), SS_tot = sum(w * (y - mean_y)²).

    Returns:
        mse_masked: (B,)
        r2_masked: (B,)
        ss_tot: (B,) for debugging
    """
    B = recon_norm.shape[0]
    err2 = (recon_norm - target_norm).pow(2)
    w = weight
    if w.shape != err2.shape:
        # broadcast weight to err2
        w = w.expand_as(err2)

    flat_w = w.reshape(B, -1)
    flat_err = err2.reshape(B, -1)
    flat_y = target_norm.reshape(B, -1)

    sum_w = flat_w.sum(dim=1).clamp_min(eps)
    mse_masked = (flat_err * flat_w).sum(dim=1) / sum_w

    mean_y = (flat_y * flat_w).sum(dim=1) / sum_w
    # SS_tot per sample: sum w (y - mean_y)^2
    ss_tot = ((flat_y - mean_y.unsqueeze(1)).pow(2) * flat_w).sum(dim=1)
    ss_res = (flat_err * flat_w).sum(dim=1)
    r2 = 1.0 - ss_res / (ss_tot + eps)
    return mse_masked, r2, ss_tot


def _gaussian_window(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    window_2d = g[:, None] @ g[None, :]
    return window_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)


def _ssim_map_2d(
    x: torch.Tensor,
    y: torch.Tensor,
    window: torch.Tensor,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """
    SSIM map for single-channel images x, y: (N, 1, H, W).
    Returns (N, 1, H, W) SSIM in [-1, 1] range (typically [0,1] for aligned images).
    """
    pad = window.shape[-1] // 2
    mu_x = F.conv2d(x, window, padding=pad)
    mu_y = F.conv2d(y, window, padding=pad)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=pad) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=pad) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=pad) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / (den + 1e-12)
    return ssim_map


def masked_ssim_per_sample(
    recon_norm: torch.Tensor,
    target_norm: torch.Tensor,
    weight: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Per-sample SSIM averaged over masked pixels only (masked = where weight > 0).

    recon_norm, target_norm: (B, C, T, H, W) or (B, C, H, W)
    weight: same spatial layout as one channel, (B, 1, T, H, W) or (B, 1, H, W)
    """
    if recon_norm.dim() == 4:
        # (B, C, H, W) -> (B, 1, 1, H, W) for uniform API
        recon_norm = recon_norm.unsqueeze(2)
        target_norm = target_norm.unsqueeze(2)
        if weight.dim() == 4:
            weight = weight.unsqueeze(2)

    B, C, T, H, W = recon_norm.shape
    device = recon_norm.device
    dtype = recon_norm.dtype
    # Adaptive odd window for small frames
    ws = min(window_size, H, W)
    if ws % 2 == 0:
        ws -= 1
    ws = max(3, ws)
    window = _gaussian_window(ws, sigma, device, dtype)

    out = []
    for b in range(B):
        num = 0.0
        den = 0.0
        for t in range(T):
            x = recon_norm[b : b + 1, :, t]  # (1, C, H, W)
            y = target_norm[b : b + 1, :, t]
            w = weight[b, 0, t]  # (H, W)

            # Per-channel SSIM map then mean over C
            maps = []
            for c in range(C):
                xc = x[:, c : c + 1]
                yc = y[:, c : c + 1]
                m = _ssim_map_2d(xc, yc, window)
                maps.append(m)
            ssim_map = torch.stack(maps, dim=1).mean(dim=1, keepdim=True)  # (1,1,H,W)

            w_exp = w.unsqueeze(0).unsqueeze(0)
            num = num + (ssim_map * w_exp).sum()
            den = den + w_exp.sum()
        out.append(num / den.clamp_min(eps))
    return torch.stack(out, dim=0)


def aggregate_batch_metrics(
    recon_norm: torch.Tensor,
    target_norm: torch.Tensor,
    weight: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Batch means for mse_masked, r2_masked, ssim_masked (masked pixels only)."""
    mse_ps, r2_ps, ss_tot_ps = masked_mse_r2_per_sample(recon_norm, target_norm, weight)
    ssim_ps = masked_ssim_per_sample(recon_norm, target_norm, weight)
    return {
        "mse_masked": mse_ps.mean(),
        "r2_masked": r2_ps.mean(),
        # Used as the denominator term for R²:
        # SS_tot = sum(w * (y - mean_y)^2) on masked pixels.
        # Returned per-sample so temporal eval can debug R² dips.
        "ss_tot_masked_per_sample": ss_tot_ps,
        "ss_tot_masked": ss_tot_ps.mean(),
        "ssim_masked": ssim_ps.mean(),
        "mse_masked_per_sample": mse_ps,
        "r2_masked_per_sample": r2_ps,
        "ssim_masked_per_sample": ssim_ps,
    }
