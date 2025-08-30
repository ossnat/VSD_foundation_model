# ==================================
# File: src/utils/visualization.py
# ==================================

import torch
import torchvision.utils as vutils
import os


def save_reconstruction_grid(target, recon, mask, fname="recon.png"):
    """Save grid image with [target | recon] for the first item in batch (first 8 frames).
    target, recon: (B,C,T,H,W) tensors in [-inf, inf]
    mask: (B, N) mask in {0,1} – used only for filename annotation here.
    """
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with torch.no_grad():
        tgt = target[0, :, :8]  # (C, T, H, W)
        rec = recon[0, :, :8]
        # make image grids per time, then concatenate along width
        rows = []
        for t in range(tgt.shape[1]):
            row = torch.cat([tgt[:, t], rec[:, t]], dim=-1)  # (C, H, 2W)
            rows.append(row)
        grid = torch.cat(rows, dim=-2)  # (C, 8*H, 2W)
        vutils.save_image((grid - grid.min())/(grid.max()-grid.min()+1e-6), fname)
