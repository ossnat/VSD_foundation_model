#!/usr/bin/env python3
"""
Draw simplified block diagrams for MAE 2D and MAE 2D+LSTM (matplotlib → PNG).

Usage:
  python scripts/plot_model_architecture.py
  python scripts/plot_model_architecture.py --out-dir docs/figures

Requires: matplotlib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def _box(ax, xy, w, h, text, fc="#E8F4FC", ec="#2C5282"):
    x, y = xy
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, wrap=True)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color="#333333",
        )
    )


def plot_mae_2d(out_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")
    ax.set_title("MAE 2D (mae_2d): ResNet18 encoder + transposed-conv decoder", fontsize=12, pad=12)

    _box(ax, (1, 11.5), 8, 1.0, "Input: masked frame\n(B, 1, H, W)")
    _arrow(ax, 5, 11.5, 5, 10.8)

    _box(ax, (0.8, 8.8), 8.4, 1.8, "MAEResNet18Backbone\n(ResNet18 trunk, no classifier)\n→ (B, 512, H/32, W/32)")
    _arrow(ax, 5, 8.8, 5, 8.0)

    _box(ax, (0.8, 5.5), 8.4, 2.2, "MAEDecoder2D\n5× ConvTranspose2d (stride 2)\n32× upsample → (B, 1, H, W)")
    _arrow(ax, 5, 5.5, 5, 4.7)

    _box(ax, (1, 2.5), 8, 1.8, "MAELoss\n(masked patches; L1 / MSE / + SSIM)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_mae_2d_lstm(out_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 18)
    ax.axis("off")
    ax.set_title("MAE 2D + LSTM (mae_2d_lstm): per-frame CNN + LSTM + shared 2D decoder", fontsize=11, pad=12)

    _box(ax, (0.8, 15.5), 8.4, 1.0, "Input: masked clip\n(B, 1, T, H, W)")
    _arrow(ax, 5, 15.5, 5, 14.8)

    _box(ax, (0.5, 12.5), 9, 2.0, "Shared MAEResNet18 per frame\n(B·T,1,H,W) → (B·T,512,H',W')\nGAP → (B, T, 512)")
    _arrow(ax, 5, 12.5, 5, 11.7)

    _box(ax, (0.8, 9.5), 8.4, 1.8, "LSTM + Linear\n→ reshape to (B, 512, T, H', W')")
    _arrow(ax, 5, 9.5, 5, 8.7)

    _box(ax, (0.8, 6.0), 8.4, 2.2, "Video2DLSTMDecoder\nMAEDecoder2D on each of T frames\n→ (B, 1, T, H, W)")
    _arrow(ax, 5, 6.0, 5, 5.2)

    _box(ax, (1, 2.5), 8, 1.8, "MAELoss on masked spatiotemporal patches")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for PNGs (default: docs/figures under project root).",
    )
    args = p.parse_args()
    root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.out_dir) if args.out_dir else root / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_mae_2d(out_dir / "mae_2d_architecture.png")
    plot_mae_2d_lstm(out_dir / "mae_2d_lstm_architecture.png")
    print("Done.")


if __name__ == "__main__":
    main()
