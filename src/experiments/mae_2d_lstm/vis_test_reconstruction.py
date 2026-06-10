"""Back-compat wrapper for reconstruction plotting."""

from __future__ import annotations

from src.experiments.eval_plots import save_reconstruction_figure


def save_test_reconstruction_figure(*args, **kwargs):
    return save_reconstruction_figure(*args, **kwargs)
