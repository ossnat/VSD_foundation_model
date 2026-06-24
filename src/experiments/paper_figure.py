"""
Publication-style composite figures: per-frame assets + YAML-driven panel layout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
except Exception:
    plt = None
    GridSpec = None

from src.data import load_dataset
from src.experiments.eval_core import choose_eval_indices, enforce_mae_2d_clip_contract
from src.experiments.recon_eval import (
    METRIC_CHOICES,
    build_single_h5_split_csv,
    choose_h5_trials,
    find_sample_index,
    label_for_metric,
    merge_checkpoint_config_snapshot,
    metric_value,
    normalize_trial_name,
    reconstruct_batch,
    resolve_h5_path,
    scramble_spatial,
    start_frame_from_batch,
    tensor_to_2d_frame,
)
from src.experiments.mae_2d_lstm.build_dataloaders import _filter_split_csv_missing_files
from src.experiments.mae_2d_lstm.load_config import load_and_prepare_config
from src.models import build_ssl_model
from src.experiments.eval_core import resolve_and_load_checkpoint
from src.utils.logger import set_seed


@dataclass
class FrameArrays:
    trial: str
    frame_no: int
    original: np.ndarray
    masked_input: np.ndarray
    reconstructed: np.ndarray
    mask_pixels: np.ndarray
    mask_only: np.ndarray
    metrics: Dict[str, float] = field(default_factory=dict)


def load_figure_config(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        if path.suffix in (".yaml", ".yml"):
            return yaml.safe_load(f)
        return json.load(f)


def _panel_letter(index: int) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if index < len(letters):
        return letters[index]
    return f"P{index}"


def _panels_in_reading_order(
    panels: Sequence[Dict[str, Any]],
) -> List[Tuple[int, Dict[str, Any]]]:
    return sorted(enumerate(panels), key=lambda ic: (int(ic[1]["row"]), int(ic[1]["col"])))


def _panel_letter_map(panels: Sequence[Dict[str, Any]]) -> Dict[int, str]:
    return {orig_i: _panel_letter(li) for li, (orig_i, _) in enumerate(_panels_in_reading_order(panels))}


def _figure_layout(figure_spec: Dict[str, Any], *, show_colorbar: bool) -> Dict[str, float]:
    layout = figure_spec.get("layout") or {}
    return {
        "save_pad_inches": float(
            layout.get("save_pad_inches", 0.08 if show_colorbar else 0.15)
        ),
        "left": float(layout.get("left", 0.06)),
        "right": float(layout.get("right", 0.90 if show_colorbar else 0.93)),
        "top": float(layout.get("top", 0.94)),
        "bottom": float(layout.get("bottom", 0.06)),
    }


def _figure_typography(figure_spec: Dict[str, Any]) -> Dict[str, float]:
    typo = figure_spec.get("typography") or {}
    return {
        "title_fontsize": float(typo.get("title_fontsize", figure_spec.get("title_fontsize", 16))),
        "label_fontsize": float(typo.get("label_fontsize", figure_spec.get("label_fontsize", 14))),
        "tick_fontsize": float(typo.get("tick_fontsize", figure_spec.get("tick_fontsize", 11))),
        "panel_label_fontsize": float(
            typo.get("panel_label_fontsize", figure_spec.get("panel_label_fontsize", 20))
        ),
        "suptitle_fontsize": float(typo.get("suptitle_fontsize", figure_spec.get("suptitle_fontsize", 16))),
        "legend_fontsize": float(typo.get("legend_fontsize", figure_spec.get("legend_fontsize", 11))),
        "colorbar_label_fontsize": float(
            typo.get("colorbar_label_fontsize", figure_spec.get("colorbar_label_fontsize", 12))
        ),
        "scatter_point_size": float(typo.get("scatter_point_size", figure_spec.get("scatter_point_size", 55))),
    }


def _trials_description(trials: Optional[Sequence[str]]) -> str:
    if not trials:
        return "all trials in the evaluation split"
    names = [normalize_trial_name(t) for t in trials]
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def _default_panel_caption(
    panel: Dict[str, Any],
    letter: str,
    *,
    default_trial: str,
    data_cfg: Dict[str, Any],
    scatter_cfg: Dict[str, Any],
) -> str:
    if panel.get("caption"):
        return str(panel["caption"]).strip()

    ptype = str(panel.get("type", "")).lower()
    title = panel.get("title", ptype.replace("_", " ").title())
    mask_ratio = data_cfg.get("mask_ratio", 0.75)
    frame_start = data_cfg.get("frame_start", "?")
    frame_end = data_cfg.get("frame_end", "?")
    scatter_trials = data_cfg.get("trials")
    n_points = int(scatter_cfg.get("n_points", 100))

    if ptype == "scatter":
        x_m = panel.get("x_metric", panel.get("x", "mse"))
        y_m = panel.get("y_metric", panel.get("y", "correlation"))
        kinds = scatter_cfg.get("kinds", ["real", "scrambled"])
        kind_desc = []
        if "real" in kinds:
            kind_desc.append("Blue = real frames")
        if "scrambled" in kinds:
            kind_desc.append("Red = spatially scrambled controls")
        return (
            f"Panel {letter}. {title}. "
            f"Each point is one frame sampled from v3 {data_cfg.get('split', 'test')} split"
            + (f" ({', '.join(data_cfg['monkeys'])})" if data_cfg.get("monkeys") else "")
            + f" (frames {frame_start}–{frame_end}, up to {n_points} points, seed-controlled). "
            f"X-axis: {label_for_metric(x_m)}; Y-axis: {label_for_metric(y_m)}. "
            f"{'; '.join(kind_desc)}."
        )

    trial, frame_no, h5 = _resolve_panel_frame(panel, default_trial)
    trial_frame = f"{trial}, frame {frame_no}"
    if h5:
        trial_frame += f" ({h5})"

    templates = {
        "mask_only": (
            f"Panel {letter}. {title}. Mask layout for {trial_frame}: "
            f"white patches = visible (unmasked) regions, black = masked patches "
            f"({int(float(mask_ratio) * 100)}% masking ratio). No original pixel values shown."
        ),
        "original": (
            f"Panel {letter}. {title}. Original normalized VSD frame from {trial_frame}."
        ),
        "masked_input": (
            f"Panel {letter}. {title}. Model input for {trial_frame} after "
            f"{int(float(mask_ratio) * 100)}% random patch masking; visible patches keep "
            f"original values, masked patches are set to zero."
        ),
        "reconstructed": (
            f"Panel {letter}. {title}. MAE 2D model reconstruction for {trial_frame} "
            f"from the masked input (same mask as panel masked-input for this trial/frame)."
        ),
        "diff": (
            f"Panel {letter}. {title}. Pixel-wise absolute error |original − reconstructed| "
            f"for {trial_frame}."
        ),
    }
    return templates.get(ptype, f"Panel {letter}. {title}. ({ptype}, {trial_frame}).")


def write_figure_captions(
    out_path: Path,
    panels: Sequence[Dict[str, Any]],
    *,
    default_trial: str,
    data_cfg: Dict[str, Any],
    scatter_cfg: Dict[str, Any],
    suptitle: str = "",
) -> Path:
    caption_path = out_path.with_suffix(".txt")
    letter_map = _panel_letter_map(panels)
    lines: List[str] = []
    if suptitle:
        lines.append(suptitle)
        lines.append("")
    lines.append("Figure panel captions")
    lines.append("=" * 40)
    lines.append("")

    for orig_i, panel in _panels_in_reading_order(panels):
        letter = letter_map[orig_i]
        caption = _default_panel_caption(
            panel,
            letter,
            default_trial=default_trial,
            data_cfg=data_cfg,
            scatter_cfg=scatter_cfg,
        )
        lines.append(caption)
        lines.append("")

    caption_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"[paper_figure] Saved captions: {caption_path}")
    return caption_path


def expand_patch_mask_to_pixels(
    mask: np.ndarray,
    height: int,
    width: int,
    patch_size: Sequence[int],
) -> np.ndarray:
    p_t, p_h, p_w = (int(patch_size[0]), int(patch_size[1]), int(patch_size[2]))
    if mask.ndim == 4:
        mask = mask[0]
    if mask.ndim == 3:
        n_pt, n_ph, n_pw = mask.shape
    elif mask.ndim == 2:
        n_pt, n_ph, n_pw = 1, mask.shape[0], mask.shape[1]
        mask = mask.reshape(1, n_ph, n_pw)
    else:
        raise ValueError(f"Unsupported mask shape: {mask.shape}")

    out = np.zeros((n_pt * p_t, n_ph * p_h, n_pw * p_w), dtype=np.float32)
    for t_idx in range(n_pt):
        for h_idx in range(n_ph):
            for w_idx in range(n_pw):
                val = float(mask[t_idx, h_idx, w_idx])
                ts, hs, ws = t_idx * p_t, h_idx * p_h, w_idx * p_w
                out[ts : ts + p_t, hs : hs + p_h, ws : ws + p_w] = val
    if out.shape[0] == 1:
        out = out[0]
    return out[:height, :width]


def mask_pixels_from_batch(
    batch: Dict[str, Any],
    height: int,
    width: int,
    patch_size: Sequence[int],
) -> np.ndarray:
    """Patch mask expanded to pixels (1 = visible / kept, 0 = masked)."""
    raw = batch["mask"]
    if torch.is_tensor(raw):
        raw = raw.detach().cpu().numpy()
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim == 5:
        raw = raw[0]
    if raw.ndim == 4:
        raw = raw[0]
    if raw.ndim == 3 and raw.shape[0] == 1:
        raw = raw[0]
    if raw.shape[-2] == height and raw.shape[-1] == width:
        return raw.astype(np.float32)
    return expand_patch_mask_to_pixels(raw, height, width, patch_size)


def build_mask_only_canvas(mask_pixels: np.ndarray) -> np.ndarray:
    """
    Same patch layout as masked_input: masked patches = 0 (black), visible = 1 (white).
    No original frame pixel values.
    """
    visible = mask_pixels > 0.5
    out = np.zeros_like(mask_pixels, dtype=np.float32)
    out[visible] = 1.0
    return out


def _place_panel_label_outside(
    fig: Any,
    ax: Any,
    label: str,
    typography: Dict[str, float],
) -> None:
    pos = ax.get_position()
    fig.text(
        pos.x0 - 0.006,
        pos.y1 + 0.006,
        label,
        transform=fig.transFigure,
        ha="right",
        va="bottom",
        fontsize=typography["panel_label_fontsize"],
        fontweight="bold",
        color="black",
    )


def frame_asset_dir(out_dir: Path, trial: str, frame_no: int) -> Path:
    return out_dir / "frames" / normalize_trial_name(trial) / f"frame_{frame_no:04d}"


def collect_required_trials(figure_cfg: Dict[str, Any], default_trial: str) -> List[str]:
    """Union of trials referenced in detail_frame_specs and image panels."""
    trials: set[str] = set()
    for trial, _ in collect_detail_specs(figure_cfg, default_trial):
        trials.add(trial)
    for name in figure_cfg.get("data", {}).get("trials") or []:
        trials.add(normalize_trial_name(name))
    if not trials:
        trials.add(normalize_trial_name(default_trial))
    return sorted(trials)


def collect_detail_specs(figure_cfg: Dict[str, Any], default_trial: str) -> List[Tuple[str, int]]:
    """
    (trial, frame) pairs to export. Union of detail_frame_specs / detail_frames
    and every non-scatter panel that references a frame.
    """
    seen: set[Tuple[str, int]] = set()
    out: List[Tuple[str, int]] = []

    def _add(trial: str, frame_no: int) -> None:
        key = (normalize_trial_name(trial), int(frame_no))
        if key not in seen:
            seen.add(key)
            out.append(key)

    for spec in figure_cfg.get("detail_frame_specs") or []:
        _add(spec.get("trial", default_trial), spec["frame"])

    if not figure_cfg.get("detail_frame_specs"):
        for frame_no in figure_cfg.get("detail_frames") or []:
            _add(default_trial, frame_no)

    for panel in figure_cfg.get("figure", {}).get("panels") or []:
        ptype = str(panel.get("type", "")).lower()
        if ptype == "scatter" or "frame" not in panel:
            continue
        _add(panel.get("trial", default_trial), panel["frame"])

    return out


def _arrays_from_batch(
    batch: Dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    patch_size: Sequence[int],
    trial: str,
    frame_no: int,
) -> FrameArrays:
    recon, target, masked, _ = reconstruct_batch(model, batch, device)
    orig = tensor_to_2d_frame(target).detach().cpu().numpy()
    masked_np = tensor_to_2d_frame(masked).detach().cpu().numpy()
    recon_np = tensor_to_2d_frame(recon).detach().cpu().numpy()
    h, w = orig.shape
    mask_px = mask_pixels_from_batch(batch, h, w, patch_size)
    mask_only = build_mask_only_canvas(mask_px)

    metrics = {
        "mse": metric_value("mse", recon, target),
        "correlation": metric_value("correlation", recon, target),
        "r2": metric_value("r2", recon, target),
        "ssim": metric_value("ssim", recon, target),
    }
    return FrameArrays(
        trial=trial,
        frame_no=frame_no,
        original=orig,
        masked_input=masked_np,
        reconstructed=recon_np,
        mask_pixels=mask_px,
        mask_only=mask_only,
        metrics=metrics,
    )


def export_frame_assets(
    arrays: FrameArrays,
    out_dir: Path,
    *,
    cmap: str = "hot",
    dpi: int = 150,
) -> Dict[str, Path]:
    if plt is None:
        raise RuntimeError("matplotlib is required for paper figure export")

    frame_dir = frame_asset_dir(out_dir, arrays.trial, arrays.frame_no)
    frame_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    image_keys = {
        "original": arrays.original,
        "masked_input": arrays.masked_input,
        "reconstructed": arrays.reconstructed,
        "mask_pixels": arrays.mask_pixels,
        "mask_only": arrays.mask_only,
        "diff": np.abs(arrays.original - arrays.reconstructed),
    }
    flat = np.concatenate(
        [v.ravel() for k, v in image_keys.items() if k not in ("mask_pixels", "mask_only")]
    )
    vmin, vmax = np.percentile(flat, 5), np.percentile(flat, 95)

    for key, img in image_keys.items():
        path = frame_dir / f"{key}.png"
        fig, ax = plt.subplots(figsize=(3, 3))
        if key == "mask_pixels":
            im = ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        elif key == "mask_only":
            im = ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        elif key == "diff":
            dmax = max(float(img.max()), 1e-8)
            im = ax.imshow(img, cmap="magma", vmin=0, vmax=dmax)
        else:
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(key.replace("_", " "))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths[key] = path

    meta_path = frame_dir / "metrics.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "trial": arrays.trial,
                "frame_no": arrays.frame_no,
                "metrics": arrays.metrics,
            },
            f,
            indent=2,
        )
    paths["metrics"] = meta_path
    return paths


@dataclass
class ScatterSeries:
    x_metric: str
    y_metric: str
    real_points: List[Tuple[float, float]]
    scrambled_points: List[Tuple[float, float]]


def collect_scatter_series(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    x_metric: str,
    y_metric: str,
    n_frames: int,
    seed: int,
    include_scrambled: bool = True,
) -> ScatterSeries:
    gen = torch.Generator(device=device if device.type == "cuda" else "cpu")
    gen.manual_seed(seed + 12345)
    real_pts: List[Tuple[float, float]] = []
    scr_pts: List[Tuple[float, float]] = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_frames:
                break
            if not isinstance(batch, dict) or "video_masked" not in batch:
                continue
            recon, target, _, _ = reconstruct_batch(model, batch, device)
            real_pts.append((metric_value(x_metric, recon, target), metric_value(y_metric, recon, target)))
            if include_scrambled:
                batch_scr = dict(batch)
                batch_scr["video_target"] = scramble_spatial(batch["video_target"].to(device), gen).cpu()
                batch_scr["video_masked"] = scramble_spatial(batch["video_masked"].to(device), gen).cpu()
                batch_scr["mask"] = scramble_spatial(batch["mask"].to(device), gen).cpu()
                recon_s, target_s, _, _ = reconstruct_batch(model, batch_scr, device)
                scr_pts.append(
                    (metric_value(x_metric, recon_s, target_s), metric_value(y_metric, recon_s, target_s))
                )
    return ScatterSeries(x_metric, y_metric, real_pts, scr_pts)


def _h5_basename(h5: str | None) -> str | None:
    if not h5:
        return None
    return Path(h5).name


def _frame_asset_key(trial: str, frame_no: int, h5: str | None = None) -> Tuple[str, int, str]:
    return (normalize_trial_name(trial), int(frame_no), _h5_basename(h5) or "")


def _h5_lookup_from_figure(figure_cfg: Dict[str, Any], default_trial: str) -> Dict[Tuple[str, int], str | None]:
    """Map (trial, frame) -> optional h5 basename from YAML specs and image panels."""
    out: Dict[Tuple[str, int], str | None] = {}
    for spec in figure_cfg.get("detail_frame_specs") or []:
        trial = normalize_trial_name(spec.get("trial", default_trial))
        frame_no = int(spec["frame"])
        out[(trial, frame_no)] = _h5_basename(spec.get("h5"))
    for panel in figure_cfg.get("figure", {}).get("panels") or []:
        if str(panel.get("type", "")).lower() == "scatter" or "frame" not in panel:
            continue
        trial = normalize_trial_name(panel.get("trial", default_trial))
        frame_no = int(panel["frame"])
        out[(trial, frame_no)] = _h5_basename(panel.get("h5"))
    return out


def _resolve_panel_frame(panel: Dict[str, Any], default_trial: str) -> Tuple[str, int, str | None]:
    trial = normalize_trial_name(panel.get("trial", default_trial))
    if "frame" not in panel:
        raise ValueError(f"Panel {panel.get('type')} requires 'frame'")
    return trial, int(panel["frame"]), _h5_basename(panel.get("h5"))


def compose_paper_figure(
    panels: Sequence[Dict[str, Any]],
    frame_assets: Dict[Tuple[str, int, str], FrameArrays],
    scatters: Dict[Tuple[str, str], ScatterSeries],
    out_path: Path,
    *,
    default_trial: str,
    figsize: Sequence[float] = (14, 10),
    dpi: int = 300,
    cmap: str = "hot",
    shared_color_scale: str = "global",
    show_colorbar: bool = True,
    show_panel_labels: bool = True,
    suptitle: str = "",
    typography: Optional[Dict[str, float]] = None,
    layout: Optional[Dict[str, float]] = None,
) -> Path:
    if plt is None or GridSpec is None:
        raise RuntimeError("matplotlib is required for paper figure composition")

    max_row = max(int(p["row"]) for p in panels)
    max_col = max(int(p["col"]) for p in panels)
    nrows, ncols = max_row + 1, max_col + 1

    typo = typography or _figure_typography({})
    lay = layout or _figure_layout({}, show_colorbar=show_colorbar)
    letter_map = _panel_letter_map(panels)

    fig = plt.figure(figsize=tuple(figsize), dpi=dpi)
    gs = GridSpec(nrows, ncols, figure=fig, wspace=0.25, hspace=0.3)
    if not show_colorbar:
        fig.subplots_adjust(
            left=lay["left"],
            right=lay["right"],
            top=lay["top"],
            bottom=lay["bottom"],
        )

    image_values: List[np.ndarray] = []
    for assets in frame_assets.values():
        for key in ("original", "masked_input", "reconstructed"):
            image_values.append(assets.__dict__[key].ravel())
    if image_values:
        flat = np.concatenate(image_values)
        global_vmin, global_vmax = np.percentile(flat, 5), np.percentile(flat, 95)
    else:
        global_vmin, global_vmax = 0.0, 1.0

    last_im = None
    panel_label_axes: List[Tuple[Any, str]] = []
    for pi, panel in enumerate(panels):
        row, col = int(panel["row"]), int(panel["col"])
        ax = fig.add_subplot(gs[row, col])
        ptype = str(panel["type"]).lower()
        title = panel.get("title", ptype.replace("_", " ").title())

        if ptype == "scatter":
            x_m = panel.get("x_metric", panel.get("x", "mse"))
            y_m = panel.get("y_metric", panel.get("y", "correlation"))
            series = scatters.get((x_m, y_m))
            if series is None:
                raise KeyError(f"Missing scatter data for ({x_m}, {y_m})")
            pt_size = typo["scatter_point_size"]
            if series.real_points:
                xr, yr = zip(*series.real_points)
                ax.scatter(
                    xr, yr, s=pt_size, alpha=0.8, color="#1f77b4", label="real", edgecolors="none"
                )
            if series.scrambled_points:
                xs, ys = zip(*series.scrambled_points)
                ax.scatter(
                    xs, ys, s=pt_size, alpha=0.8, color="#d62728", label="scrambled", edgecolors="none"
                )
            ax.set_xlabel(label_for_metric(x_m), fontsize=typo["label_fontsize"])
            ax.set_ylabel(label_for_metric(y_m), fontsize=typo["label_fontsize"])
            ax.tick_params(labelsize=typo["tick_fontsize"])
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=typo["legend_fontsize"])
            title = panel.get("title", f"{label_for_metric(x_m)} vs {label_for_metric(y_m)}")

        else:
            trial, frame_no, h5 = _resolve_panel_frame(panel, panel.get("trial", default_trial))
            assets = frame_assets.get(_frame_asset_key(trial, frame_no, h5))
            if assets is None:
                raise KeyError(f"Missing frame assets for trial={trial} frame={frame_no} h5={h5!r}")

            if ptype == "mask_only":
                ax.imshow(assets.mask_only, cmap="gray", vmin=0, vmax=1)
            elif ptype == "original":
                img = assets.original
                last_im = ax.imshow(img, cmap=cmap, vmin=global_vmin, vmax=global_vmax)
            elif ptype == "masked_input":
                img = assets.masked_input
                last_im = ax.imshow(img, cmap=cmap, vmin=global_vmin, vmax=global_vmax)
            elif ptype == "reconstructed":
                img = assets.reconstructed
                last_im = ax.imshow(img, cmap=cmap, vmin=global_vmin, vmax=global_vmax)
            elif ptype == "diff":
                img = np.abs(assets.original - assets.reconstructed)
                dmax = max(float(img.max()), 1e-8)
                last_im = ax.imshow(img, cmap="magma", vmin=0, vmax=dmax)
            else:
                raise ValueError(f"Unknown panel type: {ptype}")
            ax.set_xticks([])
            ax.set_yticks([])

        if show_panel_labels:
            panel_label_axes.append((ax, panel.get("label", letter_map[pi])))
        ax.set_title(title, fontsize=typo["title_fontsize"])

    if show_panel_labels and panel_label_axes:
        fig.canvas.draw()
        for ax, label in panel_label_axes:
            _place_panel_label_outside(fig, ax, label, typo)

    if show_colorbar and last_im is not None:
        cbar = fig.colorbar(
            last_im,
            ax=fig.axes,
            fraction=0.02,
            pad=0.02,
            label="normalized signal",
        )
        cbar.ax.tick_params(labelsize=typo["tick_fontsize"])
        cbar.set_label("normalized signal", fontsize=typo["colorbar_label_fontsize"])
    if suptitle:
        fig.suptitle(suptitle, y=1.02, fontsize=typo["suptitle_fontsize"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=lay["save_pad_inches"])
    plt.close(fig)
    print(f"[paper_figure] Saved composite: {out_path}")
    return out_path


def prepare_eval_cfg(
    figure_cfg: Dict[str, Any],
    project_root: Path,
    training_config: str,
    ckpt_dir: Path,
) -> Dict[str, Any]:
    data = figure_cfg.get("data", {})
    overrides: Dict[str, Any] = {}
    for key in ("monkeys", "mask_ratio", "clip_length", "patch_size", "frame_start", "frame_end"):
        if key in data and data[key] is not None:
            overrides[key] = data[key]
    if "patch_size" in overrides:
        overrides["patch_size"] = list(overrides["patch_size"])

    cfg = load_and_prepare_config(training_config, project_root=project_root, overrides=overrides)
    cfg = merge_checkpoint_config_snapshot(cfg, ckpt_dir, project_root, log_prefix="paper_figure")
    cfg.update(overrides)
    cfg = enforce_mae_2d_clip_contract(cfg)
    cfg["ckpt_dir"] = str(ckpt_dir)
    return cfg


def build_eval_loader(
    cfg: Dict[str, Any],
    figure_cfg: Dict[str, Any],
    project_root: Path,
    out_dir: Path,
) -> DataLoader:
    data = figure_cfg["data"]
    h5_root = Path(data.get("h5_root", project_root.parent / "Data" / "FoundationData" / "ProcessedData"))

    if data.get("h5_path"):
        h5_path = resolve_h5_path(data["h5_path"], h5_root)
        trials = data.get("trials")
        trial_entries = choose_h5_trials(h5_path, trials)
        split_csv = build_single_h5_split_csv(out_dir / "splits", h5_path, trial_entries)
        cfg_h5 = dict(cfg)
        cfg_h5["split_csv_path"] = str(split_csv)
        cfg_h5["monkeys"] = None
        split_name = data.get("split", "val")
        dataset_loader = load_dataset(
            cfg_h5,
            split=split_name,
            batch_size=1,
            num_workers=0,
            shuffle=False,
        )
        base_dataset = dataset_loader.dataset
    else:
        cfg_split = dict(cfg)
        split_csv_path = cfg_split.get("split_csv_path")
        if split_csv_path:
            cfg_split["split_csv_path"] = _filter_split_csv_missing_files(split_csv_path, project_root)
        split_name = data.get("split", "test")
        dataset_loader = load_dataset(
            cfg_split,
            split=split_name,
            batch_size=1,
            num_workers=0,
            shuffle=False,
        )
        base_dataset = dataset_loader.dataset

    scatter_cfg = figure_cfg.get("scatter", {})
    n_points = int(scatter_cfg.get("n_points", 100))
    seed = int(figure_cfg.get("seed", 17))
    selected_idx, _ = choose_eval_indices(
        base_dataset,
        target_files=data.get("target_files"),
        subset_size=n_points,
        seed=seed,
    )
    return DataLoader(
        Subset(base_dataset, selected_idx),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=cfg.get("pin_memory", False),
    )


def run_paper_figure_job(
    figure_cfg: Dict[str, Any],
    project_root: Path,
    *,
    figure_config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    set_seed(int(figure_cfg.get("seed", 17)))

    ckpt_dir = Path(figure_cfg["checkpoint_dir"]).resolve()
    training_config = figure_cfg.get("training_config", "configs/MAE_2D_full.yaml")
    out_dir = Path(figure_cfg.get("out_dir", ckpt_dir / "paper_figure")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = prepare_eval_cfg(figure_cfg, project_root, training_config, ckpt_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_ssl_model(cfg).to(device)
    ckpt_file, load_mode = resolve_and_load_checkpoint(
        model, ckpt_dir, figure_cfg.get("checkpoint_path"), device
    )
    model.eval()

    data = figure_cfg["data"]
    trial_list = data.get("trials") or ["trial_000000"]
    default_trial = normalize_trial_name(trial_list[0])
    patch_size = cfg.get("patch_size", [1, 8, 8])

    detail_specs = collect_detail_specs(figure_cfg, default_trial)
    detail_frames = [f for _, f in detail_specs]

    required_trials = collect_required_trials(figure_cfg, default_trial)

    if data.get("h5_path"):
        h5_root = Path(data.get("h5_root", project_root.parent / "Data" / "FoundationData" / "ProcessedData"))
        h5_path = resolve_h5_path(data["h5_path"], h5_root)
        trial_entries = choose_h5_trials(h5_path, required_trials)
        split_csv = build_single_h5_split_csv(out_dir / "splits", h5_path, trial_entries)
        cfg_h5 = dict(cfg)
        cfg_h5["split_csv_path"] = str(split_csv)
        cfg_h5["monkeys"] = None
        dataset_loader = load_dataset(cfg_h5, split="val", batch_size=1, num_workers=0, shuffle=False)
        dataset = dataset_loader.dataset
    else:
        cfg_split = dict(cfg)
        if cfg_split.get("split_csv_path"):
            cfg_split["split_csv_path"] = _filter_split_csv_missing_files(
                cfg_split["split_csv_path"], project_root
            )
        dataset_loader = load_dataset(
            cfg_split,
            split=data.get("split", "test"),
            batch_size=1,
            num_workers=0,
            shuffle=False,
        )
        dataset = dataset_loader.dataset

    frame_assets: Dict[Tuple[str, int, str], FrameArrays] = {}
    h5_lookup = _h5_lookup_from_figure(figure_cfg, default_trial)

    for trial, frame_no in detail_specs:
        h5 = h5_lookup.get((trial, frame_no))
        idx = find_sample_index(dataset, trial, frame_no, h5_basename=h5)
        batch = dataset[idx]
        batch_b = {
            k: (v.unsqueeze(0) if torch.is_tensor(v) else v)
            for k, v in batch.items()
            if k in ("video_masked", "video_target", "mask", "start_frame", "end_frame")
        }
        arrays = _arrays_from_batch(batch_b, model, device, patch_size, trial, frame_no)
        frame_assets[_frame_asset_key(trial, frame_no, h5)] = arrays
        export_frame_assets(arrays, out_dir, cmap=figure_cfg.get("figure", {}).get("cmap", "hot"))

    scatter_cfg = figure_cfg.get("scatter", {})
    scatter_kinds = scatter_cfg.get("kinds", ["real", "scrambled"])
    include_scrambled = "scrambled" in scatter_kinds
    loader = build_eval_loader(cfg, figure_cfg, project_root, out_dir)

    scatters: Dict[Tuple[str, str], ScatterSeries] = {}
    for spec in scatter_cfg.get("panels", []):
        x_m = spec.get("x_metric", spec.get("x", "mse"))
        y_m = spec.get("y_metric", spec.get("y", "correlation"))
        if x_m not in METRIC_CHOICES or y_m not in METRIC_CHOICES:
            raise ValueError(f"Invalid scatter metrics: {x_m}, {y_m}")
        scatters[(x_m, y_m)] = collect_scatter_series(
            model,
            loader,
            device,
            x_metric=x_m,
            y_metric=y_m,
            n_frames=int(scatter_cfg.get("n_points", 100)),
            seed=int(figure_cfg.get("seed", 17)),
            include_scrambled=include_scrambled,
        )

    figure_spec = figure_cfg.get("figure", {})
    panels = figure_spec.get("panels")
    if not panels:
        raise ValueError("figure.panels must be defined in the figure config")

    composite_path = out_dir / figure_spec.get("out_filename", "paper_figure_composite.png")
    suptitle = str(figure_spec.get("suptitle", ""))
    typography = _figure_typography(figure_spec)
    show_colorbar = bool(figure_spec.get("colorbar", False))
    figure_layout = _figure_layout(figure_spec, show_colorbar=show_colorbar)
    compose_paper_figure(
        panels=panels,
        frame_assets=frame_assets,
        scatters=scatters,
        out_path=composite_path,
        default_trial=default_trial,
        figsize=figure_spec.get("figsize", [14, 10]),
        dpi=int(figure_spec.get("dpi", 300)),
        cmap=figure_spec.get("cmap", "hot"),
        shared_color_scale=figure_spec.get("shared_color_scale", "global"),
        show_colorbar=show_colorbar,
        show_panel_labels=bool(figure_spec.get("panel_labels", True)),
        suptitle=suptitle,
        typography=typography,
        layout=figure_layout,
    )
    caption_path = write_figure_captions(
        composite_path,
        panels,
        default_trial=default_trial,
        data_cfg=data,
        scatter_cfg=scatter_cfg,
        suptitle=suptitle,
    )

    summary = {
        "figure_config": str(figure_config_path) if figure_config_path else None,
        "checkpoint_dir": str(ckpt_dir),
        "checkpoint_file": str(ckpt_file),
        "checkpoint_load_mode": load_mode,
        "out_dir": str(out_dir),
        "detail_frames": detail_frames,
        "detail_frame_specs": [
            {"trial": t, "frame": f} for t, f in detail_specs
        ],
        "trials_loaded": required_trials,
        "frame_asset_dirs": [
            str(frame_asset_dir(out_dir, t, f)) for t, f in detail_specs
        ],
        "composite": str(composite_path),
        "captions": str(caption_path),
        "panel_letters": {
            _panel_letter_map(panels)[i]: {
                "type": p.get("type"),
                "row": p.get("row"),
                "col": p.get("col"),
                "title": p.get("title"),
            }
            for i, p in enumerate(panels)
        },
        "scatter_metrics": list(scatters.keys()),
    }
    with open(out_dir / "paper_figure_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary
