"""
Shared reconstruction, metrics, and checkpoint-config helpers for eval scripts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import pandas as pd
import torch
import torch.nn.functional as F

from src.experiments.eval_plots import _forward_reconstruction

METRIC_CHOICES = (
    "mse",
    "rmse",
    "mae",
    "correlation",
    "r2",
    "r2_corrected",
    "ssim",
    "psnr",
)


def normalize_trial_name(name: str) -> str:
    text = str(name).strip()
    if re.match(r"^trial_\d+$", text):
        return text
    if text.isdigit():
        return f"trial_{int(text):06d}"
    return text


def label_for_metric(name: str) -> str:
    mapping = {
        "mse": "MSE",
        "rmse": "RMSE",
        "mae": "MAE",
        "correlation": "Correlation (Pearson r)",
        "r2": "R²",
        "r2_corrected": "Corrected R2",
        "ssim": "SSIM",
        "psnr": "PSNR (dB)",
    }
    return mapping.get(name.strip().lower(), name)


def safe_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    x = x.reshape(-1)
    y = y.reshape(-1)
    xc = x - x.mean()
    yc = y - y.mean()
    den = torch.sqrt((xc.pow(2).sum()) * (yc.pow(2).sum())).clamp_min(eps)
    return float((xc * yc).sum() / den)


def global_ssim(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    x = x.reshape(-1)
    y = y.reshape(-1)
    mux, muy = x.mean(), y.mean()
    sigx = x.var(unbiased=False)
    sigy = y.var(unbiased=False)
    sigxy = ((x - mux) * (y - muy)).mean()
    c1 = 0.01**2
    c2 = 0.03**2
    num = (2 * mux * muy + c1) * (2 * sigxy + c2)
    den = (mux.pow(2) + muy.pow(2) + c1) * (sigx + sigy + c2)
    return float(num / (den + eps))


def metric_value(name: str, pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.detach()
    target = target.detach()
    nm = name.strip().lower()
    d = pred - target
    mse = float(d.pow(2).mean())
    if nm == "mse":
        return mse
    if nm == "rmse":
        return float(mse**0.5)
    if nm == "mae":
        return float(d.abs().mean())
    if nm == "correlation":
        return safe_corr(pred, target)
    if nm == "r2":
        y = target.reshape(-1)
        yhat = pred.reshape(-1)
        ss_res = float(((y - yhat).pow(2)).sum())
        ss_tot = float(((y - y.mean()).pow(2)).sum())
        return 1.0 - ss_res / (ss_tot + 1e-8)
    if nm == "r2_corrected":
        y = target.reshape(-1)
        n = int(y.numel())
        p = 1
        r2 = metric_value("r2", pred, target)
        if n <= p + 1:
            return r2
        return 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
    if nm == "ssim":
        return global_ssim(pred, target)
    if nm == "psnr":
        return 10.0 * float(torch.log10(torch.tensor(1.0 / max(mse, 1e-10))))
    raise ValueError(f"Unsupported metric: {name}")


def scramble_spatial(x: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    if x.dim() == 4:
        b, c, h, w = x.shape
        flat = x.reshape(b, c, h * w)
        out = torch.empty_like(flat)
        for bi in range(b):
            perm = torch.randperm(h * w, generator=rng, device=x.device)
            out[bi] = flat[bi, :, perm]
        return out.reshape(b, c, h, w)
    if x.dim() == 5:
        b, c, t, h, w = x.shape
        flat = x.reshape(b, c, t, h * w)
        out = torch.empty_like(flat)
        for bi in range(b):
            for ti in range(t):
                perm = torch.randperm(h * w, generator=rng, device=x.device)
                out[bi, :, ti] = flat[bi, :, ti, perm]
        return out.reshape(b, c, t, h, w)
    raise ValueError(f"Unsupported tensor shape for scramble: {tuple(x.shape)}")


def reconstruct_batch(
    model: torch.nn.Module,
    batch: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns reconstruction, target, masked input, mask."""
    return _forward_reconstruction(model, batch, device)


def remap_data_path(path_value: str, data_root: Path) -> str:
    p = Path(path_value)
    if p.exists():
        return str(p.resolve())
    text = str(p).replace("\\", "/")
    marker = "/Data/"
    idx = text.find(marker)
    if idx >= 0:
        suffix = text[idx + len(marker) :]
        candidate = (data_root / suffix).resolve()
        if candidate.exists():
            return str(candidate)
    return str(p)


def merge_checkpoint_config_snapshot(
    cfg: Dict[str, Any],
    ckpt_dir: Path,
    project_root: Path,
    *,
    log_prefix: str = "recon_eval",
) -> Dict[str, Any]:
    snap_path = ckpt_dir / "analysis" / "config_used.json"
    if not snap_path.is_file():
        return cfg
    with open(snap_path, "r") as f:
        snap = json.load(f)
    for key in (
        "model",
        "backbone",
        "channels",
        "hidden_dim",
        "normalize_loss",
        "loss_type",
        "alpha",
        "ssim_window_size",
        "ssim_sigma",
        "mask_ratio",
        "clip_length",
        "patch_size",
        "frame_start",
        "frame_end",
        "monkeys",
        "crop_loss",
        "crop_loss_radius",
        "val_frame_stride",
    ):
        if key in snap:
            cfg[key] = snap[key]
    data_root = project_root.parent / "Data"
    for key in ("split_csv_path", "stats_json_path", "processed_root"):
        value = snap.get(key)
        if isinstance(value, str) and value:
            cfg[key] = remap_data_path(value, data_root)
    print(f"[{log_prefix}] Merged training snapshot: {snap_path}")
    return cfg


def collect_trial_datasets(h5_path: Path) -> List[Tuple[str, Tuple[int, ...]]]:
    out: List[Tuple[str, Tuple[int, ...]]] = []
    with h5py.File(h5_path, "r") as h5:
        def visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset) and re.match(r"^trial_\d+$", name.split("/")[-1]):
                out.append((name, tuple(obj.shape)))

        h5.visititems(visitor)
    return out


def is_square_pixels_shape(shape: Tuple[int, ...]) -> bool:
    if len(shape) != 2:
        return False
    n_pixels = int(shape[0])
    side = int(n_pixels**0.5)
    return side * side == n_pixels and shape[1] >= 1


def resolve_h5_path(path_value: str, h5_root: Path) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else (h5_root / p).resolve()


def choose_h5_trials(
    h5_path: Path,
    explicit: Optional[Sequence[str]],
) -> List[Tuple[str, Tuple[int, ...]]]:
    all_ds = collect_trial_datasets(h5_path)
    candidates = [(n, s) for n, s in all_ds if is_square_pixels_shape(s)]
    if explicit:
        wanted = {normalize_trial_name(t) for t in explicit}
        selected = [(n, s) for n, s in candidates if n.split("/")[-1] in wanted or n in wanted]
        if not selected:
            raise ValueError(f"Requested trials {sorted(wanted)} not found in {h5_path}")
        return selected
    if not candidates:
        raise ValueError(f"No compatible trial_* datasets in {h5_path}")
    return candidates


def build_single_h5_split_csv(
    out_dir: Path,
    h5_path: Path,
    trial_entries: Sequence[Tuple[str, Tuple[int, ...]]],
) -> Path:
    rows = []
    monkey = h5_path.parent.name
    for trial_dataset, shape in trial_entries:
        rows.append(
            {
                "target_file": str(h5_path),
                "trial_dataset": trial_dataset.split("/")[-1],
                "shape": str(tuple(int(x) for x in shape)),
                "split": "val",
                "monkey": monkey,
                "date": "local",
                "condition": "local",
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "single_h5_split.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def find_sample_index(
    dataset: Any,
    trial_dataset: str,
    frame_no: int,
    *,
    h5_basename: str | None = None,
) -> int:
    trial_name = normalize_trial_name(trial_dataset)
    wanted_h5 = Path(h5_basename).name if h5_basename else None
    matches: List[int] = []
    for i, (row_idx, clip_start) in enumerate(dataset.data_structure):
        row = dataset.trials.iloc[row_idx]
        ds_name = str(row["trial_dataset"])
        if ds_name != trial_name and not ds_name.endswith("/" + trial_name):
            continue
        if wanted_h5:
            tf = str(row.get("target_file", ""))
            if Path(tf).name != wanted_h5 and not tf.endswith(wanted_h5):
                continue
        if int(clip_start) == int(frame_no):
            matches.append(i)
    if not matches:
        hint = f" h5={wanted_h5!r}" if wanted_h5 else ""
        raise ValueError(f"No sample for trial={trial_name!r} frame={frame_no}{hint}")
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous sample for trial={trial_name!r} frame={frame_no}"
            + (f" h5={wanted_h5!r}" if wanted_h5 else " (add h5 to disambiguate)")
            + f": {len(matches)} matches"
        )
    return matches[0]


def start_frame_from_batch(batch: Dict[str, Any]) -> int:
    start_frames = batch.get("start_frame", None)
    if torch.is_tensor(start_frames):
        return int(start_frames[0].item())
    if isinstance(start_frames, (list, tuple)) and len(start_frames) > 0:
        return int(start_frames[0])
    if start_frames is not None:
        try:
            return int(start_frames)
        except Exception:
            pass
    return 0


def tensor_to_2d_frame(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 3:
        return t[0]
    if t.dim() == 2:
        return t
    if t.dim() == 4:
        return t[0, 0]
    if t.dim() == 5:
        return t[0, 0, 0]
    raise ValueError(f"Unsupported tensor shape for 2D frame: {tuple(t.shape)}")
