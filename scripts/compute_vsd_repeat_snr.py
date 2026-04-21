#!/usr/bin/env python3
"""
Estimate repeat-based SNR from VSD HDF5 sessions (frodo / gandalf).

Data layout (matches project CSV + datasets.py):
  - One session file may contain groups per condition (e.g. "2", "4"), each with
    datasets trial_000000, trial_000001, ... of shape (n_pixels, T) with
    n_pixels a perfect square (typically 100x100).

SNR (scalar summaries, per condition group):
  - Stack K repeats -> (K, H, W, T).
  - mu = mean over K.
  - noise_var(x,y,t) = sample variance over K (ddof=1) at each pixel/time.
  - baseline b(x,y) = mean over pre-stimulus times of mu(x,y,t).
  - evoked(x,y,t) = mu(x,y,t) - b(x,y) for t in post window.
  - signal_power = mean(evoked^2) over post window and ROI.
  - noise_power_pre = mean(noise_var) over pre window and ROI (noise floor).
  - SNR = signal_power / (noise_power_pre + eps)

ROI:
  - full square HxW
  - circle radius R (default 35) centered at spatial center-of-mass of
    mean absolute mu averaged over post-stimulus frames (falls back to image center).

Default processed root: <parent of project>/Data/FoundationData/ProcessedData
  (project = directory containing this `scripts/` folder).

Repeat grouping: by default uses `local_splits/split_v2_seed17_session_split.csv` when present,
grouping CSV rows by (HDF5 path, `condition`) so root-level `trial_*` datasets are not mixed
across conditions. Use `--scan-h5-only` to ignore the CSV (not recommended for flat layouts).

Usage:
  PYTHONPATH=. python scripts/compute_vsd_repeat_snr.py
  PYTHONPATH=. python scripts/compute_vsd_repeat_snr.py --data-root /path/to/ProcessedData
  PYTHONPATH=. python scripts/compute_vsd_repeat_snr.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:
    import h5py
except ImportError as e:  # pragma: no cover
    print("This script requires h5py. Install with: pip install h5py", file=sys.stderr)
    raise


EPS = 1e-12


def _default_processed_root(project_root: Path) -> Path:
    """Project lives in .../VSD_foundation_model; Data is sibling under parent."""
    return project_root.parent / "Data" / "FoundationData" / "ProcessedData"


def _iter_h5_files(monkey_dir: Path) -> Iterator[Path]:
    if not monkey_dir.is_dir():
        return
    for p in sorted(monkey_dir.glob("*.h5")):
        if p.is_file():
            yield p


def _reshape_pixels_time(arr: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """(n_pixels, T) -> (H, W, T)."""
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D (n_pixels, T), got shape {arr.shape}")
    n_pixels, t = arr.shape
    h = int(math.isqrt(n_pixels))
    if h * h != n_pixels:
        raise ValueError(f"n_pixels={n_pixels} is not a perfect square")
    v = arr.reshape(h, h, t).astype(np.float64, copy=False)
    return v, h, h


def _circle_mask(h: int, w: int, cy: float, cx: float, radius: float) -> np.ndarray:
    yy, xx = np.ogrid[0:h, 0:w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return (dist <= radius).astype(np.float64)


def _com_from_frame(img: np.ndarray) -> Tuple[float, float]:
    """Center of mass (y, x) for non-negative weights img (H, W)."""
    img = np.maximum(img.astype(np.float64), 0.0)
    s = img.sum()
    if s < EPS:
        h, w = img.shape
        return (h / 2.0 - 0.5, w / 2.0 - 0.5)
    yy, xx = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    cy = float((yy * img).sum() / s)
    cx = float((xx * img).sum() / s)
    return cy, cx


def _clip_frame_indices(
    t_max: int,
    pre_lo: int,
    pre_hi: int,
    post_lo: int,
    post_hi: int,
    shutter_off: Optional[int],
) -> Tuple[slice, slice]:
    """Inclusive frame indices on [0, T-1]; shutter_off exclusive (frames >= off removed)."""
    last = t_max - 1
    if shutter_off is not None:
        try:
            so = int(shutter_off)
            last = min(last, max(0, so - 1))
        except (TypeError, ValueError):
            pass
    pre_lo = max(0, min(pre_lo, last))
    pre_hi = max(0, min(pre_hi, last))
    post_lo = max(0, min(post_lo, last))
    post_hi = max(0, min(post_hi, last))
    if pre_hi < pre_lo:
        pre_lo, pre_hi = pre_hi, pre_lo
    if post_hi < post_lo:
        post_lo, post_hi = post_hi, post_lo
    return slice(pre_lo, pre_hi + 1), slice(post_lo, post_hi + 1)


def _stack_condition_trials(
    h5_path: Path,
    group_name: str,
    trial_names: Sequence[str],
) -> np.ndarray:
    """Load trials -> (K, H, W, T)."""
    arrs: List[np.ndarray] = []
    t_ref: Optional[int] = None
    with h5py.File(h5_path, "r") as f:
        grp = f[group_name]
        for name in trial_names:
            ds = grp[name]
            if not isinstance(ds, h5py.Dataset):
                continue
            v, h, w = _reshape_pixels_time(np.asarray(ds[...], dtype=np.float32))
            if t_ref is None:
                t_ref = v.shape[2]
            else:
                t_common = min(t_ref, v.shape[2])
                if t_common != v.shape[2]:
                    v = v[:, :, :t_common]
                    t_ref = t_common
            arrs.append(v)
    if not arrs:
        raise RuntimeError("No arrays loaded")
    t_final = min(a.shape[2] for a in arrs)
    out = np.stack([a[:, :, :t_final] for a in arrs], axis=0)
    return out


def _discover_condition_groups(h5_path: Path) -> List[Tuple[str, List[str]]]:
    """
    Return list of (group_name, [trial_dataset_name, ...]) with at least 2 trials,
    each trial (n_pixels, T) with square n_pixels.
    """
    found: List[Tuple[str, List[str]]] = []
    with h5py.File(h5_path, "r") as f:
        for gname in f.keys():
            obj = f[gname]
            if not isinstance(obj, h5py.Group):
                continue
            trial_names: List[str] = []
            for dname in sorted(obj.keys()):
                ds = obj[dname]
                if not isinstance(ds, h5py.Dataset) or ds.ndim != 2:
                    continue
                n0 = int(ds.shape[0])
                r = int(math.isqrt(n0))
                if r * r != n0:
                    continue
                trial_names.append(dname)
            if len(trial_names) >= 2:
                found.append((gname, trial_names))
    return found


def _discover_flat_trials(h5_path: Path) -> Optional[Tuple[str, List[str]]]:
    """If file has no condition groups, use all top-level 2D square datasets as repeats."""
    names: List[str] = []
    with h5py.File(h5_path, "r") as f:
        for dname in sorted(f.keys()):
            ds = f[dname]
            if not isinstance(ds, h5py.Dataset) or ds.ndim != 2:
                continue
            n0 = int(ds.shape[0])
            r = int(math.isqrt(n0))
            if r * r != n0:
                continue
            names.append(dname)
    if len(names) < 2:
        return None
    return ("", names)  # empty group -> special load path


def _stack_flat_trials(h5_path: Path, trial_names: Sequence[str]) -> np.ndarray:
    arrs: List[np.ndarray] = []
    with h5py.File(h5_path, "r") as f:
        for name in trial_names:
            v, _, _ = _reshape_pixels_time(np.asarray(f[name][...], dtype=np.float32))
            arrs.append(v)
    t_final = min(a.shape[2] for a in arrs)
    return np.stack([a[:, :, :t_final] for a in arrs], axis=0)


def _resolve_h5_path(target_file: Path, processed_root: Path) -> Optional[Path]:
    if target_file.is_file():
        return target_file.resolve()
    monkey = target_file.parent.name
    alt = processed_root / monkey / target_file.name
    if alt.is_file():
        return alt.resolve()
    return None


def compute_snr_for_stack(
    y: np.ndarray,
    pre_slice: slice,
    post_slice: slice,
    circle_radius: float,
) -> Dict[str, Any]:
    """
    y: (K, H, W, T)
    """
    k, h, w, t = y.shape
    if k < 2:
        raise ValueError("Need at least 2 repeats for noise variance")

    mu = y.mean(axis=0)  # (H, W, T)
    noise_var = np.nanvar(y.astype(np.float64), axis=0, ddof=1)  # (H, W, T)
    # Constant pixels across repeats -> NaN variance; treat as zero noise contribution
    noise_var = np.nan_to_num(noise_var, nan=0.0, posinf=0.0, neginf=0.0)

    # ROI masks
    mu_post = np.abs(mu[:, :, post_slice]).mean(axis=2)  # (H, W)
    cy, cx = _com_from_frame(mu_post)
    mask_circle = _circle_mask(h, w, cy, cx, circle_radius)
    mask_full = np.ones((h, w), dtype=np.float64)

    # Baseline per pixel from mu in pre window
    pre_mu = mu[:, :, pre_slice]  # (H, W, Tpre)
    b = pre_mu.mean(axis=2)  # (H, W)

    post_mu = mu[:, :, post_slice]
    # broadcast b
    evoked = post_mu - b[:, :, np.newaxis]

    def roi_mean_sq_evoked(m: np.ndarray) -> float:
        wts = m[:, :, np.newaxis]
        return float((evoked**2 * wts).sum() / (wts.sum() * evoked.shape[2] + EPS))

    def roi_mean_noise_pre(m: np.ndarray) -> float:
        nv_pre = noise_var[:, :, pre_slice]
        wts = m[:, :, np.newaxis]
        return float((nv_pre * wts).sum() / (wts.sum() * nv_pre.shape[2] + EPS))

    sig_full = roi_mean_sq_evoked(mask_full)
    noi_full = roi_mean_noise_pre(mask_full)
    sig_circ = roi_mean_sq_evoked(mask_circle)
    noi_circ = roi_mean_noise_pre(mask_circle)

    return {
        "K_repeats": k,
        "H": h,
        "W": w,
        "T": t,
        "circle_center_yx": [cy, cx],
        "circle_radius": circle_radius,
        "snr_full_field": sig_full / (noi_full + EPS),
        "snr_circle": sig_circ / (noi_circ + EPS),
        "signal_power_post_full": sig_full,
        "noise_power_pre_full": noi_full,
        "signal_power_post_circle": sig_circ,
        "noise_power_pre_circle": noi_circ,
        "mean_abs_mu_post_full": float(np.abs(mu[:, :, post_slice]).mean()),
        "mean_noise_var_pre_full": float(noise_var[:, :, pre_slice].mean()),
        "mean_noise_var_post_full": float(noise_var[:, :, post_slice].mean()),
    }


@dataclass
class GroupResult:
    monkey: str
    h5_file: str
    condition_group: str
    n_trials: int
    metrics: Dict[str, Any]


def process_h5_file(
    h5_path: Path,
    monkey: str,
    pre_lo: int,
    pre_hi: int,
    post_lo: int,
    post_hi: int,
    circle_radius: float,
    shutter_off: Optional[int],
) -> List[GroupResult]:
    out: List[GroupResult] = []
    groups = _discover_condition_groups(h5_path)
    if not groups:
        flat = _discover_flat_trials(h5_path)
        if flat is None:
            return out
        gname, trials = flat
        y = _stack_flat_trials(h5_path, trials) if gname == "" else _stack_condition_trials(h5_path, gname, trials)
        pre_sl, post_sl = _clip_frame_indices(y.shape[3], pre_lo, pre_hi, post_lo, post_hi, shutter_off)
        met = compute_snr_for_stack(y, pre_sl, post_sl, circle_radius)
        met["pre_frame_slice"] = [pre_sl.start, pre_sl.stop - 1]
        met["post_frame_slice"] = [post_sl.start, post_sl.stop - 1]
        label = "flat_root" if gname == "" else gname
        out.append(
            GroupResult(
                monkey=monkey,
                h5_file=h5_path.name,
                condition_group=label,
                n_trials=y.shape[0],
                metrics=met,
            )
        )
        return out

    for gname, trials in groups:
        y = _stack_condition_trials(h5_path, gname, trials)
        pre_sl, post_sl = _clip_frame_indices(y.shape[3], pre_lo, pre_hi, post_lo, post_hi, shutter_off)
        met = compute_snr_for_stack(y, pre_sl, post_sl, circle_radius)
        met["pre_frame_slice"] = [pre_sl.start, pre_sl.stop - 1]
        met["post_frame_slice"] = [post_sl.start, post_sl.stop - 1]
        out.append(
            GroupResult(
                monkey=monkey,
                h5_file=h5_path.name,
                condition_group=gname,
                n_trials=y.shape[0],
                metrics=met,
            )
        )
    return out


def build_repeat_groups_from_csv(
    csv_path: Path,
    processed_root: Path,
    monkeys: Sequence[str],
) -> List[Tuple[str, Path, int, List[str]]]:
    """
    From split CSV, build (monkey, h5_path, condition, [trial_dataset,...]) with >=2 repeats.

    Trial names are de-duplicated in CSV row order (same HDF5 key cannot appear twice per group).
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    need = {"target_file", "condition", "trial_dataset"}
    if not need.issubset(df.columns):
        return []

    buckets: Dict[Tuple[str, str, int], List[str]] = {}
    for _, row in df.iterrows():
        tf = Path(str(row["target_file"]))
        monkey = tf.parent.name
        if monkey not in monkeys:
            continue
        resolved = _resolve_h5_path(tf, processed_root)
        if resolved is None:
            continue
        try:
            cond = int(row["condition"])
        except (TypeError, ValueError):
            continue
        trial_ds = str(row["trial_dataset"]).strip()
        key = (monkey, str(resolved), cond)
        buckets.setdefault(key, []).append(trial_ds)

    out: List[Tuple[str, Path, int, List[str]]] = []
    for (monkey, path_s, cond), names in buckets.items():
        uniq = list(dict.fromkeys(names))
        if len(uniq) >= 2:
            out.append((monkey, Path(path_s), cond, uniq))
    return out


def filter_groups_by_max_files(
    groups: List[Tuple[str, Path, int, List[str]]],
    monkeys: Sequence[str],
    max_files: Optional[int],
) -> List[Tuple[str, Path, int, List[str]]]:
    if max_files is None or max_files <= 0:
        return groups
    allowed_by_monkey: Dict[str, set] = {}
    for m in monkeys:
        basenames = sorted({p.name for mon, p, _, _ in groups if mon == m})
        allowed_by_monkey[m] = set(basenames[:max_files])
    return [g for g in groups if g[1].name in allowed_by_monkey.get(g[0], set())]


def _load_shutter_map_from_csv(csv_path: Path) -> Dict[str, float]:
    """Map basename(target_file) -> shutter_off (first occurrence)."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    mp: Dict[str, float] = {}
    if "target_file" not in df.columns or "shutter_off" not in df.columns:
        return mp
    for _, row in df.iterrows():
        key = Path(str(row["target_file"])).name
        if key in mp:
            continue
        so = row["shutter_off"]
        if hasattr(so, "item"):
            try:
                so = float(so.item())
            except Exception:
                so = float(so)
        mp[key] = float(so) if not (isinstance(so, float) and math.isnan(so)) else float("nan")
    return mp


def run_self_test() -> None:
    """Create synthetic H5 with known SNR-ish structure and assert formulas run."""
    import tempfile

    rng = np.random.default_rng(0)
    h = w = 20
    n_pix = h * w
    t = 40
    k = 5
    pre = slice(5, 15)
    post = slice(20, 35)
    # True signal blob center
    yy, xx = np.mgrid[0:h, 0:w]
    blob = np.exp(-((yy - 10) ** 2 + (xx - 10) ** 2) / 18.0).astype(np.float64)
    base = 0.1 * rng.standard_normal((h, w, t))
    post_amp = 2.0
    mu = base.copy()
    mu[:, :, post] += post_amp * blob[:, :, np.newaxis]
    y = np.zeros((k, h, w, t), dtype=np.float64)
    for i in range(k):
        y[i] = mu + 0.5 * rng.standard_normal((h, w, t))
    tmp = Path(tempfile.mkdtemp(prefix="vsd_snr_selftest_"))
    h5_path = tmp / "synth_session.h5"
    with h5py.File(h5_path, "w") as f:
        g = f.create_group("condA")
        for i in range(k):
            flat = y[i].reshape(n_pix, t).astype(np.float32)
            g.create_dataset(f"trial_{i:06d}", data=flat)
    res = process_h5_file(h5_path, monkey="test", pre_lo=5, pre_hi=14, post_lo=20, post_hi=34, circle_radius=8.0, shutter_off=None)
    assert len(res) == 1
    snr = res[0].metrics["snr_full_field"]
    assert snr > 1.0, snr
    print("[self-test] OK synthetic session SNR:", snr)
    # cleanup
    h5_path.unlink(missing_ok=True)
    try:
        tmp.rmdir()
    except OSError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute repeat-based SNR from VSD HDF5 sessions.")
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="ProcessedData root containing frodo/ and gandalf/ (default: sibling Data/...).",
    )
    parser.add_argument(
        "--split-csv",
        type=str,
        default=None,
        help=(
            "Split CSV with columns target_file, condition, trial_dataset (and shutter_off). "
            "If omitted, uses local_splits/split_v2_seed17_session_split.csv when present. "
            "Use --scan-h5-only to ignore CSV and stack all root-level trials (not recommended)."
        ),
    )
    parser.add_argument(
        "--scan-h5-only",
        action="store_true",
        help="Do not use CSV: infer repeats from HDF5 layout only (may mix conditions if trials are flat).",
    )
    parser.add_argument("--pre-lo", type=int, default=5)
    parser.add_argument("--pre-hi", type=int, default=25)
    parser.add_argument("--post-lo", type=int, default=30)
    parser.add_argument("--post-hi", type=int, default=55)
    parser.add_argument("--circle-radius", type=float, default=35.0)
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on number of .h5 files processed per monkey (sorted by name).",
    )
    parser.add_argument("--out-json", type=str, default=None, help="Write full results JSON to this path.")
    parser.add_argument("--self-test", action="store_true", help="Run synthetic H5 sanity check and exit.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    if args.self_test:
        run_self_test()
        return

    processed = Path(args.data_root) if args.data_root else _default_processed_root(project_root)

    split_csv_path: Optional[Path] = None
    if not args.scan_h5_only:
        if args.split_csv:
            p = Path(args.split_csv)
            if not p.is_absolute():
                p = project_root / p
            split_csv_path = p if p.exists() else None
            if split_csv_path is None:
                print(f"Warning: split CSV not found at {args.split_csv!r}, trying default.")
        if split_csv_path is None:
            candid = project_root / "local_splits" / "split_v2_seed17_session_split.csv"
            if candid.exists():
                split_csv_path = candid

    shutter_map: Dict[str, float] = {}
    if split_csv_path is not None and split_csv_path.exists():
        shutter_map = _load_shutter_map_from_csv(split_csv_path)
        print(f"Loaded shutter map for {len(shutter_map)} basenames from {split_csv_path}")

    monkeys = ["frodo", "gandalf"]
    all_results: List[GroupResult] = []

    use_csv = (
        not args.scan_h5_only
        and split_csv_path is not None
        and split_csv_path.exists()
    )
    if use_csv:
        csv_groups = build_repeat_groups_from_csv(split_csv_path, processed, monkeys)
        csv_groups = filter_groups_by_max_files(csv_groups, monkeys, args.max_files)
        print(f"CSV grouping: {len(csv_groups)} session×condition groups (>=2 repeats).")
        for monkey, h5_path, cond, trial_names in csv_groups:
            so: Optional[int] = None
            key = h5_path.name
            if key in shutter_map and not (isinstance(shutter_map[key], float) and math.isnan(shutter_map[key])):
                try:
                    so = int(shutter_map[key])
                except (TypeError, ValueError):
                    so = None
            try:
                y = _stack_flat_trials(h5_path, trial_names)
                pre_sl, post_sl = _clip_frame_indices(
                    y.shape[3], args.pre_lo, args.pre_hi, args.post_lo, args.post_hi, so
                )
                met = compute_snr_for_stack(y, pre_sl, post_sl, args.circle_radius)
                met["pre_frame_slice"] = [pre_sl.start, pre_sl.stop - 1]
                met["post_frame_slice"] = [post_sl.start, post_sl.stop - 1]
                gr = GroupResult(
                    monkey=monkey,
                    h5_file=h5_path.name,
                    condition_group=f"cond_{cond}",
                    n_trials=y.shape[0],
                    metrics=met,
                )
            except Exception as e:
                print(f"  SKIP {monkey}/{h5_path.name} cond={cond}: {e}")
                continue
            all_results.append(gr)
            m = gr.metrics
            print(
                f"  {monkey}/{gr.h5_file}  {gr.condition_group}  K={m['K_repeats']}  "
                f"SNR_full={m['snr_full_field']:.4f}  SNR_circle={m['snr_circle']:.4f}  "
                f"cy_cx=({m['circle_center_yx'][0]:.2f},{m['circle_center_yx'][1]:.2f})"
            )
    else:
        print(
            "Warning: CSV grouping disabled or CSV missing. "
            "Using HDF5 layout only — flat sessions may mix experimental conditions."
        )
        for monkey in monkeys:
            mdir = processed / monkey
            if not mdir.is_dir():
                print(f"[{monkey}] directory not found: {mdir}")
                continue
            files = list(_iter_h5_files(mdir))
            if args.max_files is not None:
                files = files[: max(0, int(args.max_files))]
            print(f"[{monkey}] processing {len(files)} HDF5 file(s) from {mdir}")
            for h5_path in files:
                so: Optional[int] = None
                key = h5_path.name
                if key in shutter_map and not (
                    isinstance(shutter_map[key], float) and math.isnan(shutter_map[key])
                ):
                    try:
                        so = int(shutter_map[key])
                    except (TypeError, ValueError):
                        so = None
                try:
                    groups = process_h5_file(
                        h5_path,
                        monkey=monkey,
                        pre_lo=args.pre_lo,
                        pre_hi=args.pre_hi,
                        post_lo=args.post_lo,
                        post_hi=args.post_hi,
                        circle_radius=args.circle_radius,
                        shutter_off=so,
                    )
                except Exception as e:
                    print(f"  SKIP {h5_path.name}: {e}")
                    continue
                for gr in groups:
                    all_results.append(gr)
                    m = gr.metrics
                    print(
                        f"  {monkey}/{gr.h5_file}  group={gr.condition_group}  K={m['K_repeats']}  "
                        f"SNR_full={m['snr_full_field']:.4f}  SNR_circle={m['snr_circle']:.4f}  "
                        f"cy_cx=({m['circle_center_yx'][0]:.2f},{m['circle_center_yx'][1]:.2f})"
                    )

    # Aggregate per monkey
    summary: Dict[str, Any] = {"processed_root": str(processed), "by_monkey": {}}
    for monkey in monkeys:
        rows = [asdict(r) for r in all_results if r.monkey == monkey]
        if not rows:
            summary["by_monkey"][monkey] = {"n_groups": 0}
            continue
        snr_f = [r["metrics"]["snr_full_field"] for r in rows]
        snr_c = [r["metrics"]["snr_circle"] for r in rows]
        summary["by_monkey"][monkey] = {
            "n_groups": len(rows),
            "snr_full_field_mean": float(np.mean(snr_f)),
            "snr_full_field_median": float(np.median(snr_f)),
            "snr_circle_mean": float(np.mean(snr_c)),
            "snr_circle_median": float(np.median(snr_c)),
        }

    print("\n=== Summary (per monkey) ===")
    print(json.dumps(summary["by_monkey"], indent=2))

    payload = {"summary": summary, "groups": [asdict(r) for r in all_results]}
    if args.out_json:
        outp = Path(args.out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote full results to {outp}")


if __name__ == "__main__":
    main()
