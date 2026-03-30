"""
Interactive HDF5 viewer: visualize frames from a chosen dataset between start/end indices.

Typical VSD layout: dataset shape (n_pixels, n_frames) with n_pixels = H*H.
Also supports (H, W, T) and (T, H, W).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np


# ~5 cm per subplot in inches (matplotlib uses inches)
CM_PER_INCH = 2.54
DEFAULT_PANEL_CM = 5.0

# Curated colormaps: [1] current default + [2..6] common alternatives (see docs)
COLORMAP_PRESETS: List[Tuple[str, str]] = [
    ("hot", "Classic heat (default)"),
    ("inferno", "Perceptually uniform, dark → bright"),
    ("viridis", "Perceptually uniform, colorblind-friendly"),
    ("gray", "Grayscale"),
    ("turbo", "Rainbow-like (smoother than jet)"),
    ("coolwarm", "Diverging (best if data is centered around 0)"),
]


def _is_valid_matplotlib_cmap(name: str) -> bool:
    try:
        from matplotlib import colormaps

        colormaps[name]
        return True
    except Exception:
        try:
            import matplotlib.cm as cm

            cm.get_cmap(name)
            return True
        except Exception:
            return False


def resolve_cmap_arg(user_input: str) -> str:
    """
    Map preset index [1..N] to a preset name, or pass through a matplotlib colormap name.
    """
    s = user_input.strip()
    if not s:
        return COLORMAP_PRESETS[0][0]
    if s.isdigit():
        i = int(s)
        if 1 <= i <= len(COLORMAP_PRESETS):
            return COLORMAP_PRESETS[i - 1][0]
    low = s.lower()
    for name, _ in COLORMAP_PRESETS:
        if name.lower() == low:
            return name
    return s


def _is_perfect_square(n: int) -> bool:
    if n < 0:
        return False
    r = int(np.sqrt(n))
    return r * r == n


def _collect_datasets(h5: h5py.File) -> List[Tuple[str, Tuple[int, ...], str]]:
    items: List[Tuple[str, Tuple[int, ...], str]] = []

    def visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            items.append((name, tuple(obj.shape), str(obj.dtype)))

    h5.visititems(visitor)
    return sorted(items, key=lambda x: x[0])


def _load_dataset(h5: h5py.File, path: str) -> h5py.Dataset:
    if path not in h5:
        raise KeyError(f"Dataset not found: {path!r}")
    return h5[path]


def _slice_to_frames(arr: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Convert raw array to (H, W, T) for plotting.
    Returns (volume, layout_note).
    """
    if arr.ndim == 2:
        a, b = arr.shape
        # (n_pixels, T) — project convention
        if _is_perfect_square(a) and b <= a:
            h = int(np.sqrt(a))
            vol = arr.reshape(h, h, b)
            return vol, f"reshaped ({a}, {b}) -> ({h},{h},{b}) [pixels, frames]"
        # (T, n_pixels)
        if _is_perfect_square(b) and a <= b:
            h = int(np.sqrt(b))
            vol = arr.T.reshape(h, h, a)
            return vol, f"reshaped ({a}, {b}) -> ({h},{h},{a}) [frames, pixels] transposed"
        raise ValueError(
            f"2D array shape {arr.shape}: cannot infer square spatial layout. "
            "Expected (n_pixels, T) or (T, n_pixels) with n_pixels a perfect square."
        )

    if arr.ndim == 3:
        h, w, d = arr.shape
        # Heuristic: smallest dim is often time
        if d <= min(h, w) and d <= 64:
            return arr, f"interpreted as (H,W,T)=({h},{w},{d})"
        if h <= min(w, d) and h <= 64:
            return np.transpose(arr, (1, 2, 0)), f"interpreted as (T,H,W) -> (H,W,T)"
        # default: (H, W, T)
        return arr, f"interpreted as (H,W,T)=({h},{w},{d})"

    if arr.ndim == 4:
        # e.g. (C, H, W, T) — use first channel
        if arr.shape[0] <= 8:
            return arr[0], f"4D (C,H,W,T): used channel 0, full shape {arr.shape}"
        raise ValueError(f"Unsupported 4D shape {arr.shape}")

    raise ValueError(f"Unsupported array ndim={arr.ndim}, shape={arr.shape}")


def _extract_frame_range(
    vol: np.ndarray, start: int, end: int
) -> Tuple[np.ndarray, int, int]:
    """vol: (H, W, T). Inclusive end."""
    _, _, t = vol.shape
    start = max(0, int(start))
    end = min(t - 1, int(end))
    if start > end:
        raise ValueError(f"Invalid range: start={start} > end={end} (T={t})")
    sl = vol[:, :, start : end + 1]
    return sl, start, end


def _sanitize_for_filename(s: str) -> str:
    s = s.replace(os.sep, "_").replace("/", "_")
    s = re.sub(r"[^\w.\-]+", "_", s)
    return s.strip("_") or "dataset"


def plot_frames_grid(
    frames_hw_t: np.ndarray,
    frame_indices: range,
    title: str,
    panel_cm: float = DEFAULT_PANEL_CM,
    cmap: str = "hot",
    max_cols: int = 6,
) -> plt.Figure:
    """
    frames_hw_t: (H, W, T_local)
    frame_indices: actual frame numbers for titles
    """
    _, _, nt = frames_hw_t.shape
    ncols = min(max_cols, max(1, nt))
    nrows = int(np.ceil(nt / ncols))
    panel_in = panel_cm / CM_PER_INCH
    fig_w = ncols * panel_in * 1.15
    fig_h = nrows * panel_in * 1.2

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )
    flat = frames_hw_t.reshape(-1)
    vmin = float(np.nanpercentile(flat, 1)) if np.any(np.isfinite(flat)) else 0.0
    vmax = float(np.nanpercentile(flat, 99)) if np.any(np.isfinite(flat)) else 1.0
    if vmin >= vmax:
        vmax = vmin + 1e-6

    idx_list = list(frame_indices)
    for i in range(nrows * ncols):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        if i < nt:
            im = ax.imshow(
                frames_hw_t[:, :, i],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            ax.set_title(f"frame {idx_list[i]}", fontsize=9)
        ax.axis("off")

    fig.suptitle(title, fontsize=11, y=1.02)
    plt.tight_layout()
    return fig


def _chunked_ranges(start: int, end: int, chunk: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    s = start
    while s <= end:
        e = min(end, s + chunk - 1)
        out.append((s, e))
        s = e + 1
    return out


def _h5_files_in_folder(folder: Path) -> List[Path]:
    return sorted(folder.glob("*.h5")) + sorted(folder.glob("*.hdf5"))


def _prompt_folder_until_ok() -> Path:
    """Step (1/6): ask for folder until it exists and contains at least one .h5/.hdf5."""
    while True:
        raw = input("\n(1/6) Enter folder path (directory containing .h5 / .hdf5 files): ").strip()
        if not raw:
            print("  Please enter a non-empty path.")
            continue
        folder = Path(raw).expanduser().resolve()
        if not folder.is_dir():
            print(f"  Not a directory: {folder}")
            continue
        files = _h5_files_in_folder(folder)
        if not files:
            print(f"  No .h5 or .hdf5 files found in: {folder}")
            retry = input("  Try another folder? [Y/n]: ").strip().lower()
            if retry in ("n", "no"):
                sys.exit(0)
            continue
        return folder


def _prompt_pick_h5_file(folder: Path, h5_files: List[Path]) -> Path:
    """Step (2/6): list files and let user pick."""
    print(f"\n(2/6) HDF5 file — found {len(h5_files)} file(s) in:\n    {folder}")
    for i, p in enumerate(h5_files, 1):
        print(f"    [{i}] {p.name}")
    while True:
        sel = input("    Enter list number, or full filename (e.g. session_xxx.h5): ").strip()
        if not sel:
            print("    Please enter a number or filename.")
            continue
        if sel.isdigit():
            idx = int(sel)
            if 1 <= idx <= len(h5_files):
                return h5_files[idx - 1]
            print(f"    Enter a number between 1 and {len(h5_files)}.")
            continue
        cand = folder / sel
        if cand.is_file():
            return cand.resolve()
        matches = [p for p in h5_files if p.name == sel]
        if len(matches) == 1:
            return matches[0]
        print(f"    File not found: {cand}")


def _prompt_pick_dataset(datasets: List[Tuple[str, Tuple[int, ...], str]]) -> str:
    """Step (3/6): pick dataset path inside the file."""
    print("\n(3/6) Dataset inside this file:")
    for i, (name, shape, dtype) in enumerate(datasets, 1):
        print(f"    [{i}] {name}  shape={shape}  dtype={dtype}")
    while True:
        sel = input("    Enter list number, or full dataset path: ").strip()
        if not sel:
            print("    Please enter a number or path.")
            continue
        if sel.isdigit():
            idx = int(sel)
            if 1 <= idx <= len(datasets):
                return datasets[idx - 1][0]
            print(f"    Enter a number between 1 and {len(datasets)}.")
            continue
        return sel.strip().strip("/")


def _prompt_frame_range(t_max: int) -> Tuple[int, int]:
    """Step (4/6): start and end frame indices (inclusive)."""
    print(
        f"\n(4/6) Frame range — this dataset has T={t_max} frames "
        f"(valid indices: 0 .. {t_max - 1}, inclusive)."
    )
    while True:
        raw = input(
            "    Enter start_frame and end_frame (inclusive), e.g. 20 40: "
        ).strip()
        parts = raw.replace(",", " ").split()
        if len(parts) != 2:
            print("    Please enter exactly two integers: start end")
            continue
        try:
            sf, ef = int(parts[0]), int(parts[1])
        except ValueError:
            print("    Invalid integers.")
            continue
        if sf < 0 or ef < 0 or sf > ef:
            print("    Need 0 <= start <= end.")
            continue
        if ef > t_max - 1:
            print(f"    end_frame must be <= {t_max - 1}.")
            continue
        if sf > t_max - 1:
            print(f"    start_frame must be <= {t_max - 1}.")
            continue
        return sf, ef


def _prompt_colormap() -> str:
    """Step (5/6): pick colormap from presets or enter any matplotlib name."""
    print("\n(5/6) Colormap — choose a preset or type any matplotlib colormap name:")
    for i, (name, desc) in enumerate(COLORMAP_PRESETS, 1):
        print(f"    [{i}] {name:12} — {desc}")
    while True:
        raw = input(
            "    Enter preset number [1–6], or colormap name [default: 1 = hot]: "
        ).strip()
        if not raw:
            resolved = COLORMAP_PRESETS[0][0]
            print(f"    Using colormap: {resolved}")
            return resolved
        resolved = resolve_cmap_arg(raw)
        if _is_valid_matplotlib_cmap(resolved):
            print(f"    Using colormap: {resolved}")
            return resolved
        print(
            f"    Unknown colormap {resolved!r}. "
            "Use 1–6 or a valid matplotlib name (see matplotlib.org/colormaps)."
        )


def _prompt_save(folder: Path) -> bool:
    """Step (6/6): whether to save PNGs."""
    out_dir = folder / "frames"
    print(f"\n(6/6) Save figures under:\n    {out_dir}/")
    while True:
        ans = input("    Save figure(s) to disk? [y/N]: ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no", ""):
            return False
        print("    Please answer y or n (default: no).")


def _plot_viewer_session(
    folder: Path,
    chosen: Path,
    ds_path: str,
    vol: np.ndarray,
    note: str,
    sf_u: int,
    ef_u: int,
    do_save: bool,
    panel_cm: float,
    frames_per_figure: int,
    max_cols: int,
    show_plot: bool,
    cmap: str,
) -> None:
    """Render and optionally save figures for the selected volume and frame range."""
    stem = chosen.stem
    safe_ds = _sanitize_for_filename(ds_path)
    chunks = _chunked_ranges(sf_u, ef_u, frames_per_figure)
    out_dir = folder / "frames"
    if do_save:
        out_dir.mkdir(parents=True, exist_ok=True)

    nloc = ef_u - sf_u + 1
    print(f"\nPlotting frames {sf_u}..{ef_u} ({nloc} frames), colormap={cmap!r}.")

    for _ci, (cs, ce) in enumerate(chunks):
        sl = vol[:, :, cs : ce + 1]
        rng = range(cs, ce + 1)
        title = f"{chosen.name}\n{ds_path}\nframes {cs}–{ce} ({note})"
        fig = plot_frames_grid(
            sl,
            rng,
            title=title,
            panel_cm=panel_cm,
            max_cols=max_cols,
            cmap=cmap,
        )
        if do_save:
            fname = f"{stem}__{safe_ds}__{cs}_{ce}.png"
            fpath = out_dir / fname
            fig.savefig(fpath, dpi=150, bbox_inches="tight")
            print(f"Saved: {fpath}")
        if show_plot:
            plt.show()
        plt.close(fig)


def run_viewer(
    folder: Optional[Path],
    h5_name: Optional[str],
    dataset_path: Optional[str],
    start_frame: Optional[int],
    end_frame: Optional[int],
    save: Optional[bool],
    panel_cm: float = DEFAULT_PANEL_CM,
    frames_per_figure: int = 24,
    max_cols: int = 6,
    show_plot: bool = True,
    cmap: str = "hot",
    interactive: bool = False,
) -> None:
    """
    If interactive=True, prompts: folder → file → dataset → frames → colormap → save.
    Otherwise uses provided paths (prompts only for missing CLI pieces).
    """
    if interactive:
        print("=== HDF5 frame viewer ===")
        print(
            "Follow the prompts in order: (1) folder → (2) file → (3) dataset → "
            "(4) frames → (5) colormap → (6) save.\n"
        )

        folder = _prompt_folder_until_ok()
        h5_files = _h5_files_in_folder(folder)
        chosen = _prompt_pick_h5_file(folder, h5_files)
        print(f"\nUsing file: {chosen}")

        with h5py.File(chosen, "r") as h5:
            datasets = _collect_datasets(h5)
            if not datasets:
                print("No datasets found in file.")
                return
            ds_path = _prompt_pick_dataset(datasets)
            dset = _load_dataset(h5, ds_path)
            arr = np.asarray(dset[...])

        vol, note = _slice_to_frames(arr)
        _, _, t_max = vol.shape
        print(f"\nLayout: {note}")
        print(f"Volume shape (H, W, T) = {vol.shape}, T = {t_max} (frames 0..{t_max - 1})")

        sf, ef = _prompt_frame_range(t_max)
        _, sf_u, ef_u = _extract_frame_range(vol, sf, ef)
        cmap = _prompt_colormap()
        do_save = _prompt_save(folder)

        _plot_viewer_session(
            folder=folder,
            chosen=chosen,
            ds_path=ds_path,
            vol=vol,
            note=note,
            sf_u=sf_u,
            ef_u=ef_u,
            do_save=do_save,
            panel_cm=panel_cm,
            frames_per_figure=frames_per_figure,
            max_cols=max_cols,
            show_plot=show_plot,
            cmap=cmap,
        )
        return

    # --- Non-interactive (CLI) mode ---
    if folder is None:
        raise ValueError("folder is required when interactive=False")
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a directory: {folder}")

    h5_files = _h5_files_in_folder(folder)
    if not h5_files:
        print(f"No .h5 / .hdf5 files in {folder}")
        return

    chosen: Optional[Path] = None
    if h5_name:
        cand = folder / h5_name
        if cand.is_file():
            chosen = cand
        else:
            matches = [p for p in h5_files if p.name == h5_name]
            chosen = matches[0] if matches else None

    if chosen is None:
        print("\nAvailable HDF5 files:")
        for i, p in enumerate(h5_files, 1):
            print(f"  [{i}] {p.name}")
        sel = input("Select file number or enter filename: ").strip()
        if sel.isdigit():
            chosen = h5_files[int(sel) - 1]
        else:
            chosen = folder / sel
            if not chosen.is_file():
                raise FileNotFoundError(f"File not found: {chosen}")

    print(f"Using file: {chosen}")

    with h5py.File(chosen, "r") as h5:
        datasets = _collect_datasets(h5)
        if not datasets:
            print("No datasets found in file.")
            return

        ds_path = dataset_path
        if not ds_path:
            print("\nDatasets in file:")
            for i, (name, shape, dtype) in enumerate(datasets, 1):
                print(f"  [{i}] {name}  shape={shape}  dtype={dtype}")
            sel = input("Select dataset number or full path: ").strip()
            if sel.isdigit():
                ds_path = datasets[int(sel) - 1][0]
            else:
                ds_path = sel.strip("/")

        dset = _load_dataset(h5, ds_path)
        arr = np.asarray(dset[...])
        vol, note = _slice_to_frames(arr)
        _, _, t_max = vol.shape
        print(f"Layout: {note}")
        print(f"Volume shape (H, W, T) = {vol.shape}, T = {t_max} (frames 0..{t_max - 1})")

        sf = start_frame
        ef = end_frame
        if sf is None or ef is None:
            raw = input(
                f"Enter start and end frame indices (inclusive, 0..{t_max - 1}), e.g. 30 100: "
            ).strip()
            parts = raw.replace(",", " ").split()
            if len(parts) != 2:
                raise ValueError("Expected two integers: start end")
            sf, ef = int(parts[0]), int(parts[1])

        assert sf is not None and ef is not None
        _, sf_u, ef_u = _extract_frame_range(vol, sf, ef)

        do_save = save
        if do_save is None:
            ans = input("Save figure(s) to disk? [y/N]: ").strip().lower()
            do_save = ans in ("y", "yes")

    _plot_viewer_session(
        folder=folder,
        chosen=chosen,
        ds_path=ds_path,
        vol=vol,
        note=note,
        sf_u=sf_u,
        ef_u=ef_u,
        do_save=do_save,
        panel_cm=panel_cm,
        frames_per_figure=frames_per_figure,
        max_cols=max_cols,
        show_plot=show_plot,
        cmap=cmap,
    )


def interactive_viewer() -> None:
    run_viewer(
        folder=None,
        h5_name=None,
        dataset_path=None,
        start_frame=None,
        end_frame=None,
        save=None,
        show_plot=True,
        cmap="hot",  # ignored when interactive=True (user picks in step 5/6)
        interactive=True,
    )


def _cmap_epilog() -> str:
    lines = ["Preset colormaps (--cmap may be a name or index 1–%d):" % len(COLORMAP_PRESETS)]
    for i, (name, desc) in enumerate(COLORMAP_PRESETS, 1):
        lines.append(f"  {i}. {name}: {desc}")
    lines.append('Example: --cmap 3  or  --cmap viridis')
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description="View frames from an HDF5 dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_cmap_epilog(),
    )
    p.add_argument("--folder", type=str, default=None, help="Directory containing the .h5 file")
    p.add_argument("--file", type=str, default=None, help="HDF5 filename inside folder")
    p.add_argument("--dataset", type=str, default=None, help="Path to dataset inside file")
    p.add_argument("--start", type=int, default=None, help="Start frame (inclusive)")
    p.add_argument("--end", type=int, default=None, help="End frame (inclusive)")
    p.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save PNGs to <folder>/frames/ (default: ask)",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Do not call plt.show() (useful with --save in headless runs)",
    )
    p.add_argument("--panel-cm", type=float, default=DEFAULT_PANEL_CM, help="Subplot panel size in cm")
    p.add_argument(
        "--chunk-frames",
        type=int,
        default=24,
        help="Max frames per saved figure (split into multiple PNGs if needed)",
    )
    p.add_argument("--max-cols", type=int, default=6, help="Columns in the subplot grid")
    p.add_argument(
        "--cmap",
        type=str,
        default="hot",
        help="Colormap: preset index 1–%d or any matplotlib name (default: hot)."
        % len(COLORMAP_PRESETS),
    )
    args = p.parse_args(argv)

    if args.folder is None:
        interactive_viewer()
        return

    cmap_resolved = resolve_cmap_arg(args.cmap)
    if not _is_valid_matplotlib_cmap(cmap_resolved):
        print(f"Error: unknown colormap {cmap_resolved!r}. Use --help for presets 1–6.", file=sys.stderr)
        sys.exit(2)

    run_viewer(
        folder=Path(args.folder),
        h5_name=args.file,
        dataset_path=args.dataset,
        start_frame=args.start,
        end_frame=args.end,
        save=args.save,
        panel_cm=args.panel_cm,
        frames_per_figure=max(1, args.chunk_frames),
        max_cols=max(1, args.max_cols),
        show_plot=not args.no_show,
        cmap=cmap_resolved,
    )


if __name__ == "__main__":
    main()
