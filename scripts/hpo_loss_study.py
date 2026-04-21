#!/usr/bin/env python3
"""
Small HPO study for MAE losses on:
  - mae_2d (2D CNN)
  - mae_2d_lstm (2D CNN + LSTM)

Study protocol:
  1) Tune hyperparameters within each loss family.
  2) Freeze best config per loss and compare losses with a common metric.
  3) Compare best 2D vs best 2D+LSTM per loss.

Outputs:
  - CSV/JSON tables for all trials and best selections
  - PNG plots for within-loss objective and across-loss/model comparisons

Notes:
  - Uses existing training/evaluation stack (`Trainer`, `build_dataloaders`,
    `build_ssl_model`, `load_and_prepare_config`).
  - Designed for small/medium studies (random search).
  - Supports loss families: mse, l1, l1_mse
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.experiments.mae_2d_lstm.build_dataloaders import build_dataloaders
from src.experiments.mae_2d_lstm.load_config import load_and_prepare_config
from src.models import build_ssl_model
from src.training.trainer import Trainer
from src.utils.logger import TBLogger, set_seed


LOSS_FAMILIES = ("mse", "l1", "l1_mse")
MODEL_CHOICES = ("mae_2d", "mae_2d_lstm")


@dataclass
class TrialRecord:
    model: str
    loss_type: str
    trial_idx: int
    seed: int
    objective_name: str
    objective_mode: str
    objective_value: float
    common_metric_name: str
    common_metric_value: float
    hparams: Dict[str, Any]
    val_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    trial_dir: str

    def to_flat_row(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "model": self.model,
            "loss_type": self.loss_type,
            "trial_idx": self.trial_idx,
            "seed": self.seed,
            "objective_name": self.objective_name,
            "objective_mode": self.objective_mode,
            "objective_value": self.objective_value,
            "common_metric_name": self.common_metric_name,
            "common_metric_value": self.common_metric_value,
            "trial_dir": self.trial_dir,
        }
        for k, v in self.hparams.items():
            row[f"hp_{k}"] = v
        for k, v in self.val_metrics.items():
            row[f"val_{k}"] = v
        for k, v in self.test_metrics.items():
            row[f"test_{k}"] = v
        return row


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Small HPO study across MAE losses and model families.")
    p.add_argument("--config-2d", type=str, default="configs/MAE_2D_full.yaml")
    p.add_argument("--config-2d-lstm", type=str, default="configs/MAE_2D_LSTM_full.yaml")
    p.add_argument("--models", type=str, nargs="+", default=list(MODEL_CHOICES), choices=MODEL_CHOICES)
    p.add_argument("--loss-families", type=str, nargs="+", default=list(LOSS_FAMILIES))
    p.add_argument("--trials-per-loss", type=int, default=4)
    p.add_argument("--epochs-per-trial", type=int, default=3)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--output-dir", type=str, default=None, help="Defaults to checkpoints_hpo/<timestamp>")

    # Familiar training/data overrides (same spirit as train scripts)
    p.add_argument("--monkeys", type=str, nargs="*", default=None)
    p.add_argument("--mask-ratio", type=float, default=None)
    p.add_argument("--frame-start", type=int, default=None)
    p.add_argument("--frame-end", type=int, default=None)
    p.add_argument("--clip-length", type=int, default=None, help="Used for mae_2d_lstm; mae_2d forces 1.")
    p.add_argument("--patch-size", type=int, nargs=3, metavar=("T", "H", "W"), default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--train-num-workers", type=int, default=2)
    p.add_argument("--val-num-workers", type=int, default=0)
    p.add_argument("--test-num-workers", type=int, default=0)

    # Lightweight sanity / speed controls
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--max-test-samples", type=int, default=None)
    p.add_argument("--cleanup-checkpoints", action=argparse.BooleanOptionalAction, default=True)

    # Common ranking metric across losses
    p.add_argument("--common-metric", type=str, default="ssim_masked", help="e.g. ssim_masked, mse_masked, loss")
    p.add_argument("--common-mode", type=str, choices=("max", "min"), default="max")
    return p.parse_args()


def _sample_hparams(loss_type: str, model: str, rng: random.Random) -> Dict[str, Any]:
    # Small random search space for quick studies
    hp: Dict[str, Any] = {
        "lr": 10 ** rng.uniform(math.log10(3e-5), math.log10(3e-4)),
        "weight_decay": 10 ** rng.uniform(math.log10(1e-3), math.log10(8e-2)),
    }
    if model == "mae_2d_lstm":
        hp["lstm_hidden"] = rng.choice([128, 256, 384])
    else:
        hp["hidden_dim"] = rng.choice([128, 256, 384])

    if loss_type == "l1_mse":
        hp["alpha"] = rng.choice([0.3, 0.5, 0.7, 0.84])
    return hp


def _objective_for_loss(loss_type: str, val_metrics: Dict[str, float]) -> Tuple[str, str, float]:
    # "Proper metric within family"
    #  - mse: optimize val mse_masked (min)
    #  - l1/l1_mse: optimize val loss (min), since that's the actual training objective
    if loss_type == "mse":
        return "mse_masked", "min", float(val_metrics.get("mse_masked", float("inf")))
    return "loss", "min", float(val_metrics.get("loss", float("inf")))


def _extract_common_metric(metrics: Dict[str, Any], name: str) -> float:
    v = metrics.get(name)
    if v is None:
        if name == "ssim_masked":
            return float("nan")
        if name in ("mse_masked", "loss"):
            return float("inf")
        return float("nan")
    return float(v)


def _is_better(a: float, b: float, mode: str) -> bool:
    if math.isnan(a):
        return False
    if math.isnan(b):
        return True
    return a > b if mode == "max" else a < b


def _subset_loader(loader: DataLoader, max_samples: Optional[int], num_workers: int) -> DataLoader:
    if max_samples is None:
        return loader
    n = min(max(1, int(max_samples)), len(loader.dataset))
    subset = Subset(loader.dataset, list(range(n)))
    pin_memory = getattr(loader, "pin_memory", False)
    return DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def _save_rows_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _save_json(obj: Any, out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _plot_objective_by_trial(records: List[TrialRecord], out_png: Path) -> None:
    losses = sorted({r.loss_type for r in records})
    if not losses:
        return
    n = len(losses)
    fig, axes = plt.subplots(n, 1, figsize=(8, 3 * n), squeeze=False)
    for i, loss in enumerate(losses):
        ax = axes[i, 0]
        rows = sorted([r for r in records if r.loss_type == loss], key=lambda x: x.trial_idx)
        if not rows:
            continue
        x = [r.trial_idx for r in rows]
        y = [r.objective_value for r in rows]
        ax.plot(x, y, marker="o")
        ax.set_title(f"{rows[0].model} | {loss} | objective={rows[0].objective_name} ({rows[0].objective_mode})")
        ax.set_xlabel("trial")
        ax.set_ylabel("objective")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_best_loss_bar(best_rows: List[Dict[str, Any]], metric: str, out_png: Path) -> None:
    if not best_rows:
        return
    labels = [r["loss_type"] for r in best_rows]
    vals = [float(r.get(metric, float("nan"))) for r in best_rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, vals)
    ax.set_ylabel(metric)
    ax.set_title("Best per loss (validation common metric)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _cleanup_trial_checkpoints(trial_dir: Path) -> None:
    for p in trial_dir.glob("epoch_*.pt"):
        try:
            p.unlink()
        except OSError:
            pass


def _build_trial_overrides(args: argparse.Namespace, model: str, loss_type: str, hp: Dict[str, Any]) -> Dict[str, Any]:
    o: Dict[str, Any] = {
        "model": model,
        "loss_type": loss_type,
        "epochs": int(args.epochs_per_trial),
        "seed": int(args.seed),
    }
    o.update(hp)
    if args.monkeys:
        o["monkeys"] = list(args.monkeys)
    if args.mask_ratio is not None:
        o["mask_ratio"] = float(args.mask_ratio)
    if args.frame_start is not None:
        o["frame_start"] = int(args.frame_start)
    if args.frame_end is not None:
        o["frame_end"] = int(args.frame_end)
    if args.batch_size is not None:
        o["batch_size"] = int(args.batch_size)
    if args.clip_length is not None:
        o["clip_length"] = int(args.clip_length)
    if args.patch_size is not None:
        o["patch_size"] = [int(v) for v in args.patch_size]

    # Keep mae_2d input shape valid
    if model == "mae_2d":
        o["clip_length"] = 1
        ps = o.get("patch_size")
        if ps is not None and len(ps) == 3:
            ps[0] = 1
            o["patch_size"] = ps
    return o


def _run_single_trial(
    project_root: Path,
    base_config: str,
    trial_dir: Path,
    trial_seed: int,
    args: argparse.Namespace,
    model: str,
    loss_type: str,
    hp: Dict[str, Any],
    prebuilt_loaders: Optional[Tuple[DataLoader, DataLoader, DataLoader]] = None,
) -> TrialRecord:
    set_seed(trial_seed)
    overrides = _build_trial_overrides(args, model=model, loss_type=loss_type, hp=hp)
    cfg = load_and_prepare_config(
        base_cfg_path=base_config,
        project_root=project_root,
        data_root=None,
        overrides=overrides,
    )
    cfg["ckpt_dir"] = str((trial_dir / "ckpt").resolve())
    cfg["log_dir"] = str((trial_dir / "tb").resolve())
    cfg["results_dir"] = str((trial_dir / "results").resolve())
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    os.makedirs(cfg["log_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if prebuilt_loaders is None:
        train_loader, val_loader, test_loader = build_dataloaders(
            cfg,
            project_root=project_root,
            batch_size=args.batch_size,
            train_num_workers=args.train_num_workers,
            val_num_workers=args.val_num_workers,
            test_num_workers=args.test_num_workers,
        )
        train_loader = _subset_loader(train_loader, args.max_train_samples, num_workers=args.train_num_workers)
        val_loader = _subset_loader(val_loader, args.max_val_samples, num_workers=args.val_num_workers)
        test_loader = _subset_loader(test_loader, args.max_test_samples, num_workers=args.test_num_workers)
    else:
        train_loader, val_loader, test_loader = prebuilt_loaders

    model_obj = build_ssl_model(cfg).to(device)
    trainer = Trainer(model=model_obj, logger=TBLogger(log_dir=cfg["log_dir"]), cfg=cfg, device=device)
    _ = trainer.fit(train_loader, val_loader)
    val_metrics = trainer.evaluate_metrics(val_loader, split_name="val")
    test_metrics = trainer.evaluate_metrics(test_loader, split_name="test")

    objective_name, objective_mode, objective_value = _objective_for_loss(loss_type, val_metrics)
    common_value = _extract_common_metric(val_metrics, args.common_metric)
    rec = TrialRecord(
        model=model,
        loss_type=loss_type,
        trial_idx=-1,  # assigned by caller
        seed=trial_seed,
        objective_name=objective_name,
        objective_mode=objective_mode,
        objective_value=objective_value,
        common_metric_name=args.common_metric,
        common_metric_value=common_value,
        hparams=hp,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        trial_dir=str(trial_dir),
    )
    return rec


def _build_cached_loaders_for_model(
    project_root: Path,
    base_config: str,
    args: argparse.Namespace,
    model: str,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build and subset train/val/test loaders once per model and reuse across trials.
    This avoids repeated CPU preload cost in HPO loops.
    """
    data_overrides = _build_trial_overrides(args, model=model, loss_type="mse", hp={})
    cfg_data = load_and_prepare_config(
        base_cfg_path=base_config,
        project_root=project_root,
        data_root=None,
        overrides=data_overrides,
    )
    train_loader, val_loader, test_loader = build_dataloaders(
        cfg_data,
        project_root=project_root,
        batch_size=args.batch_size,
        train_num_workers=args.train_num_workers,
        val_num_workers=args.val_num_workers,
        test_num_workers=args.test_num_workers,
    )
    train_loader = _subset_loader(train_loader, args.max_train_samples, num_workers=args.train_num_workers)
    val_loader = _subset_loader(val_loader, args.max_val_samples, num_workers=args.val_num_workers)
    test_loader = _subset_loader(test_loader, args.max_test_samples, num_workers=args.test_num_workers)
    return train_loader, val_loader, test_loader


def _select_best_by_objective(records: Sequence[TrialRecord]) -> Optional[TrialRecord]:
    if not records:
        return None
    mode = records[0].objective_mode
    best = records[0]
    for r in records[1:]:
        if _is_better(r.objective_value, best.objective_value, mode):
            best = r
    return best


def _build_model_summary_rows(
    model: str,
    records: List[TrialRecord],
    loss_families: Sequence[str],
    common_metric: str,
    common_mode: str,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    best_per_loss: List[Dict[str, Any]] = []
    for loss in loss_families:
        rows = [r for r in records if r.loss_type == loss]
        best = _select_best_by_objective(rows)
        if best is None:
            continue
        row = {
            "model": model,
            "loss_type": loss,
            "best_trial_idx": best.trial_idx,
            "objective_name": best.objective_name,
            "objective_mode": best.objective_mode,
            "objective_value": best.objective_value,
            "val_common_metric": best.common_metric_value,
            "test_common_metric": _extract_common_metric(best.test_metrics, common_metric),
        }
        row.update({f"val_{k}": v for k, v in best.val_metrics.items()})
        row.update({f"test_{k}": v for k, v in best.test_metrics.items()})
        row.update({f"hp_{k}": v for k, v in best.hparams.items()})
        best_per_loss.append(row)

    best_across: Optional[Dict[str, Any]] = None
    for row in best_per_loss:
        v = float(row.get("val_common_metric", float("nan")))
        if best_across is None or _is_better(v, float(best_across.get("val_common_metric", float("nan"))), common_mode):
            best_across = row
    return best_per_loss, best_across


def main() -> None:
    args = _parse_args()
    invalid_losses = [x for x in args.loss_families if x not in LOSS_FAMILIES]
    if invalid_losses:
        raise ValueError(f"Unsupported loss families: {invalid_losses}. Supported: {list(LOSS_FAMILIES)}")

    project_root = Path(__file__).resolve().parent.parent
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_dir) if args.output_dir else (project_root / "checkpoints_hpo" / f"loss_study_{now}")
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[hpo] output dir: {out_root}")
    set_seed(args.seed)
    rng = random.Random(args.seed)

    config_by_model = {
        "mae_2d": args.config_2d,
        "mae_2d_lstm": args.config_2d_lstm,
    }

    all_records: List[TrialRecord] = []
    best_across_models: List[Dict[str, Any]] = []
    best_per_loss_by_model: Dict[str, List[Dict[str, Any]]] = {}
    cached_loaders_by_model: Dict[str, Tuple[DataLoader, DataLoader, DataLoader]] = {}

    total_trials = len(args.models) * len(args.loss_families) * int(args.trials_per_loss)
    completed_trials = 0

    print(
        "[hpo] plan: "
        f"models={args.models}, losses={args.loss_families}, "
        f"trials_per_loss={args.trials_per_loss}, total_trials={total_trials}"
    )

    for model in args.models:
        print(f"\n[hpo] === Model: {model} ===")
        print(f"[hpo] building and caching loaders once for model={model} ...")
        cached_loaders_by_model[model] = _build_cached_loaders_for_model(
            project_root=project_root,
            base_config=config_by_model[model],
            args=args,
            model=model,
        )
        tr, va, te = cached_loaders_by_model[model]
        print(
            f"[hpo] cached loader sizes for {model}: "
            f"train={len(tr.dataset)}, val={len(va.dataset)}, test={len(te.dataset)}"
        )
        model_records: List[TrialRecord] = []
        for loss in args.loss_families:
            print(f"[hpo] -- Loss family: {loss}")
            for t_idx in range(1, int(args.trials_per_loss) + 1):
                trial_seed = args.seed + (1000 * (args.models.index(model) + 1)) + (100 * (args.loss_families.index(loss) + 1)) + t_idx
                hp = _sample_hparams(loss_type=loss, model=model, rng=rng)
                trial_dir = out_root / model / loss / f"trial_{t_idx:03d}"
                completed_trials += 1
                print(
                    f"[hpo] trial {completed_trials}/{total_trials} | "
                    f"model={model} | loss={loss} | family_trial={t_idx}/{args.trials_per_loss}"
                )
                print(
                    "[hpo] params: "
                    f"seed={trial_seed}, monkeys={args.monkeys}, mask_ratio={args.mask_ratio}, "
                    f"frame=[{args.frame_start},{args.frame_end}], batch_size={args.batch_size}, "
                    f"clip_length={args.clip_length}, patch_size={args.patch_size}, hp={hp}"
                )
                rec = _run_single_trial(
                    project_root=project_root,
                    base_config=config_by_model[model],
                    trial_dir=trial_dir,
                    trial_seed=trial_seed,
                    args=args,
                    model=model,
                    loss_type=loss,
                    hp=hp,
                    prebuilt_loaders=cached_loaders_by_model[model],
                )
                rec.trial_idx = t_idx
                model_records.append(rec)
                all_records.append(rec)
                print(
                    "[hpo] trial done: "
                    f"objective({rec.objective_name},{rec.objective_mode})={rec.objective_value:.6f}, "
                    f"val_{args.common_metric}={rec.common_metric_value:.6f}, "
                    f"test_{args.common_metric}={_extract_common_metric(rec.test_metrics, args.common_metric):.6f}, "
                    f"trial_dir={trial_dir}"
                )
                if args.cleanup_checkpoints:
                    _cleanup_trial_checkpoints(trial_dir / "ckpt")

        # Save per-model trial table
        trial_rows = [r.to_flat_row() for r in model_records]
        model_dir = out_root / model
        _save_rows_csv(trial_rows, model_dir / "hpo_trials.csv")
        _save_json(trial_rows, model_dir / "hpo_trials.json")
        _plot_objective_by_trial(model_records, model_dir / "objective_by_trial.png")

        best_per_loss, best_across = _build_model_summary_rows(
            model=model,
            records=model_records,
            loss_families=args.loss_families,
            common_metric=args.common_metric,
            common_mode=args.common_mode,
        )
        _save_rows_csv(best_per_loss, model_dir / "best_per_loss.csv")
        _save_json(best_per_loss, model_dir / "best_per_loss.json")
        _plot_best_loss_bar(best_per_loss, metric="val_common_metric", out_png=model_dir / "best_per_loss_common_metric.png")
        best_per_loss_by_model[model] = best_per_loss
        if best_across is not None:
            best_across_models.append(best_across)

    # Global summary files
    all_rows = [r.to_flat_row() for r in all_records]
    _save_rows_csv(all_rows, out_root / "all_models_hpo_trials.csv")
    _save_json(all_rows, out_root / "all_models_hpo_trials.json")

    _save_rows_csv(best_across_models, out_root / "best_model_per_loss_summary.csv")
    _save_json(best_across_models, out_root / "best_model_per_loss_summary.json")

    # Compare best 2D vs best 2D+LSTM for each loss family
    cross_model_rows: List[Dict[str, Any]] = []
    if len(best_per_loss_by_model) >= 2:
        index: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for model_name, rows in best_per_loss_by_model.items():
            for r in rows:
                index[(model_name, r["loss_type"])] = r
        for loss in args.loss_families:
            r2d = index.get(("mae_2d", loss))
            rlstm = index.get(("mae_2d_lstm", loss))
            if r2d is None and rlstm is None:
                continue
            row: Dict[str, Any] = {"loss_type": loss}
            if r2d is not None:
                row["mae_2d_val_common"] = r2d.get("val_common_metric")
                row["mae_2d_test_common"] = r2d.get("test_common_metric")
            if rlstm is not None:
                row["mae_2d_lstm_val_common"] = rlstm.get("val_common_metric")
                row["mae_2d_lstm_test_common"] = rlstm.get("test_common_metric")

            a = float(row.get("mae_2d_val_common", "nan")) if "mae_2d_val_common" in row else float("nan")
            b = float(row.get("mae_2d_lstm_val_common", "nan")) if "mae_2d_lstm_val_common" in row else float("nan")
            if _is_better(a, b, args.common_mode):
                row["winner_by_val_common"] = "mae_2d"
            elif _is_better(b, a, args.common_mode):
                row["winner_by_val_common"] = "mae_2d_lstm"
            else:
                row["winner_by_val_common"] = "tie_or_nan"
            cross_model_rows.append(row)

    _save_rows_csv(cross_model_rows, out_root / "cross_model_per_loss.csv")
    _save_json(cross_model_rows, out_root / "cross_model_per_loss.json")

    if best_across_models:
        labels = [f"{r['model']}|{r['loss_type']}" for r in best_across_models]
        vals = [float(r.get("val_common_metric", float("nan"))) for r in best_across_models]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(labels, vals)
        ax.set_ylabel(f"val_{args.common_metric}")
        ax.set_title("Best-per-loss candidates across models")
        ax.tick_params(axis="x", labelrotation=20)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_root / "cross_model_best_candidates.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    config_snapshot = {
        "args": vars(args),
        "supported_losses": list(LOSS_FAMILIES),
        "models": list(args.models),
    }
    _save_json(config_snapshot, out_root / "study_config.json")
    print(f"\n[hpo] done. Results saved in: {out_root}")


if __name__ == "__main__":
    main()

