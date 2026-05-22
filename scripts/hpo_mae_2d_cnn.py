#!/usr/bin/env python3
"""
MAE 2D CNN hyperparameter study (step 1 of a 3-model HPO split).

Three phases (run all or individually via --phase):
  1) hparam   — random search over MAE-2D-relevant knobs (loss, lr, wd, hidden_dim, mask_ratio)
  2) scheduler — fixed LR vs cosine+warmup (best hparams from phase 1)
  3) early_stop — no early stopping vs patience-based stop (best scheduler from phase 2)

Each trial writes under ``<output_dir>/<phase>/<trial_name>/``:
  - ckpt/, logs/, eval/ (metrics JSON, temporal_eval + pearson-over-time plots, recon PNGs)
  - summary CSV/JSON and comparison bar charts under ``<output_dir>/summary/``

Usage (repo root):
  PYTHONPATH=. python scripts/hpo_mae_2d_cnn.py --output-dir runs/hpo_mae2d_cnn
  PYTHONPATH=. python scripts/hpo_mae_2d_cnn.py --phase hparam --quick
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.experiments.mae_2d_lstm.build_dataloaders import build_dataloaders
from src.experiments.mae_2d_lstm.load_config import load_and_prepare_config
from src.experiments.mae_2d_lstm.temporal_eval_plots import save_temporal_per_metric_figures
from src.experiments.mae_2d_lstm.vis_test_reconstruction import save_test_reconstruction_figure
from src.models import build_ssl_model
from src.training.trainer import Trainer
from src.utils.logger import TBLogger, set_seed

LOSS_TYPES = ("mse", "l1", "l1_mse")
PHASES = ("hparam", "scheduler", "early_stop", "all")


@dataclass
class TrialResult:
    phase: str
    trial_id: str
    seed: int
    hparams: Dict[str, Any]
    val_metrics: Dict[str, Any] = field(default_factory=dict)
    test_metrics: Dict[str, Any] = field(default_factory=dict)
    train_history: Dict[str, Any] = field(default_factory=dict)
    trial_dir: str = ""
    ranking_metric: str = "ssim_masked"
    ranking_value: float = float("nan")

    def flat_row(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "phase": self.phase,
            "trial_id": self.trial_id,
            "seed": self.seed,
            "trial_dir": self.trial_dir,
            "ranking_metric": self.ranking_metric,
            "ranking_value": self.ranking_value,
            "epochs_completed": self.train_history.get("epochs_completed"),
            "stopped_early": self.train_history.get("stopped_early"),
        }
        for k, v in self.hparams.items():
            row[f"hp_{k}"] = v
        for k, v in self.val_metrics.items():
            row[f"val_{k}"] = v
        for k, v in self.test_metrics.items():
            row[f"test_{k}"] = v
        return row


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAE 2D CNN HPO (3-phase study).")
    p.add_argument("--config", type=str, default="configs/MAE_2D_full.yaml")
    p.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=PHASES,
        help="Run one phase or all three sequentially.",
    )
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--ranking-metric", type=str, default="ssim_masked")
    p.add_argument("--ranking-mode", type=str, choices=("max", "min"), default="max")

    p.add_argument("--hparam-trials", type=int, default=8, help="Random trials in phase 1.")
    p.add_argument("--epochs", type=int, default=15, help="Max epochs per trial.")
    p.add_argument("--quick", action="store_true", help="Fewer trials/epochs and data cap.")
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--max-test-samples", type=int, default=None)

    p.add_argument("--monkeys", type=str, nargs="*", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--frame-start", type=int, default=None)
    p.add_argument("--frame-end", type=int, default=None)
    p.add_argument("--train-num-workers", type=int, default=4)
    p.add_argument("--val-num-workers", type=int, default=2)
    p.add_argument("--test-num-workers", type=int, default=2)
    p.add_argument("--cleanup-epoch-ckpts", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--early-stopping-patience", type=int, default=4)
    p.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    return p.parse_args()


def _is_better(a: float, b: float, mode: str) -> bool:
    if math.isnan(a):
        return False
    if math.isnan(b):
        return True
    return a > b if mode == "max" else a < b


def _metric_value(metrics: Dict[str, Any], name: str) -> float:
    v = metrics.get(name)
    if v is None:
        return float("nan") if name == "ssim_masked" else float("inf")
    return float(v)


def _subset_loader(loader: DataLoader, max_samples: Optional[int], num_workers: int) -> DataLoader:
    if max_samples is None:
        return loader
    n = min(max(1, int(max_samples)), len(loader.dataset))
    subset = Subset(loader.dataset, list(range(n)))
    return DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=getattr(loader, "shuffle", False),
        num_workers=num_workers,
        pin_memory=getattr(loader, "pin_memory", False),
    )


def _save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _sample_hparams(rng: random.Random) -> Dict[str, Any]:
    loss_type = rng.choice(list(LOSS_TYPES))
    hp: Dict[str, Any] = {
        "loss_type": loss_type,
        "lr": 10 ** rng.uniform(math.log10(3e-5), math.log10(3e-4)),
        "weight_decay": 10 ** rng.uniform(math.log10(1e-3), math.log10(8e-2)),
        "hidden_dim": rng.choice([128, 256, 384]),
        "mask_ratio": rng.choice([0.5, 0.6, 0.75]),
        "scheduler_type": "none",
        "early_stopping": False,
        "auto_resume": False,
        "resume": False,
        "use_amp": False,
        "clip_length": 1,
    }
    if loss_type == "l1_mse":
        hp["alpha"] = rng.choice([0.3, 0.5, 0.7, 0.84])
    return hp


def _base_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    o: Dict[str, Any] = {
        "model": "mae_2d",
        "epochs": args.epochs,
        "auto_resume": False,
        "resume": False,
        "use_amp": False,
        "clip_length": 1,
        "save_metrics_each_epoch": True,
        "temporal_eval_per_metric_plots": True,
    }
    if args.monkeys:
        o["monkeys"] = list(args.monkeys)
    if args.batch_size is not None:
        o["batch_size"] = int(args.batch_size)
    if args.frame_start is not None:
        o["frame_start"] = int(args.frame_start)
    if args.frame_end is not None:
        o["frame_end"] = int(args.frame_end)
    return o


def _build_loaders(
    project_root: Path,
    config_path: str,
    args: argparse.Namespace,
    overrides: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    cfg = load_and_prepare_config(
        base_cfg_path=config_path,
        project_root=project_root,
        overrides={**_base_overrides(args), **overrides},
    )
    cfg["model"] = "mae_2d"
    cfg["clip_length"] = 1
    train_loader, val_loader, test_loader = build_dataloaders(
        cfg,
        project_root=project_root,
        batch_size=args.batch_size,
        train_num_workers=args.train_num_workers,
        val_num_workers=args.val_num_workers,
        test_num_workers=args.test_num_workers,
    )
    train_loader = _subset_loader(train_loader, args.max_train_samples, args.train_num_workers)
    val_loader = _subset_loader(val_loader, args.max_val_samples, args.val_num_workers)
    test_loader = _subset_loader(test_loader, args.max_test_samples, args.test_num_workers)
    return train_loader, val_loader, test_loader


def _run_post_train_eval(
    trainer: Trainer,
    model: torch.nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    eval_dir: Path,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[int, Dict], Dict[int, Dict]]:
    """Metrics + temporal eval (incl. pearson-over-time) + recon figures under eval_dir."""
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = eval_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    val_metrics = trainer.evaluate_metrics(val_loader, split_name="val")
    test_metrics = trainer.evaluate_metrics(test_loader, split_name="test")
    _save_json(val_metrics, metrics_dir / "metrics_val.json")
    _save_json(test_metrics, metrics_dir / "metrics_test.json")

    temporal_dir = str(eval_dir / "temporal_eval")
    temporal_val = trainer.evaluate_metrics_over_time(
        val_loader, split_name="val", save_dir=temporal_dir
    )
    temporal_test = trainer.evaluate_metrics_over_time(
        test_loader, split_name="test", save_dir=temporal_dir
    )
    save_temporal_per_metric_figures(temporal_val, temporal_dir, split_name="val")
    save_temporal_per_metric_figures(temporal_test, temporal_dir, split_name="test")

    for split, loader in (("val", val_loader), ("test", test_loader)):
        save_test_reconstruction_figure(
            model, loader, device, out_dir=temporal_dir,
            split_name=split, num_batches=1, max_frames_per_clip=8, plot_masked=False,
        )
        save_test_reconstruction_figure(
            model, loader, device, out_dir=temporal_dir,
            split_name=split, num_batches=1, max_frames_per_clip=8, plot_masked=True,
        )

    return val_metrics, test_metrics, temporal_val, temporal_test


def _cleanup_epoch_ckpts(ckpt_dir: Path) -> None:
    for p in ckpt_dir.glob("epoch_*.pt"):
        try:
            p.unlink()
        except OSError:
            pass


def _run_trial(
    project_root: Path,
    args: argparse.Namespace,
    phase: str,
    trial_id: str,
    trial_dir: Path,
    hp: Dict[str, Any],
    trial_seed: int,
    cached_loaders: Optional[Tuple[DataLoader, DataLoader, DataLoader]] = None,
) -> TrialResult:
    set_seed(trial_seed)
    trial_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = trial_dir / "ckpt"
    log_dir = trial_dir / "logs"
    eval_dir = trial_dir / "eval"

    overrides = dict(hp)
    overrides.update(
        {
            "ckpt_dir": str(ckpt_dir.resolve()),
            "log_dir": str(log_dir.resolve()),
            "results_dir": str(eval_dir.resolve()),
            "warmup_epochs": args.warmup_epochs,
            "min_lr": args.min_lr,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "early_stopping_metric": args.ranking_metric,
            "early_stopping_mode": args.ranking_mode,
        }
    )

    cfg = load_and_prepare_config(
        base_cfg_path=args.config,
        project_root=project_root,
        overrides={**_base_overrides(args), **overrides},
    )
    cfg["model"] = "mae_2d"
    cfg["clip_length"] = 1

    if cached_loaders is None:
        train_loader, val_loader, test_loader = build_dataloaders(
            cfg,
            project_root=project_root,
            batch_size=args.batch_size,
            train_num_workers=args.train_num_workers,
            val_num_workers=args.val_num_workers,
            test_num_workers=args.test_num_workers,
        )
        train_loader = _subset_loader(train_loader, args.max_train_samples, args.train_num_workers)
        val_loader = _subset_loader(val_loader, args.max_val_samples, args.val_num_workers)
        test_loader = _subset_loader(test_loader, args.max_test_samples, args.test_num_workers)
    else:
        train_loader, val_loader, test_loader = cached_loaders

    _save_json(cfg, trial_dir / "config_used.json")
    _save_json(hp, trial_dir / "hparams.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_ssl_model(cfg).to(device)
    trainer = Trainer(model=model, logger=TBLogger(log_dir=str(log_dir)), cfg=cfg, device=device)
    history = trainer.fit(train_loader, val_loader)

    val_metrics, test_metrics, _, _ = _run_post_train_eval(
        trainer, model, val_loader, test_loader, device, eval_dir
    )

    if args.cleanup_epoch_ckpts:
        _cleanup_epoch_ckpts(ckpt_dir)

    rank_val = _metric_value(val_metrics, args.ranking_metric)
    rec = TrialResult(
        phase=phase,
        trial_id=trial_id,
        seed=trial_seed,
        hparams=hp,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        train_history=history,
        trial_dir=str(trial_dir),
        ranking_metric=args.ranking_metric,
        ranking_value=rank_val,
    )
    _save_json(rec.flat_row(), trial_dir / "trial_summary.json")
    print(
        f"[{phase}/{trial_id}] {args.ranking_metric}={rank_val:.6f} "
        f"epochs={history.get('epochs_completed')} stopped_early={history.get('stopped_early')}"
    )
    return rec


def _select_best(
    records: Sequence[TrialResult],
    mode: str,
) -> Optional[TrialResult]:
    best: Optional[TrialResult] = None
    for r in records:
        if best is None or _is_better(r.ranking_value, best.ranking_value, mode):
            best = r
    return best


def _phase_hparam(
    project_root: Path,
    out_root: Path,
    args: argparse.Namespace,
    rng: random.Random,
) -> Tuple[List[TrialResult], Optional[TrialResult]]:
    phase_dir = out_root / "phase1_hparam_search"
    phase_dir.mkdir(parents=True, exist_ok=True)
    n_trials = 3 if args.quick else args.hparam_trials

    cached = _build_loaders(project_root, args.config, args, {"loss_type": "mse"})

    records: List[TrialResult] = []
    for i in range(n_trials):
        hp = _sample_hparams(rng)
        tid = f"trial_{i:03d}"
        rec = _run_trial(
            project_root,
            args,
            "hparam",
            tid,
            phase_dir / tid,
            hp,
            trial_seed=args.seed + i,
            cached_loaders=cached,
        )
        rec.trial_id = tid
        records.append(rec)

    best = _select_best(records, args.ranking_mode)
    _save_json(
        {"best_trial_id": best.trial_id if best else None, "hparams": best.hparams if best else {}},
        phase_dir / "best_hparams.json",
    )
    return records, best


def _phase_scheduler(
    project_root: Path,
    out_root: Path,
    args: argparse.Namespace,
    base_hp: Dict[str, Any],
) -> Tuple[List[TrialResult], Optional[TrialResult]]:
    phase_dir = out_root / "phase2_scheduler"
    phase_dir.mkdir(parents=True, exist_ok=True)
    arms = [
        ("fixed_lr", {"scheduler_type": "none", "early_stopping": False}),
        (
            "cosine_warmup",
            {
                "scheduler_type": "cosine",
                "warmup_epochs": args.warmup_epochs,
                "min_lr": args.min_lr,
                "early_stopping": False,
            },
        ),
    ]
    records: List[TrialResult] = []
    for i, (name, sched_hp) in enumerate(arms):
        hp = {**base_hp, **sched_hp}
        rec = _run_trial(
            project_root,
            args,
            "scheduler",
            name,
            phase_dir / name,
            hp,
            trial_seed=args.seed + 100 + i,
        )
        records.append(rec)

    best = _select_best(records, args.ranking_mode)
    _save_json(
        {
            "best_arm": best.trial_id if best else None,
            "hparams": best.hparams if best else {},
        },
        phase_dir / "best_scheduler.json",
    )
    return records, best


def _phase_early_stop(
    project_root: Path,
    out_root: Path,
    args: argparse.Namespace,
    base_hp: Dict[str, Any],
) -> List[TrialResult]:
    phase_dir = out_root / "phase3_early_stop"
    phase_dir.mkdir(parents=True, exist_ok=True)
    arms = [
        ("no_early_stop", {"early_stopping": False}),
        (
            "early_stop",
            {
                "early_stopping": True,
                "early_stopping_patience": args.early_stopping_patience,
                "early_stopping_min_delta": args.early_stopping_min_delta,
            },
        ),
    ]
    records: List[TrialResult] = []
    for i, (name, es_hp) in enumerate(arms):
        hp = {**base_hp, **es_hp}
        rec = _run_trial(
            project_root,
            args,
            "early_stop",
            name,
            phase_dir / name,
            hp,
            trial_seed=args.seed + 200 + i,
        )
        records.append(rec)

    best = _select_best(records, args.ranking_mode)
    _save_json(
        {"best_arm": best.trial_id if best else None, "hparams": best.hparams if best else {}},
        phase_dir / "best_early_stop.json",
    )
    return records


def _plot_phase_comparison(
    records: Sequence[TrialResult],
    title: str,
    out_png: Path,
    metric: str,
) -> None:
    if not records:
        return
    labels = [r.trial_id for r in records]
    vals = [_metric_value(r.val_metrics, metric) for r in records]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))
    ax.bar(labels, vals)
    ax.set_ylabel(f"val {metric}")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _write_study_summary(out_root: Path, all_records: List[TrialResult], args: argparse.Namespace) -> None:
    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows = [r.flat_row() for r in all_records]
    _save_csv(rows, summary_dir / "all_trials.csv")
    _save_json(
        {
            "ranking_metric": args.ranking_metric,
            "ranking_mode": args.ranking_mode,
            "n_trials": len(all_records),
            "records": rows,
        },
        summary_dir / "study_summary.json",
    )

    for phase_name in ("hparam", "scheduler", "early_stop"):
        phase_recs = [r for r in all_records if r.phase == phase_name]
        if phase_recs:
            _plot_phase_comparison(
                phase_recs,
                f"Phase {phase_name}: val {args.ranking_metric}",
                summary_dir / f"compare_{phase_name}_{args.ranking_metric}.png",
                args.ranking_metric,
            )


def main() -> None:
    args = _parse_args()
    if args.quick:
        args.hparam_trials = min(args.hparam_trials, 3)
        args.epochs = min(args.epochs, 5)
        args.max_train_samples = args.max_train_samples or 400
        args.max_val_samples = args.max_val_samples or 120
        args.max_test_samples = args.max_test_samples or 120
        args.early_stopping_patience = min(args.early_stopping_patience, 2)

    project_root = _PROJECT_ROOT
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_dir) if args.output_dir else (project_root / "runs" / f"hpo_mae2d_cnn_{now}")
    out_root.mkdir(parents=True, exist_ok=True)
    _save_json(vars(args), out_root / "study_args.json")

    print(f"[hpo_mae_2d_cnn] output: {out_root}")
    print(f"[hpo_mae_2d_cnn] phase={args.phase} epochs={args.epochs} quick={args.quick}")

    rng = random.Random(args.seed)
    all_records: List[TrialResult] = []
    best_hp: Dict[str, Any] = {}

    run_phases = list(PHASES[:-1]) if args.phase == "all" else [args.phase]

    if "hparam" in run_phases:
        recs, best = _phase_hparam(project_root, out_root, args, rng)
        all_records.extend(recs)
        if best:
            best_hp = {k: v for k, v in best.hparams.items() if k not in ("scheduler_type", "early_stopping")}

    if "scheduler" in run_phases:
        if not best_hp:
            prev = out_root / "phase1_hparam_search" / "best_hparams.json"
            if prev.is_file():
                best_hp = json.loads(prev.read_text()).get("hparams", {})
            else:
                raise RuntimeError("Phase scheduler requires phase1 best_hparams or run --phase all.")
        recs, best = _phase_scheduler(project_root, out_root, args, best_hp)
        all_records.extend(recs)
        if best:
            best_hp = dict(best.hparams)

    if "early_stop" in run_phases:
        if not best_hp:
            prev = out_root / "phase2_scheduler" / "best_scheduler.json"
            if prev.is_file():
                best_hp = json.loads(prev.read_text()).get("hparams", {})
            else:
                raise RuntimeError("Phase early_stop requires phase2 best_scheduler or run --phase all.")
        all_records.extend(_phase_early_stop(project_root, out_root, args, best_hp))

    _write_study_summary(out_root, all_records, args)
    print(f"[hpo_mae_2d_cnn] Done. Summary -> {out_root / 'summary'}")


if __name__ == "__main__":
    main()
