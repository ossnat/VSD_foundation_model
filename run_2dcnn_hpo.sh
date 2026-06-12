#!/bin/bash
#SBATCH --job-name=mae2d_hpo_v3
#SBATCH --output=mae2d_hpo_v3_%j.out
#SBATCH --error=mae2d_hpo_v3_%j.err
#SBATCH --partition=generic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/lab/ossnat/VSD_FM/VSD_foundation_model

# MAE 2D CNN HPO on v3 session×condition group split.
#
# Phase 1: random search (loss incl. l1_ssim, mask, patch, crop, batch+lr)
# Phase 2: scheduler ablation (none vs cosine+warmup)
# Phase 3: early stopping ablation
#
# Submit:
#   sbatch run_2dcnn_hpo.sh
#
# Quick smoke on login node (no sbatch):
#   QUICK=1 bash run_2dcnn_hpo.sh
#
# Optional env vars:
#   MONKEYS="frodo"          — restrict to one monkey (faster debug)
#   HPARAM_TRIALS=24         — phase-1 trial count
#   EPOCHS=12                — epochs per trial
#   OUT_DIR=/path/to/runs/…  — custom output directory

set -euo pipefail

REPO="${REPO:-/home/lab/ossnat/VSD_FM/VSD_foundation_model}"
JOB_TAG="${SLURM_JOB_ID:-local}"
OUT_DIR="${OUT_DIR:-${REPO}/runs/hpo_mae2d_v3_${JOB_TAG}}"
CONFIG="${CONFIG:-configs/MAE_2D_hpo.yaml}"
HPARAM_TRIALS="${HPARAM_TRIALS:-24}"
EPOCHS="${EPOCHS:-12}"

cd "${REPO}"
source .venv/bin/activate
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"

EXTRA_ARGS=(
  --config "${CONFIG}"
  --output-dir "${OUT_DIR}"
  --ranking-metric ssim_masked
  --ranking-mode max
  --val-frame-stride 3
)

if [[ "${QUICK:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--quick --phase all)
  echo "=== QUICK mode: 3 hparam trials, 3 epochs, capped samples ==="
else
  EXTRA_ARGS+=(--phase all --hparam-trials "${HPARAM_TRIALS}" --epochs "${EPOCHS}")
  echo "=== FULL HPO: ${HPARAM_TRIALS} hparam trials, ${EPOCHS} epochs/trial ==="
fi

if [[ -n "${MONKEYS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS+=(--monkeys ${MONKEYS})
fi

echo "=== MAE 2D CNN HPO (v3 split) ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Repo:   ${REPO}"
echo "Config: ${CONFIG}"
echo "Out:    ${OUT_DIR}"
python --version
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo ""

python scripts/hpo_mae_2d_cnn.py "${EXTRA_ARGS[@]}"

echo ""
echo "=== Done ==="
echo "Summary:       ${OUT_DIR}/summary/all_trials.csv"
echo "Best hparams:  ${OUT_DIR}/phase1_hparam_search/best_hparams.json"
echo "Best schedule: ${OUT_DIR}/phase2_scheduler/best_scheduler.json"
echo "Best ES:       ${OUT_DIR}/phase3_early_stop/best_early_stop.json"
