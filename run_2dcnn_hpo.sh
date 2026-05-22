#!/bin/bash
#SBATCH --job-name=mae2d_hpo
#SBATCH --output=mae2d_hpo_%j.out
#SBATCH --error=mae2d_hpo_%j.err
#SBATCH --partition=generic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/lab/ossnat/VSD_FM/VSD_foundation_model

# MAE 2D CNN 3-phase HPO (hparam search -> scheduler ablation -> early-stop ablation).
#
# Submit:
#   sbatch run_2dcnn_hpo.sh
#
# Quick smoke on login node (no sbatch):
#   QUICK=1 bash run_2dcnn_hpo.sh

set -euo pipefail

REPO="${REPO:-/home/lab/ossnat/VSD_FM/VSD_foundation_model}"
JOB_TAG="${SLURM_JOB_ID:-local}"
OUT_DIR="${OUT_DIR:-${REPO}/runs/hpo_mae2d_cnn_${JOB_TAG}}"

cd "${REPO}"
source .venv/bin/activate
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"

EXTRA_ARGS=()
if [[ "${QUICK:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--quick --phase all)
  echo "=== QUICK mode: 3 hparam trials, 5 epochs, capped samples ==="
else
  EXTRA_ARGS+=(--phase all --hparam-trials 8 --epochs 15)
fi

# Optional: restrict to one monkey for faster cluster runs
if [[ -n "${MONKEYS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS+=(--monkeys ${MONKEYS})
fi

echo "=== MAE 2D CNN HPO ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Repo: ${REPO}"
echo "Out:  ${OUT_DIR}"
python --version
python -c "import torch; print('cuda', torch.cuda.is_available())"
echo ""

python scripts/hpo_mae_2d_cnn.py \
  --output-dir "${OUT_DIR}" \
  --ranking-metric ssim_masked \
  --ranking-mode max \
  "${EXTRA_ARGS[@]}"

echo ""
echo "=== Done ==="
echo "Summary: ${OUT_DIR}/summary/all_trials.csv"
echo "Pearson plots: ${OUT_DIR}/phase*/**/eval/temporal_eval/temporal_metrics_*_pearson_flat.png"
