#!/bin/bash
#SBATCH --job-name=mae2d_final_m75
#SBATCH --output=slurm_err_out/mae2d_final_m75_%j.out
#SBATCH --error=slurm_err_out/mae2d_final_m75_%j.err
#SBATCH --partition=generic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/dsi/ossnat/VSD_FM/VSD_foundation_model

# Full MAE 2D training + temporal eval plots (HPO trial_017, mask_ratio 0.75).
#
# Submit:
#   mkdir -p slurm_err_out
#   sbatch run_2dcnn_final_train.sh
#
# ImageNet-pretrained ResNet18 comparison (same hparams otherwise):
#   PRETRAINED=1 sbatch run_2dcnn_final_train.sh
#
# Optional env vars:
#   EPOCHS=100
#   BATCH_SIZE=128
#   RUN_DIR=/path/to/runs/…
#   MONKEYS="gandalf"   — restrict monkeys (default: all in v3 split)

set -euo pipefail

REPO="${REPO:-/home/dsi/ossnat/VSD_FM/VSD_foundation_model}"
CONFIG="${CONFIG:-configs/MAE_2D_best_mask75.yaml}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
PRETRAINED="${PRETRAINED:-0}"

JOB_TAG="${SLURM_JOB_ID:-local}"
PRETRAIN_TAG="scratch"
if [[ "${PRETRAINED}" == "1" ]]; then
  PRETRAIN_TAG="pretrained"
fi
RUN_DIR="${RUN_DIR:-${REPO}/runs/mae2d_final_mask75_${PRETRAIN_TAG}_${JOB_TAG}}"
CKPT_DIR="${CKPT_DIR:-${RUN_DIR}/ckpt}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/logs}"

mkdir -p "${REPO}/slurm_err_out"
cd "${REPO}"

# shellcheck disable=SC1091
source "${REPO}/scripts/activate_cluster_env.sh"

EXTRA_ARGS=(
  --config "${CONFIG}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --frame-start 32
  --frame-end 52
  --val-frame-stride 1
  --mask-ratio 0.75
  --ckpt-dir "${CKPT_DIR}"
  --log-dir "${LOG_DIR}"
  --results-dir "${RUN_DIR}"
  --train-num-workers 4
  --val-num-workers 2
  --test-num-workers 2
)

if [[ "${PRETRAINED}" == "1" ]]; then
  EXTRA_ARGS+=(--pretrained)
else
  EXTRA_ARGS+=(--no-pretrained)
fi

if [[ -n "${MONKEYS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS+=(--monkeys ${MONKEYS})
fi

echo "=== MAE 2D full train (mask 0.75, trial_017 hparams) ==="
echo "Host:       $(hostname)"
echo "Date:       $(date)"
echo "Repo:       ${REPO}"
echo "Config:     ${CONFIG}"
echo "Pretrained: ${PRETRAINED}"
echo "Run dir:    ${RUN_DIR}"
echo "Epochs:     ${EPOCHS}"
echo "Batch:      ${BATCH_SIZE}"
python --version
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo ""

python scripts/train_mae_2d.py "${EXTRA_ARGS[@]}"

echo ""
echo "=== Done ==="
echo "Checkpoints:   ${CKPT_DIR}"
echo "Config used:   ${CKPT_DIR}/analysis/config_used.json"
echo "Val metrics:   ${CKPT_DIR}/analysis/metrics/metrics_val.json"
echo "Test metrics:  ${CKPT_DIR}/analysis/metrics/metrics_test.json"
echo "Loss plot:     ${CKPT_DIR}/analysis/train_val_loss_vs_epoch.png"
echo "Temporal eval: ${RUN_DIR}/temporal_eval/"
