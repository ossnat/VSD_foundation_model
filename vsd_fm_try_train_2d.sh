#!/bin/bash
#SBATCH --job-name=vsd_mae2d_gandalf
#SBATCH --output=slurm_err_out/vsd_mae2d_gandalf_%j.out
#SBATCH --error=slurm_err_out/vsd_mae2d_gandalf_%j.err
#SBATCH --partition=generic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --chdir=/home/dsi/ossnat/VSD_FM/VSD_foundation_model

# Short end-to-end MAE 2D training on gandalf (generic GPU partition).
# Layout: <workspace>/Data/... sibling of this repo (see src/utils/data_paths.py).
#
# Submit from repo root:
#   mkdir -p slurm_err_out
#   sbatch vsd_fm_try_train_2d.sh
#
# Override paths or run length, e.g.:
#   REPO=/path/to/VSD_foundation_model EPOCHS=3 sbatch vsd_fm_try_train_2d.sh
#   ENV_KIND=conda sbatch vsd_fm_try_train_2d.sh

set -euo pipefail

REPO="${REPO:-/home/dsi/ossnat/VSD_FM/VSD_foundation_model}"
WORKSPACE="${WORKSPACE:-$(dirname "${REPO}")}"
DATA_ROOT="${DATA_ROOT:-${WORKSPACE}/Data}"
GANDALF_DIR="${GANDALF_DIR:-${DATA_ROOT}/FoundationData/ProcessedData/gandalf}"

ENV_KIND="${ENV_KIND:-venv}"
CONDA_ENV="${CONDA_ENV:-vsd_conda_env}"

EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-16}"
FRAME_END="${FRAME_END:-50}"

JOB_TAG="${SLURM_JOB_ID:-local}"
RUN_DIR="${RUN_DIR:-${REPO}/runs/mae2d_gandalf_try_${JOB_TAG}}"
CKPT_DIR="${CKPT_DIR:-${RUN_DIR}/ckpt}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/logs}"

mkdir -p "${REPO}/slurm_err_out"
cd "${REPO}"

# shellcheck disable=SC1091
source "${REPO}/scripts/activate_cluster_env.sh"

echo "=== VSD FM MAE 2D train (gandalf) ==="
echo "Host:     $(hostname)"
echo "Date:     $(date)"
echo "Job ID:   ${JOB_TAG}"
echo "Repo:     ${REPO}"
echo "Workspace:${WORKSPACE}"
echo "Data:     ${DATA_ROOT}"
echo "Gandalf:  ${GANDALF_DIR}"
echo "Run dir:  ${RUN_DIR}"
echo "ENV_KIND: ${ENV_KIND}"
echo "Python:   $(which python)"
python --version
echo "CUDA:     $(python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a")')"
echo ""

if [[ ! -d "${GANDALF_DIR}" ]]; then
  echo "ERROR: gandalf data directory not found: ${GANDALF_DIR}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}" "${CKPT_DIR}" "${LOG_DIR}"

python scripts/train_mae_2d.py \
  --config configs/MAE_2D_full.yaml \
  --monkeys gandalf \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --frame-end "${FRAME_END}" \
  --no-preload-into-ram \
  --no-auto-resume \
  --ckpt-dir "${CKPT_DIR}" \
  --log-dir "${LOG_DIR}" \
  --results-dir "${RUN_DIR}" \
  --train-num-workers 4 \
  --val-num-workers 2 \
  --test-num-workers 2

echo ""
echo "=== Training finished ==="
echo "Checkpoints & analysis: ${CKPT_DIR}"
echo "  config:        ${CKPT_DIR}/analysis/config_used.json"
echo "  metrics:       ${CKPT_DIR}/analysis/metrics/"
echo "  loss plot:     ${CKPT_DIR}/analysis/train_val_loss_vs_epoch.png"
echo "TensorBoard logs:       ${LOG_DIR}"
echo "Temporal eval / recon:  ${RUN_DIR}/temporal_eval/"
echo "=== Done ==="
