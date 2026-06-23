#!/bin/bash
#SBATCH --job-name=vsd_fm_test
#SBATCH --output=slurm_err_out/vsd_fm_test_%j.out
#SBATCH --error=slurm_err_out/vsd_fm_test_%j.err
#SBATCH --partition=generic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/dsi/ossnat/VSD_FM/VSD_foundation_model

# Cluster smoke test: paths, Python env (venv OR conda), CUDA, imports, optional data check.
#
# Submit:
#   mkdir -p slurm_err_out
#   sbatch vsd_fm_cluster_test.sh
#
# Login node (no GPU job):
#   ENV_KIND=conda bash vsd_fm_cluster_test.sh
#
# Env vars:
#   REPO          — repo root (default: cluster path below)
#   ENV_KIND      — venv | conda (default: venv)
#   CONDA_ENV     — conda env name when ENV_KIND=conda (default: vsd_conda_env)
#   WITH_DATA=1   — also verify v3 split files + gandalf H5 dir + pytest --with-data
#   MINI_TRAIN=1  — run 1 epoch, gandalf only, tiny batch (slow; off by default)

set -euo pipefail

REPO="${REPO:-/home/dsi/ossnat/VSD_FM/VSD_foundation_model}"
WORKSPACE="${WORKSPACE:-$(dirname "${REPO}")}"
DATA_ROOT="${DATA_ROOT:-${WORKSPACE}/Data}"
ENV_KIND="${ENV_KIND:-venv}"
CONDA_ENV="${CONDA_ENV:-vsd_conda_env}"

SPLIT_CSV="${SPLIT_CSV:-${DATA_ROOT}/FoundationData/ProcessedData/splits/split_v3_seed17_session_condition_group.csv}"
STATS_JSON="${STATS_JSON:-${DATA_ROOT}/FoundationData/ProcessedData/splits/baseline_stats_v3_seed17_session_condition_group.json}"
GANDALF_DIR="${GANDALF_DIR:-${DATA_ROOT}/FoundationData/ProcessedData/gandalf}"

mkdir -p "${REPO}/slurm_err_out"
cd "${REPO}"

echo "=== VSD FM cluster test ==="
echo "Host:      $(hostname)"
echo "Date:      $(date)"
echo "Repo:      ${REPO}"
echo "Workspace: ${WORKSPACE}"
echo "Data:      ${DATA_ROOT}"
echo "ENV_KIND:  ${ENV_KIND}"
[[ "${ENV_KIND}" == "conda" ]] && echo "CONDA_ENV: ${CONDA_ENV}"
echo ""

FAIL=0
_check() {
  local label="$1"
  local path="$2"
  if [[ -e "${path}" ]]; then
    echo "  OK   ${label}: ${path}"
  else
    echo "  FAIL ${label}: ${path}" >&2
    FAIL=1
  fi
}

echo "[1/6] Repo layout"
_check "repo root" "${REPO}"
_check "train script" "${REPO}/scripts/train_mae_2d.py"
_check "MAE config" "${REPO}/configs/MAE_2D_full.yaml"
_check "env helper" "${REPO}/scripts/activate_cluster_env.sh"
echo ""

echo "[2/6] Python environment"
# shellcheck disable=SC1091
source "${REPO}/scripts/activate_cluster_env.sh"
echo "  Python: $(which python)"
python --version
echo ""

echo "[3/6] CUDA + core imports"
if python -c "
import torch
print('  torch:', torch.__version__)
cuda = torch.cuda.is_available()
print('  cuda available:', cuda)
if cuda:
    print('  device:', torch.cuda.get_device_name(0))
"; then
  :
else
  echo "  FAIL torch/CUDA check" >&2
  FAIL=1
fi
python "${REPO}/scripts/check_env.py" --device cuda
echo ""

echo "[4/6] Data paths"
_check "split v3 CSV" "${SPLIT_CSV}"
_check "baseline stats JSON" "${STATS_JSON}"
_check "gandalf processed dir" "${GANDALF_DIR}"
if [[ -d "${GANDALF_DIR}" ]]; then
  n_h5="$(find "${GANDALF_DIR}" -maxdepth 2 -name '*.h5' 2>/dev/null | wc -l | tr -d ' ')"
  echo "  info gandalf .h5 files (depth<=2): ${n_h5}"
fi
echo ""

if [[ "${WITH_DATA:-0}" == "1" ]]; then
  echo "[5/6] Pytest with data"
  python "${REPO}/scripts/sanity_check.py" --device cuda --with-data --skip-env || FAIL=1
else
  echo "[5/6] Pytest (no H5 required; set WITH_DATA=1 to include data tests)"
  python "${REPO}/scripts/sanity_check.py" --device cuda --skip-env || FAIL=1
fi
echo ""

if [[ "${MINI_TRAIN:-0}" == "1" ]]; then
  echo "[6/6] Mini train (1 epoch, gandalf, batch 8)"
  JOB_TAG="${SLURM_JOB_ID:-local}"
  RUN_DIR="${REPO}/runs/cluster_test_train_${JOB_TAG}"
  python "${REPO}/scripts/train_mae_2d.py" \
    --config configs/MAE_2D_full.yaml \
    --monkeys gandalf \
    --epochs 1 \
    --batch-size 8 \
    --frame-end 40 \
    --no-preload-into-ram \
    --no-auto-resume \
    --ckpt-dir "${RUN_DIR}/ckpt" \
    --log-dir "${RUN_DIR}/logs" \
    --results-dir "${RUN_DIR}" \
    --train-num-workers 2 \
    --val-num-workers 1 \
    --test-num-workers 1
  echo "  mini train output: ${RUN_DIR}"
else
  echo "[6/6] Mini train skipped (set MINI_TRAIN=1 to run 1 epoch)"
fi
echo ""

if [[ "${FAIL}" -ne 0 ]]; then
  echo "=== CLUSTER TEST FAILED (see FAIL lines above) ===" >&2
  exit 1
fi

echo "=== CLUSTER TEST PASSED ==="
echo "Next: sbatch vsd_fm_try_train_2d.sh"
echo "  or: ENV_KIND=conda sbatch vsd_fm_try_train_2d.sh"
