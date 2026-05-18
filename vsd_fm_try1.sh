#!/bin/bash
#SBATCH --job-name=vsd_fm_env
#SBATCH --output=vsd_fm_env_%j.out
#SBATCH --error=vsd_fm_env_%j.err
#SBATCH --partition=cpu192G-48h
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --chdir=/home/lab/ossnat/Experiments

set -euo pipefail

# Path to your cloned repo (adjust if the clone lives elsewhere)
REPO="${REPO:-/home/lab/ossnat/Experiments/VSD_foundation_model}"

cd "${REPO}"
source .venv/bin/activate

echo "=== VSD FM env check ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Repo: ${REPO}"
echo "Python: $(which python)"
python --version
echo ""

# CPU smoke test (no data files). Use --device cuda on a GPU partition instead.
python scripts/check_env.py --device cpu

echo ""
echo "=== Done ==="
