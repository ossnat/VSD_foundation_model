#!/bin/bash
#SBATCH --job-name=vsd_fm_sanity
#SBATCH --output=vsd_fm_sanity_%j.out
#SBATCH --error=vsd_fm_sanity_%j.err
#SBATCH --partition=cpu192G-48h
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --chdir=/home/dsi/ossnat/VSD_FM/VSD_foundation_model

set -euo pipefail

REPO="${REPO:-/home/dsi/ossnat/VSD_FM/VSD_foundation_model}"

cd "${REPO}"
source .venv/bin/activate

echo "=== VSD FM sanity check ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Repo: ${REPO}"
echo "Python: $(which python)"
python --version
echo ""

python scripts/sanity_check.py --device cpu

echo ""
echo "=== Done ==="
