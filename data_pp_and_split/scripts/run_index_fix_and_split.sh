#!/bin/bash
# Rebuild trial index from H5 metadata and regenerate v3 split (cluster / full Data/).
#
# Run from VSD_foundation_model with all ProcessedData H5s present:
#   cd /path/to/VSD_foundation_model
#   bash data_pp_and_split/scripts/run_index_fix_and_split.sh
#
# Requires every session listed in all_trials_index.csv (or backup) to exist on disk.

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "${REPO}"

export PYTHONPATH=.

echo "=== Step 1: rebuild all_trials_index.csv from H5 trial_metadata_json ==="
python data_pp_and_split/scripts/rebuild_index.py

echo ""
echo "=== Step 2: validate index ==="
python data_pp_and_split/scripts/validate_index.py

echo ""
echo "=== Step 3: v3 session×condition split + baseline stats ==="
python data_pp_and_split/scripts/run_split.py

echo ""
echo "Done. Artifacts:"
echo "  Data/FoundationData/ProcessedData/splits/all_trials_index.csv"
echo "  Data/FoundationData/ProcessedData/splits/split_v3_seed17_session_condition_group.csv"
echo "  Data/FoundationData/ProcessedData/splits/baseline_stats_v3_seed17_session_condition_group.{json,h5}"
