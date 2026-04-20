#!/usr/bin/env bash
# _run_rebuild_cache_tri.sh -- Sprint 7 Task 1 server chain.
#
# Rebuilds per-structure cache_tri/ from raw masks with:
#   * largest-component filter (Sprint 7 Step 2)
#   * airway <3-node fallback (Sprint 7 Step 7)
#   * Strahler default 1 (Sprint 7 Step 6)
#
# Usage:
#   bash _run_rebuild_cache_tri.sh [raw_dir] [workers] [overwrite]
#     raw_dir    default: /home/imss/cw/COPDnonPH COPD-PH /data/nii
#     workers    default: 8
#     overwrite  default: 1 (Sprint 7 needs fresh cache; set 0 to skip existing)
set -euo pipefail

RAW_DIR="${1:-/home/imss/cw/COPDnonPH COPD-PH /data/nii}"
WORKERS="${2:-8}"
OVERWRITE="${3:-1}"

PROJ="/home/imss/cw/GCN copdnoph copdph"
SPRINT7="${PROJ}/sprint7"
CACHE="${SPRINT7}/cache_tri"
FLAG="${CACHE}/rebuild_cache_tri_done.flag"

mkdir -p "${CACHE}"
rm -f "${FLAG}"

source /home/imss/miniconda3/etc/profile.d/conda.sh
conda activate pulmonary_bv5_py39

cd "${SPRINT7}"

echo "===== sprint7 rebuild start $(date -Is) ====="
echo "raw_dir=${RAW_DIR}"
echo "labels=${PROJ}/data/labels_gold.csv"
echo "output_cache=${CACHE}"
echo "workers=${WORKERS}  overwrite=${OVERWRITE}"

OVERWRITE_FLAG=""
if [[ "${OVERWRITE}" == "1" ]]; then
    OVERWRITE_FLAG="--overwrite"
fi

python -u rebuild_cache_tri.py \
    --data_dir "${RAW_DIR}" \
    --labels   "${PROJ}/data/labels_gold.csv" \
    --output_cache "${CACHE}" \
    --workers "${WORKERS}" \
    ${OVERWRITE_FLAG}

touch "${FLAG}"
echo "===== sprint7 rebuild done $(date -Is)  (flag: ${FLAG}) ====="
