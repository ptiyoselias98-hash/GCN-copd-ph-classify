#!/usr/bin/env bash
# _run_phase2_train.sh -- Sprint 7 Task 5 Phase 2 full training.
# Usage: _run_phase2_train.sh <edge_drop_p>
set -euo pipefail

BEST_P="${1:-0.0}"

PROJ="/home/imss/cw/GCN copdnoph copdph"
SPRINT7="${PROJ}/sprint7"
CACHE="${SPRINT7}/cache_tri"
OUT="${SPRINT7}/outputs/tri_phase2"
FLAG="${OUT}/phase2_done.flag"

mkdir -p "${OUT}"
rm -f "${FLAG}"

source /home/imss/miniconda3/etc/profile.d/conda.sh
conda activate pulmonary_bv5_py39

cd "${SPRINT7}"

echo "===== sprint7 phase2 start $(date -Is) ====="
echo "cache=${CACHE}"
echo "out=${OUT}"
echo "edge_drop_p=${BEST_P}"

set +e
python -u tri_structure_pipeline.py \
    --cache_dir "${CACHE}" \
    --cache_format tri \
    --labels "${PROJ}/data/labels_gold.csv" \
    --mpap "${PROJ}/data/mpap_lookup_gold.json" \
    --output_dir "${OUT}" \
    --epochs 200 --repeats 3 \
    --edge_drop_p "${BEST_P}" \
    --label_smoothing 0.1 \
    --warmup_epochs 20 \
    --mpap_aux \
    --use_signature
RC=$?
set -e

touch "${FLAG}"
echo "===== sprint7 phase2 done $(date -Is)  rc=${RC}  (flag: ${FLAG}) ====="
exit ${RC}
