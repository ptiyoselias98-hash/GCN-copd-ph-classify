#!/usr/bin/env bash
# _run_tri_pipeline.sh — server-side chain for the tri-structure GCN pipeline.
#
# Usage:
#   bash _run_tri_pipeline.sh <phase> <epochs> <repeats> <n_folds>
#     phase   1 = existing unified cache + heuristic partition (Phase 1)
#             2 = rebuilt per-structure cache (Phase 2, requires rebuild first)
#     epochs  default 200
#     repeats default 3   (repeated CV, same convention as followup_automation)
#     n_folds default 5
#
# Exit cleanly => touch tri_phase<N>_done.flag under output dir.
set -euo pipefail

PHASE="${1:-1}"
EPOCHS="${2:-200}"
REPEATS="${3:-3}"
N_FOLDS="${4:-5}"

PROJ="/home/imss/cw/GCN copdnoph copdph"
TRI="${PROJ}/tri_structure"
OUT="${PROJ}/outputs/tri_phase${PHASE}"
FLAG="${OUT}/tri_phase${PHASE}_done.flag"

if [[ "${PHASE}" == "1" ]]; then
    CACHE="${PROJ}/cache"
elif [[ "${PHASE}" == "2" ]]; then
    CACHE="${TRI}/cache_tri"
else
    echo "ERROR: phase must be 1 or 2 (got '${PHASE}')" >&2
    exit 2
fi

if [[ ! -d "${CACHE}" ]]; then
    echo "ERROR: cache dir missing: ${CACHE}" >&2
    echo "  Phase 2 requires prior run of _run_tri_rebuild_cache.sh" >&2
    exit 3
fi

mkdir -p "${OUT}"
rm -f "${FLAG}"

source /home/imss/miniconda3/etc/profile.d/conda.sh
conda activate pulmonary_bv5_py39

cd "${TRI}"

echo "===== tri pipeline start $(date -Is) ====="
echo "phase=${PHASE} epochs=${EPOCHS} repeats=${REPEATS} n_folds=${N_FOLDS}"
echo "cache=${CACHE}"
echo "labels=${PROJ}/data/labels_gold.csv"
echo "mpap=${PROJ}/data/mpap_lookup_gold.json"
echo "output=${OUT}"

python -u tri_structure_pipeline.py \
    --cache_dir "${CACHE}" \
    --labels "${PROJ}/data/labels_gold.csv" \
    --mpap "${PROJ}/data/mpap_lookup_gold.json" \
    --output_dir "${OUT}" \
    --epochs "${EPOCHS}" \
    --repeats "${REPEATS}" \
    --n_folds "${N_FOLDS}" \
    --mpap_aux

touch "${FLAG}"
echo "===== tri pipeline done $(date -Is)  (flag: ${FLAG}) ====="
