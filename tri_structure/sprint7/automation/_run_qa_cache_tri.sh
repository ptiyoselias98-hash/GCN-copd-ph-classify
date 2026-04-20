#!/usr/bin/env bash
# _run_qa_cache_tri.sh -- Sprint 7 Task 2 QA gate.
#
# Exits 0 if QA passes, 1 if FAIL. Also writes qa_summary.txt + qa_report.json
# + two PNGs into outputs/sprint7_qa/.
set -euo pipefail

PROJ="/home/imss/cw/GCN copdnoph copdph"
SPRINT7="${PROJ}/sprint7"
CACHE="${SPRINT7}/cache_tri"
OUT="${PROJ}/outputs/sprint7_qa"
FLAG="${OUT}/qa_done.flag"

mkdir -p "${OUT}"
rm -f "${FLAG}"

source /home/imss/miniconda3/etc/profile.d/conda.sh
conda activate pulmonary_bv5_py39

cd "${SPRINT7}"

echo "===== sprint7 qa start $(date -Is) ====="
echo "cache=${CACHE}"
echo "labels=${PROJ}/data/labels_gold.csv"
echo "output=${OUT}"

# Cross-cache consistency check uses the Phase 1 unified cache if present.
OLD_CACHE="${PROJ}/cache"
OLD_ARG=()
if [[ -d "${OLD_CACHE}" ]]; then
    OLD_ARG=(--old_unified_cache "${OLD_CACHE}")
fi

# Non-zero exit from qa_cache_tri.py should still allow the flag+log to be
# readable from the orchestrator's fetch step.
set +e
python -u qa_cache_tri.py \
    --cache_dir "${CACHE}" \
    --labels "${PROJ}/data/labels_gold.csv" \
    --output_dir "${OUT}" \
    "${OLD_ARG[@]}"
RC=$?
set -e

touch "${FLAG}"
echo "===== sprint7 qa done $(date -Is)  rc=${RC}  (flag: ${FLAG}) ====="
exit ${RC}
