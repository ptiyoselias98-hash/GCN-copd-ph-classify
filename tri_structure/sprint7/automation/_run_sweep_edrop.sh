#!/usr/bin/env bash
# _run_sweep_edrop.sh -- Sprint 7 Task 3 regularisation sweep.
#
# Runs tri_structure_pipeline for p in {0.0, 0.05, 0.10, 0.15}, each with
# 1 repeat x 5 folds x 200 epochs. Sequential so only one fold trains at
# a time on the 3090. Writes outputs into sprint7/outputs/sweep_edrop_p<p>/.
set -euo pipefail

PROJ="/home/imss/cw/GCN copdnoph copdph"
SPRINT7="${PROJ}/sprint7"
CACHE="${SPRINT7}/cache_tri"
OUT_BASE="${SPRINT7}/outputs/sweep_edrop"
FLAG="${SPRINT7}/outputs/sweep_edrop_done.flag"

mkdir -p "${OUT_BASE}"
rm -f "${FLAG}"

source /home/imss/miniconda3/etc/profile.d/conda.sh
conda activate pulmonary_bv5_py39

cd "${SPRINT7}"

echo "===== sprint7 sweep start $(date -Is) ====="
echo "cache=${CACHE}"
echo "out_base=${OUT_BASE}"

RC_TOTAL=0
for p in 0.0 0.05 0.10 0.15; do
    RUN_DIR="${OUT_BASE}_p${p}"
    echo ""
    echo "----- sweep run p=${p}  out=${RUN_DIR}  $(date -Is) -----"
    set +e
    python -u tri_structure_pipeline.py \
        --cache_dir "${CACHE}" \
        --cache_format tri \
        --labels "${PROJ}/data/labels_gold.csv" \
        --mpap "${PROJ}/data/mpap_lookup_gold.json" \
        --output_dir "${RUN_DIR}" \
        --edge_drop_p ${p} \
        --repeats 1 --epochs 200 \
        --mpap_aux
    RC=$?
    set -e
    if [[ ${RC} -ne 0 ]]; then
        RC_TOTAL=${RC}
        echo "!!! sweep p=${p} failed rc=${RC}"
    fi
done

touch "${FLAG}"
echo "===== sprint7 sweep done $(date -Is)  rc=${RC_TOTAL}  (flag: ${FLAG}) ====="
exit ${RC_TOTAL}
