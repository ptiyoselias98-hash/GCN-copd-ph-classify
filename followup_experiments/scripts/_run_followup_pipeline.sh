#!/usr/bin/env bash
# _run_followup_pipeline.sh — server-side chain for follow-up runs.
#
# Usage:
#   bash _run_followup_pipeline.sh <variant> <epochs> <n_folds>
#     variant ∈ {short, medium, medium_youden, medium_youden_rep,
#                mode_gcn, mode_hybrid, mode_radiomics}
#
# Assumes:
#   - auto_run_followup.py has pushed train_plus.py, utils/training_plus.py,
#     and (for short/both) labels_gold.csv / splits_gold.json / mpap_lookup_gold.json
#     into the standard locations under /home/imss/cw/GCN copdnoph copdph/.
#   - The conda env `pulmonary_bv5_py39` exists.
#
# Exit cleanly => touch followup_<variant>_done.flag under outputs dir.
set -euo pipefail

VARIANT="${1:-short}"
EPOCHS="${2:-200}"
N_FOLDS="${3:-5}"

PROJ="/home/imss/cw/GCN copdnoph copdph"
OUT="${PROJ}/outputs/followup_${VARIANT}"
FLAG="${OUT}/followup_${VARIANT}_done.flag"

mkdir -p "${OUT}"
rm -f "${FLAG}"

# Conda activation
source /home/imss/miniconda3/etc/profile.d/conda.sh
conda activate pulmonary_bv5_py39

cd "${PROJ}"

echo "===== follow-up pipeline start $(date -Is) ====="
echo "variant=${VARIANT} epochs=${EPOCHS} n_folds=${N_FOLDS}"
echo "project=${PROJ}"
echo "output=${OUT}"

python -u train_plus.py \
    --variant "${VARIANT}" \
    --cache_dir "${PROJ}/cache" \
    --output_dir "${OUT}" \
    --epochs "${EPOCHS}" \
    --n_folds "${N_FOLDS}"

touch "${FLAG}"
echo "===== follow-up pipeline done $(date -Is)  (flag: ${FLAG}) ====="
