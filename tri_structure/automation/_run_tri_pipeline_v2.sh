#!/usr/bin/env bash
# _run_tri_pipeline_v2.sh — v2 server chain: attention pooling + graph signatures.
#
# Architectural changes vs v1:
#   * StructureEncoder uses GlobalAttention pooling (--pool_mode attn) so
#     high-information nodes dominate the structure embedding.
#   * analyse_shared_embeddings runs THREE clustering views (embedding,
#     signature, hybrid) via --use_signature.
#
# Only Phase 2 (rebuilt per-structure cache) is supported here; the signature
# features require the per-structure Data objects produced by
# _run_tri_rebuild_cache.sh.
#
# Usage:
#   bash _run_tri_pipeline_v2.sh <epochs> <repeats> <n_folds>
#     epochs  default 200
#     repeats default 3
#     n_folds default 5
set -euo pipefail

EPOCHS="${1:-200}"
REPEATS="${2:-3}"
N_FOLDS="${3:-5}"

PROJ="/home/imss/cw/GCN copdnoph copdph"
TRI="${PROJ}/tri_structure"
CACHE="${TRI}/cache_tri"
OUT="${PROJ}/outputs/tri_phase2_v2"
FLAG="${OUT}/tri_phase2_v2_done.flag"

if [[ ! -d "${CACHE}" ]]; then
    echo "ERROR: cache dir missing: ${CACHE}" >&2
    echo "  v2 requires prior run of _run_tri_rebuild_cache.sh" >&2
    exit 3
fi

mkdir -p "${OUT}"
rm -f "${FLAG}"

source /home/imss/miniconda3/etc/profile.d/conda.sh
conda activate pulmonary_bv5_py39

cd "${TRI}"

echo "===== tri pipeline v2 start $(date -Is) ====="
echo "epochs=${EPOCHS} repeats=${REPEATS} n_folds=${N_FOLDS}"
echo "cache=${CACHE}"
echo "labels=${PROJ}/data/labels_gold.csv"
echo "mpap=${PROJ}/data/mpap_lookup_gold.json"
echo "output=${OUT}"
echo "pool_mode=attn  use_signature=true"

python -u tri_structure_pipeline.py \
    --cache_dir "${CACHE}" \
    --labels "${PROJ}/data/labels_gold.csv" \
    --mpap "${PROJ}/data/mpap_lookup_gold.json" \
    --output_dir "${OUT}" \
    --epochs "${EPOCHS}" \
    --repeats "${REPEATS}" \
    --n_folds "${N_FOLDS}" \
    --mpap_aux \
    --pool_mode attn \
    --use_signature

touch "${FLAG}"
echo "===== tri pipeline v2 done $(date -Is)  (flag: ${FLAG}) ====="
