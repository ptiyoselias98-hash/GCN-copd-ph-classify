#!/usr/bin/env bash
# _run_cluster_pipeline.sh — server-side runner for the unsupervised
# vessel-topology phenotype clustering experiment.
#
# Usage:
#   bash _run_cluster_pipeline.sh
#
# Reads:
#   cache/, data/labels_gold.csv, data/mpap_lookup_gold.json
# Writes:
#   outputs/cluster_topology/{cluster_results.json,
#                            cluster_assignments.csv,
#                            umap_topology.png, umap_vascular_full.png,
#                            cluster_pipeline_done.flag}
set -euo pipefail

PROJ="/home/imss/cw/GCN copdnoph copdph"
OUT="${PROJ}/outputs/cluster_topology"
FLAG="${OUT}/cluster_pipeline_done.flag"

mkdir -p "${OUT}"
rm -f "${FLAG}"

source /home/imss/miniconda3/etc/profile.d/conda.sh
conda activate pulmonary_bv5_py39

cd "${PROJ}"

echo "===== cluster pipeline start $(date -Is) ====="
echo "project=${PROJ}"
echo "output=${OUT}"

# Ensure umap/hdbscan present (silently install if missing — user has -y by default).
python -c "import umap, hdbscan" 2>/dev/null || \
    pip install --quiet umap-learn hdbscan || true

python -u cluster_vessel_topology.py \
    --cache_dir "${PROJ}/cache" \
    --labels    "${PROJ}/data/labels_gold.csv" \
    --mpap      "${PROJ}/data/mpap_lookup_gold.json" \
    --output_dir "${OUT}"

touch "${FLAG}"
echo "===== cluster pipeline done $(date -Is)  (flag: ${FLAG}) ====="
