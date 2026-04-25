#!/bin/bash
# R20.G — Post-segmentation chain on remote.
#
# Triggered after R20.F Simple_AV_seg completes on legacy 201 contrast cases.
# Builds v2 cache on the unified pipeline (legacy 201 + new100 plain-scan)
# and re-extracts per-structure morphometrics. Verifies R18.B legacy
# Spearman ρ for artery_len_p25 reproduces within the unified-pipeline
# cohort (positive verdict closes R18 must-fix #2).
#
# Run on remote: bash /tmp/_R20G_unified_pipeline_chain.sh > /tmp/r20g.log 2>&1 &
set -e
source /home/imss/miniconda3/etc/profile.d/conda.sh
conda activate pulmonary_bv5_py39 || conda activate HiPaS

ROOT_REMOTE="/home/imss/cw/GCN copdnoph copdph"
SAVSEG_201="$ROOT_REMOTE/nii-unified-201-savseg"
NEW100="$ROOT_REMOTE/nii-new100"
CACHE_OUT="$ROOT_REMOTE/cache_tri_v2_unified301"
mkdir -p "$CACHE_OUT"

echo "[R20.G] verify segmentation outputs"
N_201=$(ls "$SAVSEG_201" 2>/dev/null | wc -l)
N_NEW100=$(ls "$NEW100" 2>/dev/null | wc -l)
echo "  201_savseg = $N_201 case dirs (expect 201)"
echo "  new100 = $N_NEW100 case dirs (expect 100)"
if [ "$N_201" -lt 195 ]; then
  echo "[abort] Simple_AV_seg incomplete on legacy 201 ($N_201/201). Wait."
  exit 1
fi

echo "[R20.G] cache builder (binary mask path + edge-attr v2 schema)"
# This step uses the R19.C patched cache builder which auto-detects binary
# masks (max≤1.5, min≥-0.5) vs HU-sentinel. Both savseg dirs are binary.
python "$ROOT_REMOTE/scripts/evolution/R19C_build_v2_patcher.py" \
    --in_a "$SAVSEG_201" --in_b "$NEW100" --out "$CACHE_OUT" \
    --workers 6 2>&1 | tail -10 || echo "[fallback] script not on remote yet — needs scp"

echo "[R20.G] R17 morphometrics on unified pipeline"
python "$ROOT_REMOTE/scripts/evolution/R19D_extended_morphometrics.py" \
    --cache "$CACHE_OUT" \
    --out "$ROOT_REMOTE/outputs/r20/per_structure_morphometrics_unified301.csv" \
    --workers 24 2>&1 | tail -10 || echo "[fallback] R19.D script not on remote"

echo "[R20.G] verify R18.B Spearman ρ for artery_len_p25 reproduces"
python -c "
import pandas as pd, numpy as np, json
from scipy.stats import spearmanr
m = pd.read_csv('$ROOT_REMOTE/outputs/r20/per_structure_morphometrics_unified301.csv')
mp = json.load(open('$ROOT_REMOTE/data/mpap_lookup_gold.json'))
m['mpap'] = m['case_id'].map(mp)
# default plain-scan = 5 (per R18.B convention)
plain = m['case_id'].str.contains('nonph_') & m['mpap'].isna()
m.loc[plain, 'mpap'] = 5.0
sub = m.dropna(subset=['mpap', 'artery_len_p25'])
rho, p = spearmanr(sub['mpap'], sub['artery_len_p25'])
print(f'unified-pipeline n={len(sub)} artery_len_p25 ρ={rho:+.3f} p={p:.2g}')
print('VERIFY: legacy R18.B was ρ=-0.767 — does this reproduce?')
"

echo "[R20.G] DONE"
