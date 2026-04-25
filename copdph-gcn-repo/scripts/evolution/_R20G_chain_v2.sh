#!/bin/bash
# R20.G v2 — chain: cache build + morph extract on unified-301 cohort.
# Run after R20.F segmentation completes (>=199/201 cases).
source /home/imss/miniconda3/etc/profile.d/conda.sh
conda activate pulmonary_bv5_py39 || conda activate HiPaS

ROOT="/home/imss/cw/GCN copdnoph copdph"
cd "$ROOT"

N_201=$(ls "$ROOT/nii-unified-201-savseg" 2>/dev/null | wc -l)
echo "[R20.G] segmentation outputs: $N_201/201"
if [ "$N_201" -lt 195 ]; then
  echo "[abort] not enough segmentations"; exit 1
fi

# Step 1: build cache_tri_v2_unified301
CACHE="$ROOT/cache_tri_v2_unified301"
mkdir -p "$CACHE"
echo "[R20.G] building cache → $CACHE"
python -u _R19C_build_v2_patched.py \
  --labels "$ROOT/copdph-gcn-repo/data/labels_extended_382.csv" \
  --data_dirs "$ROOT/nii-unified-201-savseg" "$ROOT/nii-new100" \
  --output_cache "$CACHE" \
  --workers 6 2>&1 | tail -30

N_CACHE=$(ls "$CACHE"/*_tri.pkl 2>/dev/null | wc -l)
echo "[R20.G] cache built: $N_CACHE pkl files"

# Step 2: morphometrics extraction on unified cache (script reads from
#           hard-coded paths; override via tmp-script copy)
echo "[R20.G] morphometrics extraction"
TMP_R19D="/tmp/_R20G_morph_unified301.py"
cp _R19D_extended_morph.py "$TMP_R19D"
sed -i 's|CACHE_LEGACY = .*|CACHE_LEGACY = Path("'"$CACHE"'")|' "$TMP_R19D"
sed -i 's|CACHE_NEW100 = .*|CACHE_NEW100 = Path("/dev/null")|' "$TMP_R19D"
sed -i 's|"legacy"|"unified301"|' "$TMP_R19D"
python -u "$TMP_R19D" 2>&1 | tail -20

# Step 3: copy outputs to project repo
mkdir -p "$ROOT/copdph-gcn-repo/outputs/r20"
cp "$ROOT/outputs/r19/extended_morphometrics.csv" \
   "$ROOT/copdph-gcn-repo/outputs/r20/morph_unified301.csv" 2>/dev/null || true

echo "[R20.G] DONE"
ls -la "$ROOT/copdph-gcn-repo/outputs/r20/morph_unified301.csv" 2>&1 | tail -2
