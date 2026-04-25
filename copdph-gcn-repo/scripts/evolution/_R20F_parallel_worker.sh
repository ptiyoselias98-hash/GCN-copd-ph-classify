#!/bin/bash
# R20.F parallel worker — picks up cases not yet done.
# Launch alongside primary worker for 2x throughput on GPU 1 (RTX 3090 24GB).
# NO `set -e` — tolerate per-case failures (corrupted gz, OOM); loop continues.
source /home/imss/miniconda3/etc/profile.d/conda.sh
conda activate pulmonary_bv5_py39 || conda activate HiPaS

LEGACY_NII="/home/imss/cw/Claude_COPDnonPH_COPD-PH_CT_nii/nii"
OUT_ROOT="/home/imss/cw/GCN copdnoph copdph/nii-unified-201-savseg"
mkdir -p /tmp/r20f_seg_logs

cd "/home/imss/cw/pulmonary_pipeline/code/Simple_AV_seg"

run_one() {
  local case_id="$1"
  local case_dir="$LEGACY_NII/$case_id"
  local ct="$case_dir/original.nii.gz"
  if [ ! -f "$ct" ]; then ct="$case_dir/ct.nii.gz"; fi
  if [ ! -f "$ct" ]; then return; fi
  local out_dir="$OUT_ROOT/$case_id"
  if [ -f "$out_dir/lung.nii.gz" ] && [ -f "$out_dir/artery.nii.gz" ] && [ -f "$out_dir/vein.nii.gz" ]; then
    return
  fi
  mkdir -p "$out_dir"
  echo "[gpu1-w2] $case_id"
  CUDA_VISIBLE_DEVICES=1 python -c "
import sys; sys.path.insert(0, '/home/imss/cw/pulmonary_pipeline/code/Simple_AV_seg')
import nibabel as nib, numpy as np
from prediction import predict_zoomed
img = nib.load('$ct')
arr = img.get_fdata().astype('float32')
arr_n = np.clip((arr + 1000) / 1600.0, 0, 1).transpose(2, 0, 1)
artery, vein, lung = predict_zoomed(arr_n)
def save(mask, name):
    out = np.transpose(mask.astype('uint8'), (1, 2, 0))
    nib.save(nib.Nifti1Image(out, img.affine, img.header), '$out_dir/' + name)
save(artery, 'artery.nii.gz'); save(vein, 'vein.nii.gz'); save(lung, 'lung.nii.gz')
" > /tmp/r20f_seg_logs/${case_id}_w2.log 2>&1
}

# Process cases in REVERSE order so we don't collide with worker 1 (forward).
# `|| true` ensures one bad case doesn't kill the loop.
CASES=( $(ls "$LEGACY_NII" | tac) )
for c in "${CASES[@]}"; do run_one "$c" || true; done
echo "[w2 done]"
