#!/bin/bash
# R20.F — Apply Simple_AV_seg to legacy 201 contrast cases on REMOTE GPU 1.
# Uses pulmonary_bv5_py39 conda env (Python 3.9) — same env as R15.
set -e
source /home/imss/miniconda3/etc/profile.d/conda.sh
conda activate pulmonary_bv5_py39 || conda activate HiPaS

LEGACY_NII="/home/imss/cw/Claude_COPDnonPH_COPD-PH_CT_nii/nii"
OUT_ROOT="/home/imss/cw/GCN copdnoph copdph/nii-unified-201-savseg"
mkdir -p "$OUT_ROOT"
mkdir -p /tmp/r20f_seg_logs

cd "/home/imss/cw/pulmonary_pipeline/code/Simple_AV_seg"

run_one() {
  local case_id="$1" gpu="$2"
  local case_dir="$LEGACY_NII/$case_id"
  local ct="$case_dir/original.nii.gz"
  if [ ! -f "$ct" ]; then ct="$case_dir/ct.nii.gz"; fi
  if [ ! -f "$ct" ]; then echo "[skip] $case_id: no CT"; return; fi
  local out_dir="$OUT_ROOT/$case_id"
  if [ -f "$out_dir/lung.nii.gz" ] && [ -f "$out_dir/artery.nii.gz" ] && [ -f "$out_dir/vein.nii.gz" ]; then
    echo "[exists] $case_id"; return
  fi
  mkdir -p "$out_dir"
  echo "[gpu$gpu] $case_id"
  CUDA_VISIBLE_DEVICES=$gpu python -c "
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
print('done')
" > /tmp/r20f_seg_logs/${case_id}_gpu${gpu}.log 2>&1
}

CASES=( $(ls "$LEGACY_NII") )
N=${#CASES[@]}
HALF=$((N / 2))
echo "[start] $N cases; gpu0 first half ($HALF) gpu1 second half"

# Skip GPU 0 since it's used by another user (7GB), use only GPU 1 idle
# Actually use both with detection
GPU0_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
echo "[gpu0_used_mib] $GPU0_USED"
if [ "$GPU0_USED" -gt 5000 ]; then
  echo "[note] GPU 0 busy ($GPU0_USED MiB); using GPU 1 only"
  for ((i=0; i<N; i++)); do run_one "${CASES[i]}" 1; done
else
  (
    for ((i=0; i<HALF; i++)); do run_one "${CASES[i]}" 0; done
  ) &
  P0=$!
  (
    for ((i=HALF; i<N; i++)); do run_one "${CASES[i]}" 1; done
  ) &
  P1=$!
  wait $P0 $P1
fi
echo "[done] all $N cases segmented at $OUT_ROOT"
