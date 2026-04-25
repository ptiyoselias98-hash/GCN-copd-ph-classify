"""R15.1 — Segmentation pipeline launcher for nii-new100 cases.

To be invoked by next cron fire after scp finishes.
Generates a remote bash script that batches Simple_AV_seg inference over
the 100 new plain-scan nonPH cases. Runs inference for each case_dir and
saves lung.nii.gz + artery.nii.gz + vein.nii.gz alongside the input
ct.nii.gz on remote.

Pipeline checkpoint location: /home/imss/cw/pulmonary_pipeline/code/Simple_AV_seg/
Models: lung.pth + main_AV.pth (Frangi-based vesselness in frangi_gpu.py)

After segmentation: airway segmentation needed (separate tool — TBD R16);
for now produce the lung+artery+vein triplet so v2 cache rebuild has the
3 channels needed for the GCN.

This script writes the remote bash to /tmp/launch_seg_new100.sh; another
fire will scp it and launch with `nohup bash /tmp/launch_seg_new100.sh`.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
OUT = ROOT / "outputs" / "r15"
OUT.mkdir(parents=True, exist_ok=True)
NEW100_LIST = OUT / "nii_new100_caselist.txt"
LAUNCHER = OUT / "launch_seg_new100.sh"

REMOTE_NII_DIR = "/home/imss/cw/GCN copdnoph copdph/nii-new100"
REMOTE_SEG_OUT = "/home/imss/cw/GCN copdnoph copdph/nii-new100"
SEG_CODE = "/home/imss/cw/pulmonary_pipeline/code/Simple_AV_seg"


def main():
    log = json.loads((OUT / "dcm_conversion_log.json").read_text(encoding="utf-8"))
    cases = [c["case_id"] for c in log["cases"]
             if c.get("status") == "ok" and c.get("case_id")]
    NEW100_LIST.write_text("\n".join(cases) + "\n", encoding="utf-8")
    print(f"Wrote {len(cases)} case ids → {NEW100_LIST}")

    bash = f"""#!/bin/bash
# R15.1 segmentation launcher for {len(cases)} new plain-scan nonPH cases
set -e
source /home/imss/miniconda3/etc/profile.d/conda.sh
conda activate pulmonary_bv5_py39 || conda activate HiPaS
cd "{SEG_CODE}"
mkdir -p /tmp/r15_seg_logs

run_one() {{
  local case_id="$1" gpu="$2"
  local case_dir="{REMOTE_NII_DIR}/$case_id"
  local ct="$case_dir/ct.nii.gz"
  if [ ! -f "$ct" ]; then echo "[skip] $case_id: no ct.nii.gz"; return; fi
  if [ -f "$case_dir/lung.nii.gz" ] && [ -f "$case_dir/artery.nii.gz" ] && [ -f "$case_dir/vein.nii.gz" ]; then
    echo "[exists] $case_id"; return
  fi
  echo "[gpu$gpu] $case_id"
  CUDA_VISIBLE_DEVICES=$gpu python -c "
import sys; sys.path.insert(0, '{SEG_CODE}')
import nibabel as nib, numpy as np
from prediction import predict_zoomed
img = nib.load('$ct')
arr = img.get_fdata().astype('float32')
# Normalize HU [-1000, 600] -> [0, 1]
arr_n = np.clip((arr + 1000) / 1600.0, 0, 1).transpose(2, 0, 1)
artery, vein, lung = predict_zoomed(arr_n)
def save(mask, name):
    out = np.transpose(mask.astype('uint8'), (1, 2, 0))
    nib.save(nib.Nifti1Image(out, img.affine, img.header), '$case_dir/' + name)
save(artery, 'artery.nii.gz'); save(vein, 'vein.nii.gz'); save(lung, 'lung.nii.gz')
print('done')
" > /tmp/r15_seg_logs/${{case_id}}_gpu${{gpu}}.log 2>&1
}}

# Two parallel queues, one per GPU
CASES=( {' '.join(f'"{c}"' for c in cases)} )
N=${{#CASES[@]}}
HALF=$((N / 2))
(
  for ((i=0; i<HALF; i++)); do run_one "${{CASES[i]}}" 0; done
) &
P0=$!
(
  for ((i=HALF; i<N; i++)); do run_one "${{CASES[i]}}" 1; done
) &
P1=$!
wait $P0 $P1
echo "[done] all $N cases segmented"
"""
    LAUNCHER.write_text(bash, encoding="utf-8")
    print(f"Wrote launcher → {LAUNCHER}")


if __name__ == "__main__":
    main()
