"""R20.F — Apply Simple_AV_seg to legacy 201 contrast cases on REMOTE.

Discovered remote dir `/home/imss/cw/Claude_COPDnonPH_COPD-PH_CT_nii/nii/`
contains 201 contrast cases (174 PH + 27 nonPH) with `original.nii.gz`.
This is the legacy 282 contrast subset (197) plus 4 extra. Re-segmenting
with Simple_AV_seg unifies the pipeline with new100, allowing enlarged-cohort
biological claims to survive cross-pipeline scrutiny.

Output: /home/imss/cw/GCN copdnoph copdph/nii-unified-201-savseg/<case>/
            {lung,artery,vein}.nii.gz   (binary uint8 masks)

Usage on remote (GPU 1 idle, GPU 0 has 7GB used by other user):
    CUDA_VISIBLE_DEVICES=1 python -u _R20F_unify_simple_av_seg.py
        --skip_existing > /tmp/r20f_savseg.log 2>&1 &
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import nibabel as nib

SAV_CODE = "/home/imss/cw/pulmonary_pipeline/code/Simple_AV_seg"
sys.path.insert(0, SAV_CODE)

LEGACY_NII = Path("/home/imss/cw/Claude_COPDnonPH_COPD-PH_CT_nii/nii")
OUT_ROOT = Path("/home/imss/cw/GCN copdnoph copdph") / "nii-unified-201-savseg"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()

    case_dirs = sorted(d for d in LEGACY_NII.iterdir() if d.is_dir())
    if args.limit > 0:
        case_dirs = case_dirs[:args.limit]
    print(f"[start] {len(case_dirs)} cases at {LEGACY_NII}", flush=True)

    from prediction import predict_zoomed
    import torch

    n_done = 0; n_skip = 0; n_fail = 0
    for cd in case_dirs:
        out_dir = OUT_ROOT / cd.name
        if args.skip_existing and (out_dir / "lung.nii.gz").exists() and \
                (out_dir / "artery.nii.gz").exists() and \
                (out_dir / "vein.nii.gz").exists():
            n_skip += 1
            continue
        ct_p = cd / "original.nii.gz"
        if not ct_p.exists():
            ct_p = cd / "ct.nii.gz"
        if not ct_p.exists():
            print(f"  [skip no ct] {cd.name}", flush=True)
            n_skip += 1; continue
        t0 = time.time()
        try:
            img = nib.load(str(ct_p))
            arr = img.get_fdata().astype("float32")
            arr_n = np.clip((arr + 1000) / 1600.0, 0, 1).transpose(2, 0, 1)
            artery, vein, lung = predict_zoomed(arr_n)
            out_dir.mkdir(parents=True, exist_ok=True)
            for mask, name in [(artery, "artery"), (vein, "vein"),
                               (lung, "lung")]:
                m = np.transpose(mask.astype("uint8"), (1, 2, 0))
                nib.save(nib.Nifti1Image(m, img.affine, img.header),
                         str(out_dir / f"{name}.nii.gz"))
            n_done += 1
            wall = time.time() - t0
            if n_done % 5 == 0:
                print(f"  ...{n_done}/{len(case_dirs)} done | last={wall:.1f}s",
                      flush=True)
        except Exception as e:
            print(f"  [fail] {cd.name}: {str(e)[:120]}", flush=True)
            n_fail += 1
        torch.cuda.empty_cache()

    print(f"\n[done] done={n_done} skip={n_skip} fail={n_fail}", flush=True)


if __name__ == "__main__":
    main()
