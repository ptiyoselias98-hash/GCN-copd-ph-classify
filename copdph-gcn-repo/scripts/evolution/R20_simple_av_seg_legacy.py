"""R20.A — Pipeline unification: apply Simple_AV_seg to legacy 282 contrast cases.

Goal: produce binary lung/artery/vein masks for legacy 282 cohort using
the SAME pipeline as new100 (Simple_AV_seg). Then re-run v2 cache build
+ R17 morphometrics on the unified-pipeline cohort. Closes R18 must-fix
#4 + unblocks #3 (embedding-level enlarged probe) + #5 (multi-seed CORAL
on enlarged stratum).

Trade-off: replaces HU-sentinel HiPaS-style legacy masks with
Simple_AV_seg binary masks. Expected within-pipeline biology to be
preserved (R18.B trends should reproduce within Simple_AV_seg-only
cohort if biology is real, not pipeline-specific).

Run on remote GPU (after DDPM training completes; GPU 0 free):
    CUDA_VISIBLE_DEVICES=0 python _R20A_simple_av_seg_legacy.py

Output: nii-unified-282-savseg/<case>/{lung,artery,vein}.nii.gz
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import nibabel as nib

# Add Simple_AV_seg code to path
SAV_CODE = "/home/imss/cw/pulmonary_pipeline/code/Simple_AV_seg"
sys.path.insert(0, SAV_CODE)

LEGACY_NII = Path("/home/imss/cw/GCN copdnoph copdph") / "nii-unified-282"
OUT_ROOT = Path("/home/imss/cw/GCN copdnoph copdph") / "nii-unified-282-savseg"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0,
                   help="Process only first N cases (for testing)")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip cases already processed")
    args = p.parse_args()

    # Note: nii-unified-282 may not exist on remote (only on local E:); R20.A
    # will need legacy CTs scp'd to remote first OR run locally with sitk
    # (slower). For now: print diagnostic if path missing.
    if not LEGACY_NII.exists():
        print(f"[abort] {LEGACY_NII} not on remote — need to scp legacy "
              f"282 ct.nii.gz files or run locally")
        return

    case_dirs = sorted(d for d in LEGACY_NII.iterdir() if d.is_dir())
    if args.limit > 0:
        case_dirs = case_dirs[:args.limit]
    print(f"[start] {len(case_dirs)} legacy cases")

    # Lazy import to avoid model-load cost if no cases
    from prediction import predict_zoomed
    import torch

    n_done = 0; n_skip = 0; n_fail = 0
    for cd in case_dirs:
        out_dir = OUT_ROOT / cd.name
        if args.skip_existing and (out_dir / "lung.nii.gz").exists():
            n_skip += 1
            continue
        ct_p = cd / "ct.nii.gz"
        if not ct_p.exists():
            print(f"  [skip no ct] {cd.name}")
            n_skip += 1; continue
        try:
            img = nib.load(str(ct_p))
            arr = img.get_fdata().astype("float32")
            arr_n = np.clip((arr + 1000) / 1600.0, 0, 1).transpose(2, 0, 1)
            artery, vein, lung = predict_zoomed(arr_n)
            out_dir.mkdir(parents=True, exist_ok=True)
            for mask, name in [(artery, "artery"), (vein, "vein"), (lung, "lung")]:
                m = np.transpose(mask.astype("uint8"), (1, 2, 0))
                nib.save(nib.Nifti1Image(m, img.affine, img.header),
                         str(out_dir / f"{name}.nii.gz"))
            n_done += 1
            if n_done % 10 == 0:
                print(f"  ...{n_done}/{len(case_dirs)} done")
        except Exception as e:
            print(f"  [fail] {cd.name}: {str(e)[:120]}")
            n_fail += 1
        torch.cuda.empty_cache()

    print(f"\n[done] done={n_done} skip={n_skip} fail={n_fail}")


if __name__ == "__main__":
    main()
