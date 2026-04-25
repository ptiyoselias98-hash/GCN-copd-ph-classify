"""R15.D — Lung-feature extraction for the 100 new plain-scan nonPH cases.

Produces a lung_features_new100.csv that matches the column convention of
the legacy lung_features_v2.csv so it can be appended into a unified
paren-feature panel for the enlarged within-nonPH stratum.

Run on remote where masks live (cd to project root before launch).
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import nibabel as nib

NII_ROOT = Path.cwd() / "nii-new100"
OUT = Path.cwd() / "outputs" / "r15"
OUT.mkdir(parents=True, exist_ok=True)


def extract(case_dir):
    lung_p = case_dir / "lung.nii.gz"
    art_p = case_dir / "artery.nii.gz"
    vein_p = case_dir / "vein.nii.gz"
    ct_p = case_dir / "ct.nii.gz"
    rec = {}
    if not (lung_p.exists() and ct_p.exists()):
        return None
    ct_img = nib.load(str(ct_p))
    ct = ct_img.get_fdata(caching="unchanged")
    sp = ct_img.header.get_zooms()
    vox_mL = float(np.prod(sp[:3])) / 1000.0
    lung = nib.load(str(lung_p)).get_fdata() > 0.5
    if lung.sum() == 0:
        return None
    art = (nib.load(str(art_p)).get_fdata() > 0.5) if art_p.exists() else np.zeros_like(lung, bool)
    vein = (nib.load(str(vein_p)).get_fdata() > 0.5) if vein_p.exists() else np.zeros_like(lung, bool)
    pure_paren = lung & ~art & ~vein
    paren_hu = ct[pure_paren].astype("float32")
    whole_hu = ct[lung].astype("float32")
    rec["lung_vol_mL"] = float(lung.sum() * vox_mL)
    rec["paren_n_voxels"] = int(pure_paren.sum())
    rec["whole_n_voxels"] = int(lung.sum())
    rec["artery_vol_mL"] = float(art.sum() * vox_mL)
    rec["vein_vol_mL"] = float(vein.sum() * vox_mL)
    rec["airway_vol_mL"] = 0.0
    rec["vessel_airway_over_lung"] = float((art | vein).sum() / max(lung.sum(), 1))
    if paren_hu.size >= 1000:
        rec["paren_LAA_950_frac"] = float((paren_hu < -950).mean())
        rec["paren_LAA_910_frac"] = float((paren_hu < -910).mean())
        rec["paren_LAA_856_frac"] = float((paren_hu < -856).mean())
        rec["paren_mean_HU"] = float(paren_hu.mean())
        rec["paren_std_HU"] = float(paren_hu.std(ddof=0))
        for q, name in [(5, "p5"), (25, "p25"), (50, "p50"), (75, "p75"), (95, "p95")]:
            rec[f"paren_HU_{name}"] = float(np.percentile(paren_hu, q))
    if whole_hu.size >= 1000:
        rec["whole_LAA_950_frac"] = float((whole_hu < -950).mean())
        rec["whole_LAA_910_frac"] = float((whole_hu < -910).mean())
        rec["whole_LAA_856_frac"] = float((whole_hu < -856).mean())
        rec["whole_mean_HU"] = float(whole_hu.mean())
        rec["whole_std_HU"] = float(whole_hu.std(ddof=0))
        for q, name in [(5, "p5"), (25, "p25"), (50, "p50"), (75, "p75"), (95, "p95")]:
            rec[f"whole_HU_{name}"] = float(np.percentile(whole_hu, q))
    coords = np.argwhere(lung)
    if coords.size:
        z = coords[:, 2]; zmid = float(np.median(z))
        z_top = float(np.percentile(z, 95)); z_bot = float(np.percentile(z, 5))
        z_apical_thr = (z_top + zmid) / 2
        z_basal_thr = (z_bot + zmid) / 2
        apical = lung & (np.arange(lung.shape[2])[None, None, :] >= z_apical_thr)
        basal = lung & (np.arange(lung.shape[2])[None, None, :] <= z_basal_thr)
        middle = lung & ~apical & ~basal
        for name, m in [("apical", apical), ("basal", basal), ("middle", middle)]:
            hu = ct[m & ~art & ~vein].astype("float32") if m.sum() > 0 else np.array([])
            rec[f"{name}_n_voxels"] = int(m.sum())
            if hu.size >= 100:
                rec[f"{name}_LAA_950_frac"] = float((hu < -950).mean())
                rec[f"{name}_LAA_910_frac"] = float((hu < -910).mean())
                rec[f"{name}_mean_HU"] = float(hu.mean())
        if rec.get("apical_LAA_950_frac") is not None and rec.get("basal_LAA_950_frac") is not None:
            rec["apical_basal_LAA950_gradient"] = (
                rec["apical_LAA_950_frac"] - rec["basal_LAA_950_frac"])
    rec["mask_convention"] = "binary_segmask_post_R15.1"
    return rec


def main():
    case_dirs = sorted(NII_ROOT.iterdir())
    case_dirs = [d for d in case_dirs if d.is_dir()]
    print(f"[start] {len(case_dirs)} case dirs in {NII_ROOT}")
    rows = []
    for i, d in enumerate(case_dirs):
        try:
            rec = extract(d)
        except Exception as e:
            print(f"  [fail] {d.name}: {e}"); continue
        if rec is None:
            print(f"  [skip empty] {d.name}"); continue
        rec["case_id"] = d.name
        rec["case_dir"] = str(d)
        rows.append(rec)
        if (i+1) % 25 == 0: print(f"  ...{i+1}/{len(case_dirs)}")
    if not rows:
        print("[abort] no rows"); return
    keys = sorted({k for r in rows for k in r.keys()})
    out_csv = OUT / "lung_features_new100.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id"] + [k for k in keys if k != "case_id"])
        w.writeheader()
        w.writerows(rows)
    print(f"[done] {len(rows)} rows × {len(keys)} cols → {out_csv}")


if __name__ == "__main__":
    main()
