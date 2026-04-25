"""R16.C — Repair Simple_AV_seg lung-mask oversegmentation on plain-scan.

R16.A found 79/100 cases have lung mask >8.5L (median 10.8L vs normal 1.5-8.5L).
Likely cause: the model includes outside-thoracic regions (mediastinum,
fat, sometimes part of bowel/spine).

Repair strategy:
  1. HU sanity gate: drop voxels with HU > -300 (definitely not lung
     parenchyma — soft tissue / vessels handled by separate masks)
  2. Largest-2-components-by-volume filter (lungs = 1-2 connected components)
  3. Optional bbox restriction within thoracic-cavity z-range (median z ± p95-p5)

Re-extracts paren_mean_HU, paren_LAA_950, lung_vol_mL etc. from the
repaired mask. Output: outputs/r16/lung_features_new100_repaired.csv

Run on remote.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import label as cc_label

NII_ROOT = Path.cwd() / "nii-new100"
OUT = Path.cwd() / "outputs" / "r16"
OUT.mkdir(parents=True, exist_ok=True)


def repair_lung_mask(lung, ct, hu_max=-300, keep_top_k=2):
    """Drop high-HU voxels; keep top-k connected components by volume."""
    lung_repaired = lung.copy()
    # HU gate
    lung_repaired = lung_repaired & (ct < hu_max)
    if lung_repaired.sum() == 0:
        return lung_repaired
    # Largest components
    labeled, n = cc_label(lung_repaired)
    if n <= keep_top_k:
        return lung_repaired
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # background
    keep_ids = np.argsort(sizes)[-keep_top_k:]
    keep_mask = np.isin(labeled, keep_ids)
    return lung_repaired & keep_mask


def extract(case_dir):
    lung_p = case_dir / "lung.nii.gz"
    ct_p = case_dir / "ct.nii.gz"
    art_p = case_dir / "artery.nii.gz"
    vein_p = case_dir / "vein.nii.gz"
    if not (lung_p.exists() and ct_p.exists()):
        return None
    ct_img = nib.load(str(ct_p))
    ct = ct_img.get_fdata(caching="unchanged")
    sp = ct_img.header.get_zooms()
    vox_mL = float(np.prod(sp[:3])) / 1000.0
    lung_raw = nib.load(str(lung_p)).get_fdata() > 0.5
    art = (nib.load(str(art_p)).get_fdata() > 0.5) if art_p.exists() else np.zeros_like(lung_raw, bool)
    vein = (nib.load(str(vein_p)).get_fdata() > 0.5) if vein_p.exists() else np.zeros_like(lung_raw, bool)
    # Shape align
    target = lung_raw.shape
    if ct.shape != target:
        sl = tuple(slice(0, min(ct.shape[d], target[d])) for d in range(3))
        ct = ct[sl]
        lung_raw = lung_raw[tuple(slice(0, min(lung_raw.shape[d], target[d])) for d in range(3))]
        art = art[tuple(slice(0, min(art.shape[d], target[d])) for d in range(3))]
        vein = vein[tuple(slice(0, min(vein.shape[d], target[d])) for d in range(3))]
    if lung_raw.sum() == 0:
        return None
    lung_repaired = repair_lung_mask(lung_raw, ct)
    if lung_repaired.sum() == 0:
        return None
    pure_paren = lung_repaired & ~art & ~vein
    paren_hu = ct[pure_paren].astype("float32")
    whole_hu = ct[lung_repaired].astype("float32")
    rec = {
        "lung_vol_mL_raw": float(lung_raw.sum() * vox_mL),
        "lung_vol_mL_repaired": float(lung_repaired.sum() * vox_mL),
        "lung_repair_drop_frac": float(1.0 - lung_repaired.sum()/lung_raw.sum()),
        "paren_n_voxels": int(pure_paren.sum()),
        "whole_n_voxels": int(lung_repaired.sum()),
        "artery_vol_mL": float(art.sum() * vox_mL),
        "vein_vol_mL": float(vein.sum() * vox_mL),
        "vessel_airway_over_lung": float((art | vein).sum() / max(lung_repaired.sum(), 1)),
    }
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
    coords = np.argwhere(lung_repaired)
    if coords.size:
        z = coords[:, 2]; zmid = float(np.median(z))
        z_top = float(np.percentile(z, 95)); z_bot = float(np.percentile(z, 5))
        z_apical_thr = (z_top + zmid) / 2
        z_basal_thr = (z_bot + zmid) / 2
        apical = lung_repaired & (np.arange(lung_repaired.shape[2])[None, None, :] >= z_apical_thr)
        basal = lung_repaired & (np.arange(lung_repaired.shape[2])[None, None, :] <= z_basal_thr)
        for name, m in [("apical", apical), ("basal", basal)]:
            hu = ct[m & ~art & ~vein].astype("float32") if m.sum() > 0 else np.array([])
            rec[f"{name}_n_voxels"] = int(m.sum())
            if hu.size >= 100:
                rec[f"{name}_LAA_950_frac"] = float((hu < -950).mean())
                rec[f"{name}_LAA_910_frac"] = float((hu < -910).mean())
                rec[f"{name}_mean_HU"] = float(hu.mean())
        if rec.get("apical_LAA_950_frac") is not None and rec.get("basal_LAA_950_frac") is not None:
            rec["apical_basal_LAA950_gradient"] = (
                rec["apical_LAA_950_frac"] - rec["basal_LAA_950_frac"])
    rec["lung_vol_mL"] = rec["lung_vol_mL_repaired"]  # backward-compat alias
    rec["mask_convention"] = "binary_segmask_R16C_repaired"
    return rec


def main():
    case_dirs = sorted([d for d in NII_ROOT.iterdir() if d.is_dir()])
    print(f"[start] {len(case_dirs)} cases")
    rows = []
    for i, d in enumerate(case_dirs):
        try:
            rec = extract(d)
        except Exception as e:
            print(f"  [fail] {d.name}: {e}"); continue
        if rec is None: continue
        rec["case_id"] = d.name
        rows.append(rec)
        if (i+1) % 25 == 0: print(f"  ...{i+1}/{len(case_dirs)}")
    keys = sorted({k for r in rows for k in r.keys()})
    out_csv = OUT / "lung_features_new100_repaired.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id"] + [k for k in keys if k != "case_id"])
        w.writeheader()
        w.writerows(rows)
    print(f"[done] {len(rows)} rows × {len(keys)} cols → {out_csv}")
    if rows:
        repaired_vol = [r["lung_vol_mL_repaired"] for r in rows]
        raw_vol = [r["lung_vol_mL_raw"] for r in rows]
        drop = [r["lung_repair_drop_frac"] for r in rows]
        print(f"  raw vol median: {np.median(raw_vol):.0f} mL → "
              f"repaired median: {np.median(repaired_vol):.0f} mL")
        print(f"  median drop fraction: {np.median(drop):.3f}")


if __name__ == "__main__":
    main()
