"""Compute PA / Aorta diameters per case from artery masks + CT NIfTI.

Designed to run REMOTELY (server has the NIfTI data). Produces one CSV row
per case: pa_diam_mm, ao_diam_mm, pa_ao_ratio, slice_z.

Method:
  PA diameter:
    - Load artery.nii.gz; take the axial slice with the largest connected
      artery area above the carina region (middle third of z extent).
    - Pick the largest connected component in that slice (the PA trunk).
    - Equivalent-circle diameter: 2 * sqrt(area / pi), converted to mm via
      the in-plane voxel spacing.
  Aorta diameter:
    - Prefer aorta.nii.gz if it exists.
    - Else, on the SAME axial slice, threshold CT HU ∈ [threshold_lo, hi]
      (default 150-600 HU for contrast; 30-90 for non-contrast auto-fallback),
      remove artery voxels, pick largest circular blob near the PA center.

Usage (remote):
  python compute_pa_ao_from_masks.py \
    --nii_root "/home/imss/cw/COPDnonPH COPD-PH /data/nii" \
    --out ./data/pa_ao_measurements.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except Exception:
    nib = None
try:
    from scipy import ndimage as ndi
except Exception:
    ndi = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("pa_ao")


def largest_cc_area(mask2d: np.ndarray) -> tuple[int, np.ndarray]:
    if ndi is None:
        return int(mask2d.sum()), mask2d.astype(bool)
    lbl, n = ndi.label(mask2d > 0)
    if n == 0:
        return 0, np.zeros_like(mask2d, dtype=bool)
    sizes = ndi.sum(mask2d > 0, lbl, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    return int(sizes[k - 1]), (lbl == k)


def equiv_circle_diam_mm(area_vox: int, sx: float, sy: float) -> float:
    area_mm2 = area_vox * float(sx) * float(sy)
    return float(2.0 * math.sqrt(area_mm2 / math.pi))


def measure_case(case_dir: Path) -> dict | None:
    art_p = case_dir / "artery.nii.gz"
    if not art_p.exists():
        logger.info("SKIP %s: no artery.nii.gz", case_dir.name)
        return None
    ao_p = case_dir / "aorta.nii.gz"
    ct_p = None
    for cand in ("ct.nii.gz", "image.nii.gz", "CT.nii.gz", "img.nii.gz"):
        if (case_dir / cand).exists():
            ct_p = case_dir / cand
            break

    art_img = nib.load(str(art_p))
    art = art_img.get_fdata() > 0
    aff = art_img.affine
    # voxel spacing (mm) — from affine column norms
    spacing = np.linalg.norm(aff[:3, :3], axis=0)
    sx, sy, sz = float(spacing[0]), float(spacing[1]), float(spacing[2])

    # Scan middle third of z for PA-trunk candidate; pick slice with largest
    # connected artery area.
    Z = art.shape[2]
    z_lo, z_hi = int(Z * 0.30), int(Z * 0.75)
    best_z, best_area, best_cc = -1, 0, None
    for z in range(z_lo, z_hi):
        sl = art[:, :, z]
        if sl.sum() < 20:
            continue
        a, cc = largest_cc_area(sl)
        if a > best_area:
            best_area = a; best_z = z; best_cc = cc
    if best_z < 0:
        logger.warning("%s: no valid PA slice", case_dir.name)
        return None
    pa_d = equiv_circle_diam_mm(best_area, sx, sy)

    # Aorta
    ao_d = float("nan")
    if ao_p.exists():
        ao_img = nib.load(str(ao_p))
        ao = ao_img.get_fdata() > 0
        ao_sl = ao[:, :, best_z] if ao.shape == art.shape else None
        if ao_sl is not None and ao_sl.sum() > 0:
            a, _ = largest_cc_area(ao_sl)
            ao_d = equiv_circle_diam_mm(a, sx, sy)
    if math.isnan(ao_d) and ct_p is not None:
        ct = nib.load(str(ct_p)).get_fdata()
        if ct.shape == art.shape:
            sl_ct = ct[:, :, best_z]
            # Try contrast-enhanced range first, then fallback
            for lo, hi in ((150, 600), (30, 90)):
                m = (sl_ct >= lo) & (sl_ct <= hi)
                m = m & ~art[:, :, best_z]
                if ndi is not None:
                    m = ndi.binary_opening(m, iterations=1)
                lbl, n = (ndi.label(m) if ndi is not None else (m.astype(int), 1))
                if n == 0:
                    continue
                # Find centroid nearest to PA centroid; pick that CC
                pa_cent = ndi.center_of_mass(best_cc) if ndi is not None else (0, 0)
                best_k, best_score = -1, -1e9
                for k in range(1, n + 1):
                    comp = (lbl == k)
                    size = int(comp.sum())
                    if size < 40:
                        continue
                    c = ndi.center_of_mass(comp) if ndi is not None else (0, 0)
                    # Aorta is typically medial-anterior near PA; score on
                    # size - distance penalty
                    dist = math.hypot(c[0] - pa_cent[0], c[1] - pa_cent[1])
                    score = size - 0.5 * dist
                    if score > best_score:
                        best_score = score; best_k = k
                if best_k > 0:
                    a_vox = int((lbl == best_k).sum())
                    ao_d = equiv_circle_diam_mm(a_vox, sx, sy)
                    break
    ratio = pa_d / ao_d if (ao_d and not math.isnan(ao_d) and ao_d > 0) else float("nan")
    return {
        "case_id": case_dir.name,
        "pa_diam_mm": round(pa_d, 3),
        "ao_diam_mm": round(ao_d, 3) if not math.isnan(ao_d) else "",
        "pa_ao_ratio": round(ratio, 4) if not math.isnan(ratio) else "",
        "slice_z": best_z,
        "sx_mm": round(sx, 4),
        "sy_mm": round(sy, 4),
        "sz_mm": round(sz, 4),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--nii_root", required=True)
    p.add_argument("--out", default="./data/pa_ao_measurements.csv")
    p.add_argument("--pattern", default="*")
    args = p.parse_args()

    if nib is None:
        logger.error("nibabel not available; install it in the remote env")
        return 2
    nii_root = Path(args.nii_root)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    cases = sorted([p for p in nii_root.glob(args.pattern) if p.is_dir()])
    logger.info("found %d case dirs under %s", len(cases), nii_root)

    rows = []
    for i, cd in enumerate(cases, 1):
        try:
            r = measure_case(cd)
            if r:
                rows.append(r)
                logger.info("[%d/%d] %s PA=%.2fmm Ao=%s ratio=%s",
                            i, len(cases), r["case_id"], r["pa_diam_mm"],
                            r["ao_diam_mm"], r["pa_ao_ratio"])
        except Exception as e:
            logger.exception("FAIL %s: %s", cd.name, e)

    if rows:
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        logger.info("wrote %s (%d rows)", out, len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
