"""Lung phenotype extraction v2 — protocol-robust features.

Rationale (reviewer W1): the v1 `lung_features_only.csv` computed HU/LAA on the
entire lung mask, which includes vessels whose HU is strongly contrast-enhanced.
In the protocol-balanced ablation (REPORT_v2 §13) arm_c's lung-feature gain
vanished (+0.04 → +0.006 AUC), suggesting the whole-lung features leak
protocol.

v2 fixes this by:
  1. Subtracting vessel + airway masks from the lung mask → parenchyma-only mask
  2. Reporting whole-lung AND parenchyma-only HU/LAA side-by-side (diagnostic)
  3. Adding spatial-distribution features (apical/middle/basal LAA gradient)
  4. Adding vessel-lung integration features (A/V/airway volume fractions)
  5. Adding vessel-HU summary (deliberate protocol decoder — verifies the
     protocol-confound lower bound)

All HU data come from `<case>/lung.nii.gz` and `<case>/{artery,vein,airway}.nii.gz`
where background = -2048 and non-background voxels carry raw HU.

Outputs:
  - `outputs/lung_features_v2.csv` — 282 cases × (v1 legacy + v2 new)
  - `outputs/_lung_v2_extraction.log`
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

CASES_ROOT = Path("E:/桌面文件/nii格式图/nii-unified-282")
CASES_ROOT_FALLBACK = Path("E:/桌面文件/nii格式图/nii")


def resolve_case_dir(case_id: str) -> Path:
    """Plain-scan cases have real files in nii-unified-282/<case>/.
    Contrast-enhanced cases have a `_source.txt` redirect there and actual
    files under `nii/<case>/`. We pick whichever directory contains lung.nii.gz.
    """
    primary = CASES_ROOT / case_id
    if (primary / "lung.nii.gz").exists():
        return primary
    fallback = CASES_ROOT_FALLBACK / case_id
    if (fallback / "lung.nii.gz").exists():
        return fallback
    # Try reading _source.txt
    src = primary / "_source.txt"
    if src.exists():
        target = src.read_text(encoding="utf-8", errors="ignore").strip()
        # _source.txt may contain a Windows path; normalize to forward slashes
        target_path = Path(target.replace("\\", "/"))
        if (target_path / "lung.nii.gz").exists():
            return target_path
    return primary  # will fail later
LABELS = Path(__file__).parent / "data" / "labels_expanded_282.csv"
OUT_CSV = Path(__file__).parent / "outputs" / "lung_features_v2.csv"
OUT_LOG = Path(__file__).parent / "outputs" / "_lung_v2_extraction.log"

BG = -2048
LAA_THRESH = [-950, -910, -856]
HU_PCTS = [5, 25, 50, 75, 95]
PLACEHOLDER_VOXELS = 768  # 16*16*1*1*3 signature of failed upstream segmentation


def load_mask_and_hu(path: Path) -> tuple[np.ndarray, np.ndarray, dict] | None:
    if not path.exists():
        return None
    try:
        img = nib.load(str(path))
    except Exception as exc:  # pragma: no cover
        return {"error": f"load_fail:{exc}"}  # type: ignore[return-value]
    arr = np.asarray(img.get_fdata(dtype=np.float32))
    mask = arr != BG
    hu = arr[mask]
    zooms = img.header.get_zooms()
    spacing = tuple(float(z) for z in zooms[:3])
    return mask, hu, {"spacing": spacing, "shape": arr.shape, "mask_voxels": int(mask.sum())}


def voxel_volume_ml(spacing_mm: tuple[float, float, float]) -> float:
    return float(spacing_mm[0] * spacing_mm[1] * spacing_mm[2] / 1000.0)


def hu_stats(hu: np.ndarray, prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if hu.size == 0:
        for p in HU_PCTS:
            out[f"{prefix}_HU_p{p}"] = float("nan")
        for t in LAA_THRESH:
            out[f"{prefix}_LAA_{abs(t)}_frac"] = float("nan")
        out[f"{prefix}_mean_HU"] = float("nan")
        out[f"{prefix}_std_HU"] = float("nan")
        out[f"{prefix}_n_voxels"] = 0.0
        return out
    for p in HU_PCTS:
        out[f"{prefix}_HU_p{p}"] = float(np.percentile(hu, p))
    for t in LAA_THRESH:
        out[f"{prefix}_LAA_{abs(t)}_frac"] = float((hu < t).mean())
    out[f"{prefix}_mean_HU"] = float(hu.mean())
    out[f"{prefix}_std_HU"] = float(hu.std())
    out[f"{prefix}_n_voxels"] = float(hu.size)
    return out


def load_full(path: Path) -> tuple[np.ndarray, tuple[float, float, float]] | None:
    if not path.exists():
        return None
    try:
        img = nib.load(str(path))
    except Exception:
        return None
    arr = np.asarray(img.get_fdata(dtype=np.float32))
    zooms = img.header.get_zooms()
    return arr, (float(zooms[0]), float(zooms[1]), float(zooms[2]))


def detect_convention(arr: np.ndarray) -> str:
    """Return 'hu' for HU-encoded (-2048 bg sentinel) or 'binary' (0/1 mask)."""
    # Fast check on a small central slab
    slab = arr[::32, ::32, ::32]
    if (slab == BG).any():
        return "hu"
    uniq = np.unique(slab)
    if uniq.size <= 3 and set(uniq.tolist()).issubset({0.0, 1.0, 255.0}):
        return "binary"
    # Some contrast masks encode label integers (0 = bg, >0 = struct). Treat as binary.
    if arr.max() <= 255 and arr.min() >= 0:
        return "binary"
    return "hu"


def get_mask_and_hu(
    case_dir: Path, struct: str, ct_arr: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (bool_mask, hu_values_in_struct)."""
    r = load_full(case_dir / f"{struct}.nii.gz")
    if r is None:
        return None
    arr, _ = r
    conv = detect_convention(arr)
    if conv == "hu":
        mask = arr != BG
        hu = arr[mask]
    else:
        mask = arr > 0
        if ct_arr is None:
            hu = np.empty(0, dtype=np.float32)
        else:
            hu = ct_arr[mask]
    del arr
    return mask, hu


def process_case(case_id: str) -> dict:
    case_dir = resolve_case_dir(case_id)
    row: dict[str, object] = {"case_id": case_id, "case_dir": str(case_dir)}
    lung = load_full(case_dir / "lung.nii.gz")
    if lung is None:
        row["error"] = "lung_missing_or_fail"
        return row
    lung_raw, spacing = lung
    conv = detect_convention(lung_raw)
    row["mask_convention"] = conv
    vox_ml = voxel_volume_ml(spacing)

    if conv == "hu":
        lung_mask = lung_raw != BG
        lung_hu = lung_raw[lung_mask]
        ct_arr = lung_raw  # lung_raw already holds HU inside lung
    else:
        lung_mask = lung_raw > 0
        ct_loaded = load_full(case_dir / "ct.nii.gz")
        if ct_loaded is None:
            row["error"] = "ct_missing_for_contrast_case"
            return row
        ct_arr, _ = ct_loaded
        lung_hu = ct_arr[lung_mask]
    del lung_raw

    vessels_mask = np.zeros_like(lung_mask)
    for struct in ("artery", "vein", "airway"):
        got = get_mask_and_hu(case_dir, struct, ct_arr if conv == "binary" else None)
        if got is not None:
            m, hu = got
            vx = int(m.sum())
            is_placeholder = vx <= PLACEHOLDER_VOXELS
            row[f"{struct}_placeholder"] = int(is_placeholder)
            row[f"{struct}_vol_mL"] = vx * vox_ml
            row[f"{struct}_mean_HU"] = float(hu.mean()) if hu.size else float("nan")
            if struct != "airway":
                row[f"{struct}_p95_HU"] = (
                    float(np.percentile(hu, 95)) if hu.size else float("nan")
                )
            if not is_placeholder and m.shape == lung_mask.shape:
                vessels_mask |= m
            del m, hu
        else:
            row[f"{struct}_placeholder"] = -1
            row[f"{struct}_vol_mL"] = float("nan")
            row[f"{struct}_mean_HU"] = float("nan")
            if struct != "airway":
                row[f"{struct}_p95_HU"] = float("nan")

    row["lung_vol_mL"] = int(lung_mask.sum()) * vox_ml
    row["vessel_airway_vol_mL"] = int(vessels_mask.sum()) * vox_ml
    row["vessel_airway_over_lung"] = (
        row["vessel_airway_vol_mL"] / row["lung_vol_mL"]
        if row["lung_vol_mL"] > 0
        else float("nan")
    )

    row.update(hu_stats(lung_hu, "whole"))

    paren_mask = lung_mask & ~vessels_mask
    paren_hu = ct_arr[paren_mask]
    row.update(hu_stats(paren_hu, "paren"))

    if lung_mask.any():
        zs = np.where(lung_mask.any(axis=(0, 1)))[0]
        z_lo, z_hi = int(zs[0]), int(zs[-1])
        z_split1 = z_lo + (z_hi - z_lo) // 3
        z_split2 = z_lo + 2 * (z_hi - z_lo) // 3
        for band_name, z0, z1 in (
            ("basal", z_lo, z_split1),
            ("middle", z_split1, z_split2),
            ("apical", z_split2, z_hi + 1),
        ):
            band_paren = paren_mask.copy()
            band_paren[..., :z0] = False
            band_paren[..., z1:] = False
            b_hu = ct_arr[band_paren]
            if b_hu.size:
                row[f"{band_name}_LAA_950_frac"] = float((b_hu < -950).mean())
                row[f"{band_name}_LAA_910_frac"] = float((b_hu < -910).mean())
                row[f"{band_name}_mean_HU"] = float(b_hu.mean())
                row[f"{band_name}_n_voxels"] = float(b_hu.size)
            else:
                row[f"{band_name}_LAA_950_frac"] = float("nan")
                row[f"{band_name}_LAA_910_frac"] = float("nan")
                row[f"{band_name}_mean_HU"] = float("nan")
                row[f"{band_name}_n_voxels"] = 0.0
        a = row.get("apical_LAA_950_frac", float("nan"))
        b = row.get("basal_LAA_950_frac", float("nan"))
        row["apical_basal_LAA950_gradient"] = (
            float(a - b) if not (np.isnan(a) or np.isnan(b)) else float("nan")
        )

    row["error"] = ""
    return row


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    # Load case list from labels_expanded_282.csv
    import csv as _csv

    with LABELS.open() as f:
        reader = _csv.DictReader(f)
        case_ids = [r["case_id"] for r in reader]
    print(f"Processing {len(case_ids)} cases…")
    t0 = time.time()

    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(process_case, c): c for c in case_ids}
        for i, fut in enumerate(as_completed(futures), 1):
            c = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:
                res = {"case_id": c, "error": f"exception:{exc}"}
            results.append(res)
            if i % 20 == 0 or i == len(case_ids):
                elapsed = time.time() - t0
                print(f"  {i}/{len(case_ids)}  elapsed={elapsed:.1f}s  {c}")

    # Write CSV
    all_keys = set()
    for r in results:
        all_keys.update(r.keys())
    header = ["case_id"] + sorted(k for k in all_keys if k != "case_id")
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in sorted(results, key=lambda x: x.get("case_id", "")):
            w.writerow(r)
    print(f"\nWrote {OUT_CSV}  ({len(results)} rows, {len(header)} cols)")
    elapsed = time.time() - t0
    OUT_LOG.write_text(
        json.dumps(
            {"n_cases": len(results), "elapsed_s": elapsed, "header_len": len(header)},
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
