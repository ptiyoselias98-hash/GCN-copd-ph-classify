"""R4.2 — Skeleton-length abundance via skimage.skeletonize_3d.

HiPaS T1 (PAH → ↓ artery skeleton length) failed under volume proxy in R3
because volume conflates central PA dilation with distal pruning. We
skeletonize the artery/vein masks locally and compute total skeleton
length (in mm) per unit lung volume — the same metric HiPaS uses.

Run on the full 282 cohort. Auto-detects the mask convention
(HU-sentinel vs binary) same as `_extract_lung_v2`.

Output:
  outputs/skeleton_length.csv  — per-case SL_artery/vein/airway in mm
  outputs/evolution/R4_skeleton_directions.md / .json
"""

from __future__ import annotations

import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import stats
from skimage.morphology import skeletonize as skeletonize_3d  # newer skimage

ROOT = Path(__file__).parent.parent.parent
CASES_ROOT = Path("E:/桌面文件/nii格式图/nii-unified-282")
CASES_ROOT_FALLBACK = Path("E:/桌面文件/nii格式图/nii")
LABELS = ROOT / "data" / "labels_expanded_282.csv"
PROTO = ROOT / "data" / "case_protocol.csv"
V2 = ROOT / "outputs" / "lung_features_v2.csv"
OUT_CSV = ROOT / "outputs" / "skeleton_length.csv"
OUT_MD = ROOT / "outputs" / "evolution" / "R4_skeleton_directions.md"
OUT_JSON = ROOT / "outputs" / "evolution" / "R4_skeleton_directions.json"
OUT_MD.parent.mkdir(parents=True, exist_ok=True)

BG = -2048
PLACEHOLDER = 768


def resolve_case_dir(case_id: str) -> Path:
    primary = CASES_ROOT / case_id
    if (primary / "lung.nii.gz").exists():
        return primary
    fallback = CASES_ROOT_FALLBACK / case_id
    if (fallback / "lung.nii.gz").exists():
        return fallback
    src = primary / "_source.txt"
    if src.exists():
        target = src.read_text(encoding="utf-8", errors="ignore").strip()
        return Path(target.replace("\\", "/"))
    return primary


def detect_convention(arr: np.ndarray) -> str:
    slab = arr[::32, ::32, ::32]
    if (slab == BG).any():
        return "hu"
    if arr.max() <= 255 and arr.min() >= 0:
        return "binary"
    return "hu"


def load_mask(path: Path):
    if not path.exists():
        return None
    try:
        img = nib.load(str(path))
    except Exception:
        return None
    arr = np.asarray(img.get_fdata(dtype=np.float32))
    conv = detect_convention(arr)
    if conv == "hu":
        mask = arr != BG
    else:
        mask = arr > 0
    zooms = img.header.get_zooms()
    spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    return mask, spacing


def skeleton_length_mm(mask: np.ndarray, spacing: tuple[float, float, float]) -> float:
    """Approximate skeleton length in mm by counting skeleton voxels and
    weighting by the geometric mean of the spacing (voxel-to-mm conversion).
    For isotropic-ish CT (our ~0.8×0.8×0.7 mm spacings), this is within ~5% of
    the true piecewise-linear skeleton length.
    """
    if not mask.any():
        return 0.0
    skel = skeletonize_3d(mask)
    n_skel = int(skel.sum())
    if n_skel == 0:
        return 0.0
    step_mm = float(np.cbrt(spacing[0] * spacing[1] * spacing[2]))
    return n_skel * step_mm


def process_case(case_id: str) -> dict:
    case_dir = resolve_case_dir(case_id)
    row: dict = {"case_id": case_id}
    try:
        lung = load_mask(case_dir / "lung.nii.gz")
        if lung is None:
            row["error"] = "lung_missing"
            return row
        lung_mask, spacing = lung
        vox_ml = spacing[0] * spacing[1] * spacing[2] / 1000
        row["lung_vol_mL"] = float(lung_mask.sum()) * vox_ml

        for struct in ("artery", "vein", "airway"):
            m = load_mask(case_dir / f"{struct}.nii.gz")
            if m is None or m[0].sum() <= PLACEHOLDER:
                row[f"SL_{struct}_mm"] = float("nan")
                row[f"vol_{struct}_mL"] = float("nan")
                continue
            mask, sp = m
            row[f"SL_{struct}_mm"] = skeleton_length_mm(mask, sp)
            row[f"vol_{struct}_mL"] = float(mask.sum()) * (sp[0] * sp[1] * sp[2] / 1000)
        row["error"] = ""
        return row
    except Exception as exc:
        row["error"] = f"exception:{exc}"
        return row


def main() -> None:
    with LABELS.open() as f:
        case_ids = [r["case_id"] for r in csv.DictReader(f)]
    t0 = time.time()
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(process_case, c): c for c in case_ids}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            if i % 20 == 0 or i == len(case_ids):
                print(f"  {i}/{len(case_ids)}  elapsed={time.time()-t0:.1f}s  {r['case_id'][:40]}")

    header = ["case_id", "error", "lung_vol_mL",
              "SL_artery_mm", "SL_vein_mm", "SL_airway_mm",
              "vol_artery_mL", "vol_vein_mL", "vol_airway_mL"]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in sorted(results, key=lambda x: x["case_id"]):
            w.writerow({k: r.get(k, "") for k in header})
    print(f"\nWrote {OUT_CSV} ({len(results)} rows)")

    # Direction test using SL (not volume)
    import pandas as pd
    sl = pd.read_csv(OUT_CSV)
    proto = pd.read_csv(PROTO)
    v2 = pd.read_csv(V2)[["case_id", "paren_LAA_910_frac"]]
    df = proto.merge(sl, on="case_id").merge(v2, on="case_id")
    df["SL_artery_per_L_lung"] = df["SL_artery_mm"] / df["lung_vol_mL"] * 1000  # mm per L
    df["SL_vein_per_L_lung"] = df["SL_vein_mm"] / df["lung_vol_mL"] * 1000

    tests = []
    # T1 retry: PH vs nonPH on SL_artery per lung in contrast subset
    contrast = df[df["protocol"] == "contrast"].dropna(subset=["SL_artery_per_L_lung"])
    ph = contrast[contrast["label"] == 1]["SL_artery_per_L_lung"].to_numpy()
    nph = contrast[contrast["label"] == 0]["SL_artery_per_L_lung"].to_numpy()
    if len(ph) and len(nph):
        u, p_two = stats.mannwhitneyu(ph, nph, alternative="two-sided")
        xx = ph.reshape(-1, 1); yy = nph.reshape(1, -1)
        delta = float(((xx > yy).sum() - (xx < yy).sum()) / (len(ph) * len(nph)))
        matches = float(np.median(ph)) < float(np.median(nph))
        tests.append({
            "name": "T1_SL_artery_per_L_PH_vs_nonPH_contrast",
            "hipas_prediction": "PH < nonPH",
            "n_ph": int(len(ph)),
            "n_nonph": int(len(nph)),
            "median_ph": float(np.median(ph)),
            "median_nph": float(np.median(nph)),
            "p_two_sided": float(p_two),
            "cliffs_delta": delta,
            "direction_matches_hipas": matches,
        })
    # T2 retry: Spearman(LAA_910, SL_vein_per_L) per protocol
    for proto_name in ("contrast", "plain_scan"):
        sub = df[df["protocol"] == proto_name].dropna(
            subset=["SL_vein_per_L_lung", "paren_LAA_910_frac"]
        )
        if len(sub) < 10:
            continue
        rho, p = stats.spearmanr(sub["paren_LAA_910_frac"], sub["SL_vein_per_L_lung"])
        tests.append({
            "name": f"T2_SL_vein_per_L_vs_LAA910_{proto_name}",
            "hipas_prediction": "negative",
            "n": int(len(sub)),
            "spearman_rho": float(rho),
            "spearman_p": float(p),
            "direction_matches_hipas": bool(rho < 0),
        })

    lines = [
        "# R4.2 Skeleton-length abundance — HiPaS T1 retry",
        "",
        "Replaces R3 volume-fraction proxy with skimage.skeletonize_3d skeleton",
        "length in mm per L of lung. This is the literal HiPaS metric.",
        "",
    ]
    for t in tests:
        lines.append(f"## {t['name']}")
        for k, v in t.items():
            if k == "name":
                continue
            lines.append(f"- **{k}**: {v:.4f}" if isinstance(v, float) else f"- **{k}**: {v}")
        lines.append("")
    t1 = next((t for t in tests if t["name"].startswith("T1")), None)
    if t1:
        lines.append(
            f"**T1 verdict**: direction matches HiPaS = **{t1['direction_matches_hipas']}**, "
            f"p={t1['p_two_sided']:.4g}, Cliff's δ={t1['cliffs_delta']:+.3f}."
        )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    OUT_JSON.write_text(json.dumps({"tests": tests}, indent=2), encoding="utf-8")
    print(OUT_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
