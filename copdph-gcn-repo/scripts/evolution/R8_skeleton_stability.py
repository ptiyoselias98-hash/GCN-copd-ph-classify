"""R8.2 — Skeleton stability sweep (TEASAR proxy).

kimimaro is remote-only so we can't do a true TEASAR parameter sweep locally.
As an *anatomical QC proxy* we run `skimage.morphology.skeletonize` (which
uses a different 3D skeletonization — Lee94) on the same artery masks with
pre-erosion of 0, 1, 2 voxels. Pre-erosion is a proxy for TEASAR `scale`
parameter (smaller scale = finer skeleton, lower erosion = finer skeleton).

Per-case: skeleton length in voxels for each erosion level.
Report CV (std / mean) across erosion levels per case — a high CV indicates
a skeleton metric that is sensitive to mask boundaries and therefore to
TEASAR parameter choices.

Sample: 10 random cases (5 PH contrast + 5 nonPH mixed).
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion
from skimage.morphology import skeletonize as skeletonize_3d

ROOT = Path(__file__).parent.parent.parent
PROTO = ROOT / "data" / "case_protocol.csv"
V2 = ROOT / "outputs" / "lung_features_v2.csv"
NII = Path("E:/桌面文件/nii格式图/nii-unified-282")
FALLBACK = Path("E:/桌面文件/nii格式图/nii")
OUT_MD = ROOT / "outputs" / "r8" / "R8_skeleton_stability.md"
OUT_JSON = ROOT / "outputs" / "r8" / "R8_skeleton_stability.json"
OUT_MD.parent.mkdir(parents=True, exist_ok=True)

BG = -2048


def resolve(cid):
    p = NII / cid
    if (p / "lung.nii.gz").exists():
        return p
    f = FALLBACK / cid
    if (f / "lung.nii.gz").exists():
        return f
    src = p / "_source.txt"
    if src.exists():
        t = src.read_text(encoding="utf-8", errors="ignore").strip()
        return Path(t.replace("\\", "/"))
    return p


def load_artery(cid):
    cd = resolve(cid)
    fp = cd / "artery.nii.gz"
    if not fp.exists():
        return None, None
    img = nib.load(str(fp))
    arr = np.asarray(img.get_fdata(dtype=np.float32))
    slab = arr[::32, ::32, ::32]
    if (slab == BG).any():
        mask = arr != BG
    else:
        mask = arr > 0
    zooms = img.header.get_zooms()
    step_mm = float(np.cbrt(zooms[0] * zooms[1] * zooms[2]))
    return mask, step_mm


def main():
    import pandas as pd
    proto = pd.read_csv(PROTO)
    v2 = pd.read_csv(V2)
    df = proto.merge(v2[["case_id", "artery_placeholder"]], on="case_id")
    valid = df[df["artery_placeholder"] == 0]
    rng = random.Random(20260424)
    picks = []
    for (lab, prot), n in [((1, "contrast"), 5), ((0, "contrast"), 2), ((0, "plain_scan"), 3)]:
        pool = valid[(valid["label"] == lab) & (valid["protocol"] == prot)]["case_id"].tolist()
        rng.shuffle(pool)
        picks += pool[:n]
    print(f"Picks: {len(picks)} cases")

    sweep_results = []
    for cid in picks:
        mask, step = load_artery(cid)
        if mask is None:
            continue
        row = {"case_id": cid, "step_mm": step, "mask_vox": int(mask.sum())}
        lengths = []
        for er in (0, 1, 2):
            m = mask
            if er:
                m = binary_erosion(m, iterations=er)
            if not m.any():
                row[f"SL_er{er}_vox"] = 0
                row[f"SL_er{er}_mm"] = 0.0
                continue
            skel = skeletonize_3d(m)
            n = int(skel.sum())
            row[f"SL_er{er}_vox"] = n
            row[f"SL_er{er}_mm"] = n * step
            lengths.append(n * step)
        if len(lengths) > 1 and np.mean(lengths) > 0:
            row["SL_cv"] = float(np.std(lengths) / np.mean(lengths))
            row["SL_er1_minus_er0_pct"] = (lengths[1] - lengths[0]) / lengths[0] * 100
            row["SL_er2_minus_er0_pct"] = (lengths[2] - lengths[0]) / lengths[0] * 100
        sweep_results.append(row)
        print(f"  {cid[:40]} erosion0={lengths[0]:.0f}mm 1={lengths[1]:.0f} 2={lengths[2]:.0f} CV={row.get('SL_cv', float('nan')):.3f}")

    # Summary
    cvs = [r["SL_cv"] for r in sweep_results if "SL_cv" in r]
    summary = {
        "n_cases": len(sweep_results),
        "mean_CV": float(np.mean(cvs)) if cvs else float("nan"),
        "max_CV": float(np.max(cvs)) if cvs else float("nan"),
        "cases": sweep_results,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# R8.2 — Skeleton stability under pre-erosion (TEASAR proxy)",
        "",
        "True TEASAR parameter sweep requires kimimaro which is remote-only. As a",
        "local proxy we apply skimage.skeletonize_3d (Lee94) to artery masks with",
        "increasing binary erosion (0 / 1 / 2 voxels before skeletonization). Higher",
        "erosion → coarser skeleton → mimics larger TEASAR `scale` parameter.",
        "",
        f"Cases: {len(sweep_results)} (balanced across label × protocol).",
        f"Mean coefficient of variation across erosion levels: **{summary['mean_CV']:.3f}**",
        f"Max CV: **{summary['max_CV']:.3f}**",
        "",
        "| Case | er0 SL (mm) | er1 SL (mm) | er2 SL (mm) | CV | Δ er1 (%) | Δ er2 (%) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sweep_results:
        lines.append(
            f"| `{r['case_id'][:35]}` | {r.get('SL_er0_mm', 0):.0f} | "
            f"{r.get('SL_er1_mm', 0):.0f} | {r.get('SL_er2_mm', 0):.0f} | "
            f"{r.get('SL_cv', float('nan')):.3f} | "
            f"{r.get('SL_er1_minus_er0_pct', float('nan')):+.0f}% | "
            f"{r.get('SL_er2_minus_er0_pct', float('nan')):+.0f}% |"
        )
    lines += [
        "",
        "## Reading",
        "",
        f"Mean CV = {summary['mean_CV']:.3f}, max = {summary['max_CV']:.3f}. Values < 0.2",
        "indicate low sensitivity to mask-boundary perturbations, which is a",
        "necessary (not sufficient) condition for TEASAR-parameter stability.",
        "",
        "**Caveat**: this is NOT a true TEASAR `scale × const × pdrf_exp` sweep —",
        "that requires kimimaro on the remote and is queued for a dedicated",
        "Round 9 run once GPU contention clears (kimimaro is CPU-only but remote",
        "is currently saturated with another user's dinomaly training).",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
