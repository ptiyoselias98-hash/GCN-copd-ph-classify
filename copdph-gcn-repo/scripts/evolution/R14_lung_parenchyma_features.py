"""R14.A — Pure lung parenchyma feature extraction for COPD→COPD-PH evolution.

Until now, lung masks were unused except for the global lung-features
proxy in §13. This script computes per-case parenchyma phenotypes by:

  1. Loading lung.nii.gz (HU-valued, -2048 = bg)
  2. Subtracting artery, vein, airway masks → PURE parenchyma voxels
  3. Computing per-case features that match the COPD/COPD-PH literature:
     - total_lung_vol_mL
     - LAA_-950_pct  (low-attenuation area, % vox with HU < -950 — emphysema proxy)
     - LAA_-910_pct  (mild emphysema; HU < -910)
     - mean_HU, sd_HU, skew_HU, kurt_HU (parenchyma density distribution)
     - left_right_asymmetry  (|V_L - V_R| / (V_L + V_R), Y-axis split)
     - lower_upper_ratio    (V_lower / V_upper, Z-axis split)
     - HAA_-700_pct  (high-attenuation area >-700 HU; ground-glass / fibrosis proxy)

Skip cases on `outputs/r13/seg_failures_real.json` exclusion list.

Outputs: outputs/r14/lung_parenchyma_features.{json,csv,md}
        outputs/r14/lung_parenchyma_phylo.md  (group-level summaries)
"""
from __future__ import annotations

import json
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib

ROOT = Path(__file__).parent.parent.parent
OUT = ROOT / "outputs" / "r14"
OUT.mkdir(parents=True, exist_ok=True)
NII = Path(r"E:\桌面文件\nii格式图\nii-unified-282")
PROTO = ROOT / "data" / "case_protocol.csv"
LABELS = ROOT / "data" / "labels_expanded_282.csv"
SEG_FAILS = ROOT / "outputs" / "r13" / "seg_failures_real.json"


def safe_skew_kurt(x: np.ndarray):
    if x.size < 4:
        return float("nan"), float("nan")
    mu = x.mean(); sd = x.std(ddof=0)
    if sd == 0:
        return 0.0, 0.0
    z = (x - mu) / sd
    return float((z ** 3).mean()), float((z ** 4).mean() - 3.0)


def extract_features(case_dir: Path) -> dict | None:
    rec: dict = {}
    lung_p = case_dir / "lung.nii.gz"
    if not lung_p.exists():
        return None
    img = nib.load(str(lung_p))
    arr = img.get_fdata(caching="unchanged")
    spacing = img.header.get_zooms()
    vox_mL = float(np.prod(spacing[:3])) / 1000.0  # mm³ → mL
    fg = arr > -2000
    if fg.sum() == 0:
        return {"status": "EMPTY_LUNG"}

    # Subtract vessels + airway from lung if available
    pure = fg.copy()
    for n in ("artery", "vein", "airway"):
        p = case_dir / f"{n}.nii.gz"
        if p.exists():
            try:
                m = nib.load(str(p)).get_fdata() > -2000
                if m.shape == fg.shape:
                    pure = pure & ~m
            except Exception:
                continue

    hu = arr[pure].astype(np.float32)
    if hu.size < 1000:
        return {"status": "PURE_PARENCHYMA_TOO_SMALL", "n_vox": int(hu.size)}

    # Volume (mL)
    rec["total_lung_vol_mL"] = float(fg.sum() * vox_mL)
    rec["pure_parenchyma_vol_mL"] = float(hu.size * vox_mL)
    # LAA / HAA percentages (emphysema / ground-glass proxies)
    rec["LAA_950_pct"] = float((hu < -950).mean() * 100)
    rec["LAA_910_pct"] = float((hu < -910).mean() * 100)
    rec["HAA_700_pct"] = float((hu > -700).mean() * 100)
    # HU distribution moments
    rec["mean_HU"] = float(hu.mean())
    rec["sd_HU"] = float(hu.std(ddof=0))
    sk, kt = safe_skew_kurt(hu)
    rec["skew_HU"] = sk
    rec["kurt_HU"] = kt
    # Left/right asymmetry (split along Y-axis midline)
    coords = np.argwhere(fg)
    if coords.size:
        ymid = coords[:, 1].mean()
        left = (coords[:, 1] < ymid).sum()
        right = coords.shape[0] - left
        rec["left_right_asymmetry"] = float(abs(left - right) / max(left + right, 1))
    # Upper/lower split along Z (axial)
    if coords.size:
        zmid = coords[:, 2].mean()
        lower = (coords[:, 2] < zmid).sum()
        upper = coords.shape[0] - lower
        rec["lower_upper_ratio"] = float(lower / max(upper, 1))
    rec["status"] = "ok"
    return rec


def main():
    proto = pd.read_csv(PROTO)
    labels = pd.read_csv(LABELS)
    df = labels.merge(proto[["case_id", "protocol"]], on="case_id", how="left")

    fails: set[str] = set()
    if SEG_FAILS.exists():
        sf = json.loads(SEG_FAILS.read_text(encoding="utf-8"))
        for r in sf.get("real_fails", []) + sf.get("lung_anomaly", []):
            fails.add(r["case_id"])

    rows = []
    failed = []
    for _, row in df.iterrows():
        cid = row["case_id"]
        if cid in fails:
            failed.append({"case_id": cid, "reason": "on_seg_failure_list"})
            continue
        case_dir = NII / cid
        if not case_dir.exists():
            failed.append({"case_id": cid, "reason": "no_nii_dir"})
            continue
        feats = extract_features(case_dir)
        if feats is None:
            failed.append({"case_id": cid, "reason": "no_lung_nii"})
            continue
        if feats.get("status") != "ok":
            failed.append({"case_id": cid, "reason": feats.get("status", "unknown")})
            continue
        feats["case_id"] = cid
        feats["label"] = int(row["label"])
        feats["protocol"] = str(row.get("protocol") or "")
        rows.append(feats)
        if len(rows) % 25 == 0:
            print(f"  ...{len(rows)} cases done")

    out_csv = OUT / "lung_parenchyma_features.csv"
    out_json = OUT / "lung_parenchyma_features.json"
    if rows:
        df_out = pd.DataFrame(rows)
        cols = ["case_id", "label", "protocol", "status",
                "total_lung_vol_mL", "pure_parenchyma_vol_mL",
                "LAA_950_pct", "LAA_910_pct", "HAA_700_pct",
                "mean_HU", "sd_HU", "skew_HU", "kurt_HU",
                "left_right_asymmetry", "lower_upper_ratio"]
        cols = [c for c in cols if c in df_out.columns]
        df_out[cols].to_csv(out_csv, index=False)

    out_json.write_text(json.dumps({
        "summary": {"n_extracted": len(rows), "n_failed": len(failed)},
        "rows": rows,
        "failed": failed,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nExtracted {len(rows)} cases; {len(failed)} skipped/failed.")

    # Group-level summary by (label, protocol)
    if rows:
        df_out = pd.DataFrame(rows)
        md = ["# R14.A — Pure lung parenchyma features by group",
              "",
              f"Extracted: {len(rows)} cases (excluded {len(failed)} on seg-failure list or missing NIfTI).",
              "",
              "## Group means ± SD",
              "",
              "| group (label/protocol) | n | total_lung_mL | LAA-950% | LAA-910% | mean_HU | sd_HU |",
              "|---|---|---|---|---|---|---|"]
        for (lbl, proto_), g in df_out.groupby(["label", "protocol"]):
            md.append(
                f"| label={lbl} / {proto_} | {len(g)} | "
                f"{g['total_lung_vol_mL'].mean():.0f} ± {g['total_lung_vol_mL'].std():.0f} | "
                f"{g['LAA_950_pct'].mean():.2f} ± {g['LAA_950_pct'].std():.2f} | "
                f"{g['LAA_910_pct'].mean():.2f} ± {g['LAA_910_pct'].std():.2f} | "
                f"{g['mean_HU'].mean():.1f} ± {g['mean_HU'].std():.1f} | "
                f"{g['sd_HU'].mean():.1f} ± {g['sd_HU'].std():.1f} |"
            )

        # Disease contrast — within-contrast only (protocol-balanced)
        md += ["",
               "## Within-contrast disease contrast (PH vs nonPH, both contrast-enhanced)",
               "",
               "Removes protocol confound. Compares 26-27 contrast nonPH vs ~160 contrast PH.",
               ""]
        contrast = df_out[df_out["protocol"].str.lower() == "contrast"]
        ph_c = contrast[contrast["label"] == 1]
        nph_c = contrast[contrast["label"] == 0]
        if len(ph_c) >= 5 and len(nph_c) >= 5:
            from scipy.stats import mannwhitneyu
            md.append("| feature | PH-contrast (μ±SD) | nonPH-contrast (μ±SD) | Δ (PH-nonPH) | MWU p |")
            md.append("|---|---|---|---|---|")
            for col in ["LAA_950_pct", "LAA_910_pct", "HAA_700_pct",
                        "mean_HU", "sd_HU", "skew_HU", "kurt_HU",
                        "total_lung_vol_mL", "left_right_asymmetry",
                        "lower_upper_ratio"]:
                a = ph_c[col].dropna().values
                b = nph_c[col].dropna().values
                if len(a) < 5 or len(b) < 5:
                    continue
                try:
                    u, p = mannwhitneyu(a, b, alternative="two-sided")
                    md.append(f"| {col} | {a.mean():.2f} ± {a.std():.2f} | "
                              f"{b.mean():.2f} ± {b.std():.2f} | "
                              f"{a.mean() - b.mean():+.2f} | {p:.3g} |")
                except Exception:
                    continue

        (OUT / "lung_parenchyma_phylo.md").write_text("\n".join(md), encoding="utf-8")
        print(f"Saved {OUT}/lung_parenchyma_phylo.md")


if __name__ == "__main__":
    main()
