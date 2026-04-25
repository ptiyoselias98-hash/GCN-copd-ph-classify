"""R20.H — Verify R18.B legacy ρ=-0.767 reproduces in unified-301 pipeline.

Once R20.G produces morph_unified301.csv (legacy 199 contrast Simple_AV_seg
re-segmentation + new100 plain-scan Simple_AV_seg), compute Spearman ρ for
artery_len_p25 vs mPAP across the 5-stage cohort (mpap_lookup_gold.json
for PH; default 5.0 for plain-scan; default 15.0 for contrast nonPH).

Closes R18 must-fix #2 with positive verdict if ρ ≤ -0.5 in unified pipeline.

Output: outputs/r20/unified_verify_r18b.{json,md}
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent.parent
MORPH = ROOT / "outputs" / "r20" / "morph_unified301.csv"
LABELS = ROOT / "data" / "labels_extended_382.csv"
PROTO = ROOT / "data" / "case_protocol_extended.csv"
MPAP = ROOT / "data" / "mpap_lookup_gold.json"
OUT = ROOT / "outputs" / "r20"


def main():
    if not MORPH.exists():
        print(f"WAIT: {MORPH} not yet present (R20.G running)")
        return
    morph = pd.read_csv(MORPH)
    lab = pd.read_csv(LABELS)
    pro = pd.read_csv(PROTO)
    mp = json.loads(MPAP.read_text(encoding="utf-8"))

    df = morph.merge(lab, on="case_id", how="left", suffixes=("", "_lab"))
    df = df.merge(pro[["case_id", "protocol"]], on="case_id", how="left")
    df["mpap"] = df["case_id"].map(mp)
    plain = df["protocol"].str.lower() == "plain_scan"
    contrast_nonph = (df["protocol"].str.lower() == "contrast") & (df["label"] == 0)
    df.loc[plain & df["mpap"].isna(), "mpap"] = 5.0
    df.loc[contrast_nonph & df["mpap"].isna(), "mpap"] = 15.0
    print(f"unified-301 cohort: n={len(df)}")
    print(f"  PH (mpap-resolved): {(df['label']==1).sum()}")
    print(f"  nonPH plain (default mpap=5): {plain.sum()}")
    print(f"  nonPH contrast (default mpap=15): {contrast_nonph.sum()}")
    print(f"  mpap-NaN: {df['mpap'].isna().sum()}")

    flagship = ["artery_len_p25", "artery_len_p50", "artery_tort_p10",
                "vein_len_p25"]
    out = {"n_cohort": int(len(df)), "spearman_results": {}}
    md_lines = ["# R20.H — Unified-pipeline R18.B verification", "",
                f"Cohort: n={len(df)} (Simple_AV_seg unified across legacy contrast 199 + new100 100)", ""]
    md_lines += ["| feature | n_resolved | Spearman ρ | p | n_PH | n_nonPH |",
                 "|---|---|---|---|---|---|"]
    for feat in flagship:
        if feat not in df.columns:
            md_lines.append(f"| {feat} | ABSENT |")
            continue
        sub = df.dropna(subset=["mpap", feat])
        if len(sub) < 20:
            md_lines.append(f"| {feat} | n<20 SKIP |")
            continue
        rho, p = spearmanr(sub["mpap"], sub[feat])
        n_ph = int((sub["label"] == 1).sum())
        n_n = int((sub["label"] == 0).sum())
        out["spearman_results"][feat] = {
            "n": int(len(sub)), "rho": float(rho), "p": float(p),
            "n_ph": n_ph, "n_nonph": n_n,
        }
        md_lines.append(f"| {feat} | {len(sub)} | {rho:+.3f} | {p:.2g} | {n_ph} | {n_n} |")
    legacy_artery_p25 = -0.767
    if "artery_len_p25" in out["spearman_results"]:
        new_rho = out["spearman_results"]["artery_len_p25"]["rho"]
        out["legacy_R18B_rho"] = legacy_artery_p25
        out["unified_R20H_rho"] = new_rho
        out["delta_rho"] = float(new_rho - legacy_artery_p25)
        out["sign_preserved"] = bool(new_rho < 0)
        out["effect_preserved_within_0_2"] = bool(abs(new_rho - legacy_artery_p25) < 0.2)

    md_lines += ["",
                 "## Verdict",
                 "",
                 f"Legacy R18.B (HiPaS pipeline, n=147 within-contrast): ρ = {legacy_artery_p25:+.3f}",
                 ""]
    if "artery_len_p25" in out["spearman_results"]:
        new_rho = out["spearman_results"]["artery_len_p25"]["rho"]
        if new_rho < -0.5:
            md_lines.append("**Closes R18 must-fix #2 with POSITIVE verdict**: "
                            f"unified-pipeline ρ = {new_rho:+.3f} preserves "
                            "the strong negative monotonic mPAP-vs-artery_len_p25 trend "
                            "across pipelines. Biology survives Simple_AV_seg "
                            "re-segmentation.")
        else:
            md_lines.append(f"**Closes R18 must-fix #2 with NEGATIVE/PARTIAL "
                            f"verdict**: unified-pipeline ρ = {new_rho:+.3f} "
                            f"DOES NOT reproduce legacy ρ={legacy_artery_p25:+.3f}. "
                            "Either (a) original finding was pipeline-specific "
                            "or (b) Simple_AV_seg artery segmentation differs "
                            "structurally from legacy HiPaS in artery edge-length "
                            "distribution. Honest negative.")

    (OUT / "unified_verify_r18b.json").write_text(json.dumps(out, indent=2),
                                                    encoding="utf-8")
    (OUT / "unified_verify_r18b.md").write_text("\n".join(md_lines),
                                                  encoding="utf-8")
    print(f"saved → {OUT}/unified_verify_r18b.{{json,md}}")


if __name__ == "__main__":
    main()
