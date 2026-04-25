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
    # Restrict to CONTRAST-ONLY stratum to match legacy R18.B design.
    # Mixed-protocol analysis is the pipeline-mixing artifact R19.E warned about.
    contrast_mask = ~plain
    df_orig = df.copy()
    df = df[contrast_mask].copy()
    print(f"\n=== CONTRAST-ONLY stratum (Simple_AV_seg unified) ===")
    print(f"contrast cohort n={len(df)} (PH={(df['label']==1).sum()} "
          f"nonPH={(df['label']==0).sum()})")
    # Cohen's d (PH vs nonPH) within contrast
    from scipy.stats import mannwhitneyu
    out_d = {}
    for feat in flagship:
        if feat not in df.columns: continue
        ph = df.loc[df["label"]==1, feat].dropna().values
        nonph = df.loc[df["label"]==0, feat].dropna().values
        if len(ph) >= 3 and len(nonph) >= 3:
            pooled = np.sqrt(((len(ph)-1)*ph.std(ddof=1)**2 +
                              (len(nonph)-1)*nonph.std(ddof=1)**2) /
                             (len(ph)+len(nonph)-2))
            d = (ph.mean() - nonph.mean()) / pooled if pooled > 0 else 0
            u, p = mannwhitneyu(ph, nonph, alternative="two-sided")
            out_d[feat] = {"cohen_d": float(d), "mwu_p": float(p),
                            "n_ph": len(ph), "n_nonph": len(nonph),
                            "ph_mean": float(ph.mean()),
                            "nonph_mean": float(nonph.mean())}
    out = {"n_cohort": int(len(df)), "spearman_results": {},
           "contrast_only_PH_vs_nonPH": out_d}
    md_lines = ["# R20.H — Unified-pipeline R18.B verification", "",
                f"Cohort (CONTRAST-ONLY): n={len(df)} (Simple_AV_seg legacy contrast 163 PH + 27 nonPH)",
                f"(Full unified-301 cohort had n={len(df_orig)}; restricted to "
                "contrast to avoid plain-scan vs contrast pipeline-mixing artifact "
                "from R19.E)",
                "",
                "## Contrast-only PH vs nonPH (within-pipeline within-protocol)",
                "",
                "| feature | Cohen's d | MWU p | PH mean | nonPH mean | n_PH | n_nonPH |",
                "|---|---|---|---|---|---|---|"]
    for feat, e in out_d.items():
        md_lines.append(f"| {feat} | {e['cohen_d']:+.3f} | {e['mwu_p']:.2g} | "
                        f"{e['ph_mean']:.3f} | {e['nonph_mean']:.3f} | "
                        f"{e['n_ph']} | {e['n_nonph']} |")
    md_lines += ["", ""]
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
        ph_d_signs = [out_d[f]["cohen_d"] < 0 for f in flagship if f in out_d]
        all_ph_lower = all(ph_d_signs) if ph_d_signs else False
        # All flagship features still show PH < nonPH in unified pipeline
        if new_rho < -0.5 and all_ph_lower:
            md_lines.append("**Closes R18 must-fix #2 with POSITIVE verdict**: "
                            f"unified-pipeline ρ = {new_rho:+.3f} preserves "
                            "the strong negative monotonic mPAP-vs-artery_len_p25 trend "
                            "across pipelines.")
        elif all_ph_lower:
            md_lines.append("**Closes R18 must-fix #2 with PARTIAL/MIXED verdict**: "
                            "DIRECTION preserved across pipelines (all 4 flagship "
                            "features show PH < nonPH within Simple_AV_seg unified "
                            "contrast cohort, "
                            f"artery_len_p25 d={out_d.get('artery_len_p25',{}).get('cohen_d','NA'):+.3f}, "
                            f"artery_len_p50 d={out_d.get('artery_len_p50',{}).get('cohen_d','NA'):+.3f}, "
                            f"vein_len_p25 d={out_d.get('vein_len_p25',{}).get('cohen_d','NA'):+.3f}); "
                            f"BUT magnitude reduced (ρ {new_rho:+.3f} unified "
                            f"vs {legacy_artery_p25:+.3f} legacy). The vascular remodeling "
                            "signal is pipeline-independent in DIRECTION but the legacy "
                            "ρ=-0.767 effect SIZE was Pipeline-specific (HiPaS-style masks).")
        else:
            md_lines.append(f"**Closes R18 must-fix #2 with NEGATIVE verdict**: "
                            "neither direction nor magnitude of legacy R18.B finding "
                            f"reproduces in unified pipeline (ρ={new_rho:+.3f}). "
                            "Honest negative — legacy claim must be retracted.")

    (OUT / "unified_verify_r18b.json").write_text(json.dumps(out, indent=2),
                                                    encoding="utf-8")
    (OUT / "unified_verify_r18b.md").write_text("\n".join(md_lines),
                                                  encoding="utf-8")
    print(f"saved → {OUT}/unified_verify_r18b.{{json,md}}")


if __name__ == "__main__":
    main()
