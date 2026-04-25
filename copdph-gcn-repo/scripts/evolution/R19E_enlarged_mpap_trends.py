"""R19.E — mPAP 5-stage trends on the ENLARGED 382-cohort.

Re-runs R18.B Spearman + Jonckheere-Terpstra trends with the new100
cases included. Compare ρ on enlarged vs legacy-only cohort.

Output: outputs/r19/enlarged_mpap_trends.{json,md}
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent.parent
OUT = ROOT / "outputs" / "r19"
MORPH = OUT / "per_structure_morphometrics_extended.csv"
LUNG = ROOT / "outputs" / "lung_features_v2.csv"
LUNG_NEW = OUT.parent / "r15" / "lung_features_new100.csv"
LABELS = ROOT / "data" / "labels_extended_382.csv"
PROTO = ROOT / "data" / "case_protocol_extended.csv"
SEG_FAILS = ROOT / "outputs" / "r13" / "seg_failures_real.json"
MPAP = ROOT / "data" / "mpap_lookup_gold.json"


def jonckheere_terpstra(samples):
    k = len(samples); n = sum(len(s) for s in samples)
    if k < 2 or n < k * 2: return 0.0, 0.0, 1.0
    J = 0
    for i in range(k):
        for j in range(i + 1, k):
            xi, xj = samples[i], samples[j]
            for x in xi:
                J += int((xj > x).sum()) + 0.5 * int((xj == x).sum())
    n_arr = np.array([len(s) for s in samples], float)
    mu = (n*n - (n_arr*n_arr).sum()) / 4.0
    sigma2 = (n*n*(2*n + 3) - (n_arr*n_arr*(2*n_arr + 3)).sum()) / 72.0
    if sigma2 <= 0: return float(J), 0.0, 1.0
    z = (J - mu) / np.sqrt(sigma2)
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(J), float(z), float(p)


def main():
    morph = pd.read_csv(MORPH)
    leg_lung = pd.read_csv(LUNG)
    new_lung = pd.read_csv(LUNG_NEW) if LUNG_NEW.exists() else None
    if new_lung is not None:
        new_lung["is_new"] = 1; leg_lung["is_new"] = 0
        common = set(leg_lung.columns) & set(new_lung.columns)
        lung = pd.concat([leg_lung[list(common)], new_lung[list(common)]], ignore_index=True)
    else:
        lung = leg_lung
    lab = pd.read_csv(LABELS); pro = pd.read_csv(PROTO)
    df = lab.merge(pro[["case_id", "protocol"]], on="case_id") \
        .merge(morph, on="case_id", how="left", suffixes=("", "_dup1")) \
        .merge(lung, on="case_id", how="left", suffixes=("", "_dup2"))
    fails = set()
    if SEG_FAILS.exists():
        sf = json.loads(SEG_FAILS.read_text(encoding="utf-8"))
        fails = {r["case_id"] for r in sf.get("real_fails", []) + sf.get("lung_anomaly", [])}
    df = df[~df["case_id"].isin(fails)].copy()
    mpap = json.loads(MPAP.read_text(encoding="utf-8"))
    df["mpap"] = df["case_id"].map(mpap)
    df.loc[df["protocol"].str.lower() == "plain_scan", "mpap"] = 5.0
    df.loc[(df["label"] == 0) & (df["protocol"].str.lower() == "contrast"), "mpap"] = 15.0
    df["stage"] = -1
    df.loc[df["protocol"].str.lower() == "plain_scan", "stage"] = 0
    df.loc[(df["label"] == 0) & (df["protocol"].str.lower() == "contrast"), "stage"] = 1
    df.loc[(df["label"] == 1) & (df["mpap"] < 25), "stage"] = 2
    df.loc[(df["label"] == 1) & (df["mpap"] >= 25) & (df["mpap"] < 35), "stage"] = 3
    df.loc[(df["label"] == 1) & (df["mpap"] >= 35), "stage"] = 4
    df = df[df["stage"] >= 0].copy()

    print(f"Stage counts:")
    print(df["stage"].value_counts().sort_index().to_string())

    target_features = [c for c in [
        "artery_tort_p10", "artery_len_p10", "artery_len_p25", "artery_len_p50",
        "artery_len_p75", "artery_len_p90", "artery_len_mean", "artery_total_len_mm",
        "vein_tort_p10", "vein_len_p25", "vein_len_p50", "vein_total_len_mm",
        "paren_std_HU", "paren_mean_HU", "paren_LAA_950_frac",
        "lung_vol_mL", "apical_basal_LAA950_gradient",
    ] if c in df.columns]
    out = {"n_total": int(len(df)),
           "stage_counts": {int(k): int(v) for k, v in df["stage"].value_counts().sort_index().items()},
           "trend_tests": {}}
    for col in target_features:
        sub = df.dropna(subset=[col, "stage"])
        if len(sub) < 30: continue
        rho, p_s = spearmanr(sub["stage"], sub[col])
        samples = [sub.loc[sub["stage"] == s, col].dropna().values for s in range(5)]
        samples = [x for x in samples if len(x) >= 5]
        J, z, p_jt = jonckheere_terpstra(samples) if len(samples) >= 2 else (0, 0, 1.0)
        out["trend_tests"][col] = {
            "n": int(len(sub)),
            "spearman_rho": float(rho), "spearman_p": float(p_s),
            "jonckheere_z": float(z), "jonckheere_p": float(p_jt),
        }
    (OUT / "enlarged_mpap_trends.json").write_text(json.dumps(out, indent=2),
                                                      encoding="utf-8")

    md = ["# R19.E — Enlarged 360-cohort mPAP 5-stage trends",
          "",
          f"Cohort n = {out['n_total']} (vs R18.B 282 cohort).",
          "Stage counts:",
          ""]
    md.append("| stage | n |")
    md.append("|---|---|")
    for s, n in out['stage_counts'].items():
        md.append(f"| {s} | {n} |")
    md += ["",
           "## Trend tests (sorted by |Spearman ρ|)",
           "",
           "| feature | n | Spearman ρ | p_spearman | Jonckheere z | p_JT |",
           "|---|---|---|---|---|---|"]
    sorted_feats = sorted(out["trend_tests"].items(),
                           key=lambda kv: -abs(kv[1]["spearman_rho"]))
    for col, r in sorted_feats:
        md.append(f"| {col} | {r['n']} | {r['spearman_rho']:+.3f} | "
                  f"{r['spearman_p']:.3g} | {r['jonckheere_z']:+.3f} | "
                  f"{r['jonckheere_p']:.3g} |")
    md += ["",
           "## Comparison to R18.B (282-cohort)",
           "",
           "R18.B at n=282 reported (top trends):",
           "- artery_len_p25 ρ=−0.767 p=9e-30",
           "- artery_len_p50 ρ=−0.753",
           "- paren_std_HU ρ=+0.629",
           "- artery_tort_p10 ρ=−0.619",
           "",
           "Enlarged cohort tests above. Trends should be at least as strong",
           "(more nonPH-plain Stage 0 cases tighten the lower-mPAP anchor).",
           ""]
    (OUT / "enlarged_mpap_trends.md").write_text("\n".join(md), encoding="utf-8")
    print(f"saved → {OUT}/enlarged_mpap_trends.{{json,md}}")
    print(f"\nTop 5 trends:")
    for col, r in sorted_feats[:5]:
        print(f"  {col}: ρ={r['spearman_rho']:+.3f} p={r['spearman_p']:.3g} (n={r['n']})")


if __name__ == "__main__":
    main()
