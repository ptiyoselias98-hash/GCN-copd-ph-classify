"""R18.B — mPAP 5-stage evolution analysis.

User clinical input 2026-04-25: plain-scan COPDNOPH ≈ no PH = mPAP 0-10.
Combined with the 113-case PH cohort's mPAP from copd-ph患者113例0331.xlsx,
we get a 5-stage continuous evolution spectrum:

  Stage 0  plain-scan nonPH   (mPAP 0-10)    n ~158 plain-scan
  Stage 1  contrast nonPH     (10-20 borderline; treat as one bin)
  Stage 2  PH early           (mPAP 25-35)
  Stage 3  PH moderate        (mPAP 35-45)
  Stage 4  PH severe          (mPAP > 45)

Tests whether R17/R16 endotype features trend monotonically across stages
(Spearman ρ + Jonckheere-Terpstra trend test for ordered alternatives).

Key features tested: artery_tort_p10 (R17 d=-1.42 LARGEST), artery_len_p25,
vein_len_p25, paren_std_HU (R16 d=+1.10), paren_mean_HU, lung_vol_mL.

Output: outputs/r18/mpap_evolution.{json,md} + fig_r18_mpap_trajectories.png

Note: legacy mpap_lookup.json uses opaque hash keys (MD5 of patient_id).
For an honest first pass, treat the plain-scan default + binary PH/nonPH
flag from labels_extended_382.csv. PH cases without resolved mPAP get
binned uniformly across stages 2-4 (3 equal random buckets) as a baseline
proxy. R19 will re-do with resolved mPAP from the xlsx.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent
OUT = ROOT / "outputs" / "r18"
OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
LABELS = ROOT / "data" / "labels_extended_382.csv"
PROTO = ROOT / "data" / "case_protocol_extended.csv"
SEG_FAILS = ROOT / "outputs" / "r13" / "seg_failures_real.json"
MORPH = ROOT / "outputs" / "r17" / "per_structure_morphometrics.csv"
LUNG = ROOT / "outputs" / "lung_features_v2.csv"


def jonckheere_terpstra(samples):
    """Jonckheere-Terpstra trend test for ordered alternatives.
    samples: list of arrays, ordered by hypothesized trend direction.
    Returns: (J statistic, normal-approx z, two-sided p-value)."""
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
    lab = pd.read_csv(LABELS); pro = pd.read_csv(PROTO)
    df = lab.merge(pro[["case_id", "protocol"]], on="case_id", how="inner")
    fails = set()
    if SEG_FAILS.exists():
        sf = json.loads(SEG_FAILS.read_text(encoding="utf-8"))
        fails = {r["case_id"] for r in sf.get("real_fails", []) + sf.get("lung_anomaly", [])}
    df = df[~df["case_id"].isin(fails)].copy()

    # Stage assignment — REAL mPAP-based binning (was random proxy in R18.B v1)
    mpap_gold_path = ROOT / "data" / "mpap_lookup_gold.json"
    if not mpap_gold_path.exists():
        raise SystemExit(f"missing {mpap_gold_path} — fetch from remote first")
    mpap_lookup = json.loads(mpap_gold_path.read_text(encoding="utf-8"))
    df["mpap"] = df["case_id"].map(mpap_lookup)
    df["stage"] = -1
    # Stage 0: plain-scan nonPH (mPAP 0-10 default per user 2026-04-25)
    df.loc[(df["label"] == 0) & (df["protocol"].str.lower() == "plain_scan"), "stage"] = 0
    df.loc[(df["label"] == 0) & (df["protocol"].str.lower() == "plain_scan"), "mpap"] = 5.0
    # Stage 1: contrast nonPH (mPAP 10-20 borderline default)
    df.loc[(df["label"] == 0) & (df["protocol"].str.lower() == "contrast"), "stage"] = 1
    df.loc[(df["label"] == 0) & (df["protocol"].str.lower() == "contrast"), "mpap"] = 15.0
    # Stage 2/3/4: PH cases binned by REAL mPAP from mpap_lookup_gold
    df.loc[(df["label"] == 1) & (df["mpap"] < 25), "stage"] = 2
    df.loc[(df["label"] == 1) & (df["mpap"] >= 25) & (df["mpap"] < 35), "stage"] = 3
    df.loc[(df["label"] == 1) & (df["mpap"] >= 35), "stage"] = 4
    # PH without resolved mPAP → drop from analysis
    df.loc[(df["label"] == 1) & (df["mpap"].isna()), "stage"] = -1

    print(f"stage counts:")
    print(df["stage"].value_counts().sort_index().to_string())

    # Merge feature CSVs
    morph = pd.read_csv(MORPH); lung = pd.read_csv(LUNG)
    feature_panel = df.merge(morph, on="case_id", how="left", suffixes=("", "_dup")) \
        .merge(lung, on="case_id", how="left", suffixes=("", "_dup2"))

    target_features = [
        "artery_tort_p10", "artery_len_p25", "artery_len_p50",
        "vein_len_p25", "vein_tort_p10",
        "paren_std_HU", "paren_mean_HU", "paren_LAA_950_frac",
        "lung_vol_mL", "apical_basal_LAA950_gradient",
    ]
    target_features = [c for c in target_features if c in feature_panel.columns]

    out = {"stage_counts": {int(k): int(v) for k, v in
                              df["stage"].value_counts().sort_index().items()},
           "stage_definition": {
               "0": "plain-scan nonPH (mPAP 0-10 default per user 2026-04-25, assigned 5.0)",
               "1": "contrast nonPH (mPAP 10-20 borderline default, assigned 15.0)",
               "2": "PH borderline (real mPAP <25 from mpap_lookup_gold)",
               "3": "PH early-moderate (real mPAP 25-35)",
               "4": "PH moderate-severe (real mPAP >=35)",
               "note": "Stages 2-4 use REAL mPAP from mpap_lookup_gold.json (R18.C resolved 106/113 PH cases). 7 PH without mPAP excluded.",
           },
           "trend_tests": {}}

    for col in target_features:
        sub = feature_panel.dropna(subset=[col, "stage"])
        sub = sub[sub["stage"] >= 0]
        if len(sub) < 30: continue
        rho, p_spearman = spearmanr(sub["stage"], sub[col])
        samples = [sub.loc[sub["stage"] == s, col].dropna().values for s in range(5)]
        samples = [x for x in samples if len(x) >= 5]
        if len(samples) >= 2:
            J, z, p_jt = jonckheere_terpstra(samples)
        else:
            J, z, p_jt = 0.0, 0.0, 1.0
        # Stage-wise mean ± SE
        stage_summary = []
        for s in range(5):
            x = sub.loc[sub["stage"] == s, col].dropna().values
            if len(x) >= 3:
                stage_summary.append({
                    "stage": s, "n": int(len(x)),
                    "mean": float(x.mean()),
                    "se": float(x.std(ddof=1) / np.sqrt(len(x))),
                    "ci95_low": float(x.mean() - 1.96 * x.std(ddof=1) / np.sqrt(len(x))),
                    "ci95_high": float(x.mean() + 1.96 * x.std(ddof=1) / np.sqrt(len(x))),
                })
        out["trend_tests"][col] = {
            "spearman_rho": float(rho), "spearman_p": float(p_spearman),
            "jonckheere_z": z, "jonckheere_p": p_jt,
            "n_stages_with_data": len(stage_summary),
            "stage_summary": stage_summary,
        }

    (OUT / "mpap_evolution.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    # MD report
    md = ["# R18.B — mPAP 5-stage evolution analysis (REAL mPAP, R18.C resolved)",
          "",
          "**User clinical input 2026-04-25**: plain-scan COPDNOPH ≈ no PH = mPAP 0-10.",
          "**R18.C resolution**: mpap_lookup_gold.json provides case_id-keyed mPAP for 106 PH cases.",
          "",
          "Stages:",
          f"- 0=plain-scan nonPH (mPAP 0-10 default, n={out['stage_counts'].get(0,0)})",
          f"- 1=contrast nonPH (mPAP 10-20 borderline, n={out['stage_counts'].get(1,0)})",
          f"- 2=PH borderline (real mPAP <25, n={out['stage_counts'].get(2,0)})",
          f"- 3=PH early-moderate (real mPAP 25-35, n={out['stage_counts'].get(3,0)})",
          f"- 4=PH moderate-severe (real mPAP ≥35, n={out['stage_counts'].get(4,0)})",
          "",
          "Excluded: PH cases without resolved mPAP (typically because patient_sn not in mpap_lookup_gold).",
          "",
          "## Trend tests across 5 ordered stages",
          "",
          "| feature | Spearman ρ | p_spearman | Jonckheere z | p_JT |",
          "|---|---|---|---|---|"]
    for col, r in out["trend_tests"].items():
        md.append(f"| {col} | {r['spearman_rho']:+.3f} | {r['spearman_p']:.3g} | "
                  f"{r['jonckheere_z']:+.3f} | {r['jonckheere_p']:.3g} |")

    md += ["",
           "## Stage-wise means (mean ± 1.96·SE)",
           ""]
    for col, r in out["trend_tests"].items():
        md.append(f"### {col}")
        md.append("")
        md.append("| stage | n | mean | 95% CI |")
        md.append("|---|---|---|---|")
        for s in r["stage_summary"]:
            md.append(f"| {s['stage']} | {s['n']} | {s['mean']:.3f} | "
                      f"[{s['ci95_low']:.3f}, {s['ci95_high']:.3f}] |")
        md.append("")

    md += ["## Caveats",
           "",
           "1. PH cases without resolved mPAP (in xlsx 113 cases minus 106 resolved + "
           "those not in extended cohort) are EXCLUDED from analysis.",
           "2. Stage 1 (contrast nonPH n=27) is small.",
           "3. Stage 2 (PH borderline mPAP<25, n="
           + str(out['stage_counts'].get(2,0)) + ") is the smallest PH bin.",
           "4. Spearman/Jonckheere are NON-PARAMETRIC trend tests for ordered "
           "alternatives — robust to non-normal distributions.",
           ""]
    (OUT / "mpap_evolution.md").write_text("\n".join(md), encoding="utf-8")

    # Trajectory figure
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    plot_features = [c for c in target_features if c in out["trend_tests"]][:10]
    for ax, col in zip(axes.flat, plot_features):
        r = out["trend_tests"][col]
        stages_x = [s["stage"] for s in r["stage_summary"]]
        means = [s["mean"] for s in r["stage_summary"]]
        ses = [s["se"] for s in r["stage_summary"]]
        ax.errorbar(stages_x, means, yerr=[1.96*s for s in ses], fmt="o-",
                    capsize=4, c="#10b981", linewidth=1.8, markersize=8)
        ax.set_xlabel("mPAP stage (0=plain → 4=severe PH)")
        ax.set_ylabel(col, fontsize=9)
        ax.set_xticks(range(5))
        ax.grid(alpha=0.3)
        sig = "✓" if r["jonckheere_p"] < 0.05 else ""
        ax.set_title(f"{col}\nρ={r['spearman_rho']:+.2f} p={r['spearman_p']:.3g} JTp={r['jonckheere_p']:.3g} {sig}",
                     fontsize=9)
    for ax in axes.flat[len(plot_features):]: ax.set_visible(False)
    plt.suptitle("R18.B — Stage-wise endotype trajectories (PROXY mPAP — random PH split)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(FIG / "fig_r18_mpap_trajectories.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"saved → {OUT}/mpap_evolution.{{json,md}} + fig_r18_mpap_trajectories.png")


if __name__ == "__main__":
    main()
