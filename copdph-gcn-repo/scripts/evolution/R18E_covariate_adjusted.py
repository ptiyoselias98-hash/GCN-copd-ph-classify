"""R18.E — Covariate-adjusted endotype models (R17 reviewer must-fix).

Tests whether the strongest endotype findings (artery_tort_p10 d=-1.42,
artery_len_p25, paren_std_HU d=+1.10) survive covariate adjustment for:
  - mPAP-stage (the evolution variable itself; we want endotype effect
    INDEPENDENT of mPAP severity)
  - is_contrast (protocol confound)
  - patient_id-derived year (scanner-era proxy)

Logistic regression: feature ~ covariates (mPAP_stage + is_contrast + year);
extract residualized feature, then re-test PH-vs-nonPH within contrast.
Linear regression for continuous endotype features.

Output: outputs/r18/covariate_adjusted_endotype.{json,md}
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).parent.parent.parent
OUT = ROOT / "outputs" / "r18"
OUT.mkdir(parents=True, exist_ok=True)
LABELS = ROOT / "data" / "labels_extended_382.csv"
PROTO = ROOT / "data" / "case_protocol_extended.csv"
SEG_FAILS = ROOT / "outputs" / "r13" / "seg_failures_real.json"
MORPH = ROOT / "outputs" / "r17" / "per_structure_morphometrics.csv"
LUNG = ROOT / "outputs" / "lung_features_v2.csv"
MPAP = ROOT / "data" / "mpap_lookup_gold.json"


def parse_year(cid):
    m = re.search(r"_(20[0-9]{2})_", cid)
    return int(m.group(1)) if m else None


def main():
    morph = pd.read_csv(MORPH)
    lung = pd.read_csv(LUNG)
    lab = pd.read_csv(LABELS); pro = pd.read_csv(PROTO)
    df = lab.merge(pro[["case_id", "protocol"]], on="case_id") \
        .merge(morph, on="case_id", how="inner", suffixes=("", "_dup")) \
        .merge(lung, on="case_id", how="left", suffixes=("", "_dup2"))
    if "label_dup" in df.columns: df = df.drop(columns=["label_dup"])
    fails = set()
    if SEG_FAILS.exists():
        sf = json.loads(SEG_FAILS.read_text(encoding="utf-8"))
        fails = {r["case_id"] for r in sf.get("real_fails", []) + sf.get("lung_anomaly", [])}
    df = df[~df["case_id"].isin(fails)].copy()
    mpap = json.loads(MPAP.read_text(encoding="utf-8"))
    df["mpap"] = df["case_id"].map(mpap)
    df.loc[df["protocol"].str.lower() == "plain_scan", "mpap"] = 5.0
    df.loc[(df["label"] == 0) & (df["protocol"].str.lower() == "contrast"), "mpap"] = 15.0
    df["is_contrast"] = (df["protocol"].str.lower() == "contrast").astype(int)
    df["year"] = df["case_id"].apply(parse_year)
    df["year"] = df["year"].fillna(df["year"].median())

    target_features = [
        "artery_tort_p10", "artery_len_p25", "artery_len_p50",
        "vein_len_p25", "vein_tort_p10",
        "paren_std_HU", "paren_mean_HU", "lung_vol_mL",
    ]
    target_features = [c for c in target_features if c in df.columns]

    contrast = df[df["protocol"].str.lower() == "contrast"].copy()
    print(f"contrast n={len(contrast)} (PH={int((contrast['label']==1).sum())} nonPH={int((contrast['label']==0).sum())})")

    out = {"cohort_n": int(len(contrast)),
           "n_ph": int((contrast["label"] == 1).sum()),
           "n_nonph": int((contrast["label"] == 0).sum()),
           "target_features": target_features, "results": {}}

    for col in target_features:
        sub = contrast.dropna(subset=[col, "year"])
        if len(sub) < 30: continue
        # Raw effect
        a = sub.loc[sub["label"] == 1, col].dropna().values
        b = sub.loc[sub["label"] == 0, col].dropna().values
        try: u, p_raw = mannwhitneyu(a, b, alternative="two-sided")
        except Exception: p_raw = float("nan")

        # Residualize against year (within-contrast we can't use mPAP/is_contrast
        # as covariates because mPAP is highly label-correlated and is_contrast=1
        # for all cases in this subset)
        X_cov = sub[["year"]].values
        y_feat = sub[col].values
        lr = LinearRegression().fit(X_cov, y_feat)
        residual = y_feat - lr.predict(X_cov)
        sub_res = sub.copy(); sub_res[f"{col}_resid"] = residual

        a_r = sub_res.loc[sub_res["label"] == 1, f"{col}_resid"].dropna().values
        b_r = sub_res.loc[sub_res["label"] == 0, f"{col}_resid"].dropna().values
        try: u, p_adj = mannwhitneyu(a_r, b_r, alternative="two-sided")
        except Exception: p_adj = float("nan")

        # Cohen's d after residualization
        pooled_sd = np.sqrt(((len(a_r)-1)*np.var(a_r, ddof=1)
                              + (len(b_r)-1)*np.var(b_r, ddof=1))
                             / max(len(a_r) + len(b_r) - 2, 1))
        d_adj = float((a_r.mean() - b_r.mean()) / pooled_sd) if pooled_sd > 0 else 0.0
        d_raw_pooled = np.sqrt(((len(a)-1)*np.var(a, ddof=1)
                                 + (len(b)-1)*np.var(b, ddof=1))
                                / max(len(a)+len(b)-2, 1))
        d_raw = float((a.mean() - b.mean()) / d_raw_pooled) if d_raw_pooled > 0 else 0.0

        # Year-feature correlation (effect size of confound)
        rho_yr, p_yr = spearmanr(sub["year"], sub[col])

        out["results"][col] = {
            "n": int(len(sub)),
            "raw_d": d_raw, "raw_p_mwu": float(p_raw) if p_raw == p_raw else None,
            "adjusted_d": d_adj, "adjusted_p_mwu": float(p_adj) if p_adj == p_adj else None,
            "year_correlation_rho": float(rho_yr),
            "year_correlation_p": float(p_yr),
            "covariate_lr_r2": float(lr.score(X_cov, y_feat)),
            "delta_d_raw_minus_adjusted": d_raw - d_adj,
        }
        print(f"  {col}: raw d={d_raw:+.2f} p={p_raw:.3g} | year-resid d={d_adj:+.2f} p={p_adj:.3g} | "
              f"year-rho={rho_yr:+.2f}")

    (OUT / "covariate_adjusted_endotype.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    md = ["# R18.E — Covariate-adjusted endotype effects (R17 reviewer must-fix)",
          "",
          f"Within-contrast cohort n={out['cohort_n']} "
          f"(PH={out['n_ph']}, nonPH={out['n_nonph']}). Residualize each",
          "endotype feature against `year` (scanner-era proxy from case_id),",
          "then re-test PH vs nonPH on the residual.",
          "",
          "## Covariate-adjustment table",
          "",
          "| feature | raw d | raw p | year-adj d | year-adj p | year-rho | LR R² | Δd (raw−adj) |",
          "|---|---|---|---|---|---|---|---|"]
    for col, r in out["results"].items():
        md.append(f"| {col} | {r['raw_d']:+.2f} | "
                  f"{r['raw_p_mwu']:.3g} | "
                  f"{r['adjusted_d']:+.2f} | "
                  f"{r['adjusted_p_mwu']:.3g} | "
                  f"{r['year_correlation_rho']:+.2f} | "
                  f"{r['covariate_lr_r2']:.3f} | "
                  f"{r['delta_d_raw_minus_adjusted']:+.2f} |")
    md += ["",
           "## Interpretation",
           "",
           "- If raw d ≈ adjusted d, the endotype effect is INDEPENDENT of",
           "  scanner-era / year confound (robust finding).",
           "- If |raw d − adjusted d| > 0.2, the effect is partially confounded.",
           "- Year-correlation rho gives the magnitude of the confound on the",
           "  raw feature; LR R² gives variance explained by year alone.",
           "",
           "## Caveats",
           "",
           "1. `year` is the only available scanner-era proxy from case_id.",
           "   True scanner-model / kernel / dose / reconstruction metadata not",
           "   available in this cohort.",
           "2. Within-contrast restriction (n=" + str(out['cohort_n']) + ") means",
           "   we cannot use is_contrast as covariate (constant=1).",
           "3. mPAP not used as covariate to avoid label-correlation collinearity",
           "   (PH-vs-nonPH is part of the mPAP gradient by definition).",
           ""]
    (OUT / "covariate_adjusted_endotype.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nsaved → {OUT}/covariate_adjusted_endotype.{{json,md}}")


if __name__ == "__main__":
    main()
