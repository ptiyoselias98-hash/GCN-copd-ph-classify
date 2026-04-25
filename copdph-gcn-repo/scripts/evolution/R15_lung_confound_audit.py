"""R15.C — Audit lung HU features for residual scanner/reconstruction confound.

R14 reviewer flagged: "lung HU features may retain acquisition/reconstruction
confound even within contrast-only". This script tests whether lung HU
moments cluster by patient_id surrogate (which acts as a scanner/reconstruction
proxy if cases from the same scanner share patient ID prefixes), and
whether splitting contrast cases by HU mean reveals systematic spatial-
or temporal-acquisition patterns.

Outputs: outputs/r15/lung_confound_audit.{json,md}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).parent.parent.parent
OUT = ROOT / "outputs" / "r15"
OUT.mkdir(parents=True, exist_ok=True)
LUNG = ROOT / "outputs" / "lung_features_v2.csv"
LABELS = ROOT / "data" / "labels_expanded_282.csv"
PROTO = ROOT / "data" / "case_protocol.csv"
SEG_FAILS = ROOT / "outputs" / "r13" / "seg_failures_real.json"


def parse_year_from_case_id(cid: str) -> int | None:
    """Extract scan year from `..._<weekday>_<month>_<day>_<year>_<idx>` case_id."""
    parts = cid.split("_")
    for p in reversed(parts):
        if p.isdigit() and 1990 <= int(p) <= 2030:
            return int(p)
    return None


def main():
    if not LUNG.exists():
        raise SystemExit("missing lung_features_v2.csv")
    df = pd.read_csv(LUNG)
    labels = pd.read_csv(LABELS); proto = pd.read_csv(PROTO)
    df = df.merge(labels, on="case_id", how="left").merge(
        proto[["case_id", "protocol"]], on="case_id", how="left")
    fails = set()
    if SEG_FAILS.exists():
        sf = json.loads(SEG_FAILS.read_text(encoding="utf-8"))
        fails = {r["case_id"] for r in sf.get("real_fails", []) + sf.get("lung_anomaly", [])}
    df = df[~df["case_id"].isin(fails)].copy()
    df["year"] = df["case_id"].apply(parse_year_from_case_id)

    # Restrict to within-contrast (no protocol confound)
    contrast = df[df["protocol"].str.lower() == "contrast"].copy()

    out: dict = {"n_total": int(len(df)),
                 "n_contrast": int(len(contrast))}

    # Test 1 — does lung HU mean drift by year (scanner-era proxy)?
    # within contrast-only
    yr_groups = contrast.groupby("year")["paren_mean_HU"].agg(["mean", "std", "count"])
    yr_groups = yr_groups[yr_groups["count"] >= 5]  # only years with ≥5 cases
    out["paren_mean_HU_by_year"] = yr_groups.reset_index().to_dict(orient="records")
    if len(yr_groups) >= 3:
        groups = [contrast.loc[contrast["year"] == y, "paren_mean_HU"].dropna().values
                  for y in yr_groups.index]
        groups = [g for g in groups if len(g) >= 5]
        if len(groups) >= 3:
            stat, p = kruskal(*groups)
            out["kruskal_paren_mean_HU_by_year"] = {
                "stat": float(stat), "p": float(p), "n_groups": len(groups),
                "interpretation": ("year systematically shifts HU (likely scanner/reconstruction confound)"
                                   if p < 0.01 else "no significant year-effect on HU within contrast")
            }

    # Test 2 — Spearman: paren_mean_HU vs label within contrast (disease confound check)
    sub = contrast.dropna(subset=["paren_mean_HU"])
    if len(sub) > 30 and len(sub["label"].unique()) > 1:
        rho, p = spearmanr(sub["paren_mean_HU"], sub["label"])
        out["spearman_HU_vs_label_contrast"] = {
            "rho": float(rho), "p": float(p), "n": int(len(sub)),
            "interpretation": ("HU correlates with disease label within contrast "
                               + ("(disease signal)" if p < 0.05 else "(weak/NS)"))
        }

    # Test 3 — KMeans on HU moments and check if clusters split year (scanner-cluster proxy)
    hu_cols = ["paren_mean_HU", "paren_std_HU", "paren_HU_p25", "paren_HU_p75",
               "paren_HU_p95", "paren_HU_p5", "whole_mean_HU", "whole_std_HU"]
    hu_cols = [c for c in hu_cols if c in contrast.columns]
    sub2 = contrast.dropna(subset=hu_cols + ["year"]).reset_index(drop=True)
    if len(sub2) >= 50 and len(hu_cols) >= 3:
        X = StandardScaler().fit_transform(sub2[hu_cols].values)
        km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
        sub2["hu_cluster"] = km.labels_
        ct = pd.crosstab(sub2["hu_cluster"], sub2["year"])
        # Year-vs-HU-cluster chi-square approximation via Kruskal on year-by-cluster
        if len(sub2["hu_cluster"].unique()) >= 2:
            yr_groups_c = [sub2.loc[sub2["hu_cluster"] == c, "year"].dropna().values
                           for c in sub2["hu_cluster"].unique()]
            yr_groups_c = [g for g in yr_groups_c if len(g) >= 5]
            if len(yr_groups_c) >= 2:
                stat, p = kruskal(*yr_groups_c)
                out["kruskal_year_by_HUcluster"] = {
                    "stat": float(stat), "p": float(p),
                    "n_clusters": len(yr_groups_c),
                    "interpretation": ("HU clusters split by scan-year (residual scanner confound)"
                                       if p < 0.01 else "HU clusters not year-correlated")
                }
        out["HUcluster_year_crosstab"] = {
            "rows_HU_cluster": ct.index.tolist(),
            "cols_year": ct.columns.tolist(),
            "table": ct.values.tolist()
        }
        sil = silhouette_score(X, km.labels_)
        out["HU_cluster_silhouette"] = float(sil)

    # Test 4 — paren_mean_HU range within-contrast PH vs nonPH (disease vs scanner separability)
    contrast_ph = contrast[contrast["label"] == 1]["paren_mean_HU"].dropna().values
    contrast_nonph = contrast[contrast["label"] == 0]["paren_mean_HU"].dropna().values
    if len(contrast_ph) >= 5 and len(contrast_nonph) >= 5:
        out["HU_range_within_contrast"] = {
            "PH": {"n": int(len(contrast_ph)),
                   "mean": float(contrast_ph.mean()),
                   "sd": float(contrast_ph.std(ddof=1)),
                   "p5_p95": [float(np.percentile(contrast_ph, 5)),
                              float(np.percentile(contrast_ph, 95))]},
            "nonPH": {"n": int(len(contrast_nonph)),
                      "mean": float(contrast_nonph.mean()),
                      "sd": float(contrast_nonph.std(ddof=1)),
                      "p5_p95": [float(np.percentile(contrast_nonph, 5)),
                                 float(np.percentile(contrast_nonph, 95))]},
            "delta_means": float(contrast_ph.mean() - contrast_nonph.mean()),
        }

    (OUT / "lung_confound_audit.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# R15.C — Lung HU residual-confound audit (within contrast-only)",
          "",
          f"Cohort: contrast-only n={out['n_contrast']} (excluding seg-failures).",
          "Tests whether lung HU features carry scanner/reconstruction-era",
          "confound that survives within-contrast restriction.",
          "",
          "## Test 1 — Year-effect on paren_mean_HU (Kruskal-Wallis)",
          ""]
    if "kruskal_paren_mean_HU_by_year" in out:
        r = out["kruskal_paren_mean_HU_by_year"]
        md.append(f"- H = {r['stat']:.2f}, p = {r['p']:.3g} ({r['n_groups']} year-groups)")
        md.append(f"- **{r['interpretation']}**")
    md.append("")
    md.append("Group means by year:")
    md.append("")
    md.append("| year | n | μ paren_mean_HU | σ |")
    md.append("|---|---|---|---|")
    for r in out.get("paren_mean_HU_by_year", []):
        md.append(f"| {r['year']} | {r['count']} | {r['mean']:.1f} | {r['std']:.1f} |")

    if "spearman_HU_vs_label_contrast" in out:
        r = out["spearman_HU_vs_label_contrast"]
        md += ["",
               "## Test 2 — Spearman HU vs disease label (within contrast)",
               "",
               f"- ρ = {r['rho']:+.3f}, p = {r['p']:.3g}, n = {r['n']}",
               f"- **{r['interpretation']}**"]

    if "HU_range_within_contrast" in out:
        r = out["HU_range_within_contrast"]
        md += ["",
               "## Test 4 — HU range within-contrast PH vs nonPH",
               "",
               "| group | n | μ HU | σ | p5–p95 |",
               "|---|---|---|---|---|",
               f"| PH | {r['PH']['n']} | {r['PH']['mean']:.1f} | {r['PH']['sd']:.1f} | "
               f"[{r['PH']['p5_p95'][0]:.0f}, {r['PH']['p5_p95'][1]:.0f}] |",
               f"| nonPH | {r['nonPH']['n']} | {r['nonPH']['mean']:.1f} | {r['nonPH']['sd']:.1f} | "
               f"[{r['nonPH']['p5_p95'][0]:.0f}, {r['nonPH']['p5_p95'][1]:.0f}] |",
               f"| Δ means | — | {r['delta_means']:+.2f} | — | — |"]

    if "kruskal_year_by_HUcluster" in out:
        r = out["kruskal_year_by_HUcluster"]
        md += ["",
               "## Test 3 — KMeans HU-cluster vs scan-year (residual scanner-confound proxy)",
               "",
               f"- Kruskal H = {r['stat']:.2f}, p = {r['p']:.3g} ({r['n_clusters']} clusters)",
               f"- HU-cluster silhouette = {out.get('HU_cluster_silhouette', float('nan')):.3f}",
               f"- **{r['interpretation']}**"]

    md += ["",
           "## Combined verdict",
           "",
           "If Test 1 + Test 3 are both significant, lung HU features carry strong",
           "scanner/era confound even within contrast-only — the R14 'lung-only AUC'",
           "claim must be re-scoped to 'lung-features-as-currently-extracted include both",
           "disease and scanner signals'. If only Test 2 is significant (HU correlates",
           "with disease label) but Tests 1+3 are NS, the lung-only AUC is genuine",
           "disease signal.",
           ""]
    (OUT / "lung_confound_audit.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Saved {OUT}/lung_confound_audit.json + .md")


if __name__ == "__main__":
    main()
