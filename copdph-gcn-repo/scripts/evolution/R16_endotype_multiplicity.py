"""R16.B — Multiplicity-corrected endotype effect-size table.

R15 reviewer flag: endotype p-values reported without multiple-testing
correction. This re-runs the within-contrast PH vs nonPH comparison from
R15.G with Holm-Bonferroni correction across the 10+ candidate features,
plus Cohen's d effect size and 95% bootstrap CI for the d.

Output: outputs/r16/endotype_corrected.{json,md}
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).parent.parent.parent
OUT = ROOT / "outputs" / "r16"
OUT.mkdir(parents=True, exist_ok=True)
LUNG = ROOT / "outputs" / "r15" / "lung_features_extended.csv"
LABELS = ROOT / "data" / "labels_extended_382.csv"
PROTO = ROOT / "data" / "case_protocol_extended.csv"
SEG_FAILS = ROOT / "outputs" / "r13" / "seg_failures_real.json"


def cohens_d(a, b):
    pooled_sd = np.sqrt(
        ((len(a)-1)*np.var(a, ddof=1) + (len(b)-1)*np.var(b, ddof=1))
        / max(len(a)+len(b)-2, 1))
    if pooled_sd == 0: return 0.0
    return (a.mean() - b.mean()) / pooled_sd


def boot_d_ci(a, b, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    arr = []
    for _ in range(n_boot):
        ai = rng.choice(len(a), len(a), replace=True)
        bi = rng.choice(len(b), len(b), replace=True)
        arr.append(cohens_d(a[ai], b[bi]))
    return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]


def holm_bonferroni(pvals, alpha=0.05):
    """Returns list of (rank, p_raw, p_corrected, reject) sorted by raw p."""
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    out = [None]*n
    for rank, i in enumerate(order):
        p_corr = pvals[i] * (n - rank)
        # Monotonic enforcement
        if rank > 0:
            prev_corr = out[order[rank-1]][2] if out[order[rank-1]] else 0
            p_corr = max(p_corr, prev_corr)
        p_corr = min(p_corr, 1.0)
        out[i] = (rank+1, pvals[i], p_corr, p_corr < alpha)
    return out


def main():
    lung = pd.read_csv(LUNG)
    lab = pd.read_csv(LABELS); pro = pd.read_csv(PROTO)
    df = lung.merge(lab[["case_id", "label"]], on="case_id") \
        .merge(pro[["case_id", "protocol"]], on="case_id")
    fails = set()
    if SEG_FAILS.exists():
        sf = json.loads(SEG_FAILS.read_text(encoding="utf-8"))
        fails = {r["case_id"] for r in sf.get("real_fails", []) + sf.get("lung_anomaly", [])}
    df = df[~df["case_id"].isin(fails)]
    contrast = df[df["protocol"].str.lower() == "contrast"].copy()
    print(f"contrast n={len(contrast)} (PH={int((contrast['label']==1).sum())} nonPH={int((contrast['label']==0).sum())})")

    candidates = [
        "paren_mean_HU", "paren_std_HU", "paren_LAA_950_frac",
        "paren_LAA_910_frac", "paren_LAA_856_frac",
        "apical_basal_LAA950_gradient",
        "apical_LAA_950_frac", "basal_LAA_950_frac",
        "lung_vol_mL", "vessel_airway_over_lung",
        "artery_vol_mL", "vein_vol_mL",
        "whole_mean_HU", "whole_std_HU",
    ]
    rows = []
    for col in candidates:
        if col not in contrast.columns: continue
        a = contrast.loc[contrast["label"] == 1, col].dropna().values
        b = contrast.loc[contrast["label"] == 0, col].dropna().values
        if len(a) < 5 or len(b) < 5: continue
        try:
            u, p = mannwhitneyu(a, b, alternative="two-sided")
        except Exception:
            p = float("nan")
        d = cohens_d(a, b)
        d_ci = boot_d_ci(a, b)
        rows.append({"feature": col, "n_PH": int(len(a)), "n_nonPH": int(len(b)),
                     "PH_mean": float(a.mean()), "PH_sd": float(a.std(ddof=1)),
                     "nonPH_mean": float(b.mean()), "nonPH_sd": float(b.std(ddof=1)),
                     "delta": float(a.mean() - b.mean()),
                     "cohens_d": float(d), "cohens_d_ci95": d_ci,
                     "p_raw": float(p) if p == p else None})
    pvals = [r["p_raw"] if r["p_raw"] is not None else 1.0 for r in rows]
    holm = holm_bonferroni(pvals)
    for r, h in zip(rows, holm):
        r["holm_rank"] = h[0]
        r["p_holm"] = float(h[2])
        r["sig_holm_05"] = bool(h[3])

    out = {"cohort_n": int(len(contrast)),
           "n_features_tested": len(rows),
           "alpha": 0.05,
           "method": "Holm-Bonferroni",
           "results": rows}
    (OUT / "endotype_corrected.json").write_text(json.dumps(out, indent=2),
                                                    encoding="utf-8")

    md = ["# R16.B — Multiplicity-corrected endotype effect sizes",
          "",
          f"Within-contrast cohort n={out['cohort_n']} (PH vs nonPH).",
          f"{out['n_features_tested']} candidate features. Holm-Bonferroni "
          f"correction at α=0.05.",
          "",
          "| feature | PH μ±SD | nonPH μ±SD | Δ | Cohen's d [95% CI] | p_raw | p_holm | sig |",
          "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: x["holm_rank"]):
        md.append(f"| {r['feature']} | {r['PH_mean']:.3f} ± {r['PH_sd']:.3f} | "
                  f"{r['nonPH_mean']:.3f} ± {r['nonPH_sd']:.3f} | "
                  f"{r['delta']:+.3f} | "
                  f"{r['cohens_d']:+.2f} [{r['cohens_d_ci95'][0]:+.2f}, {r['cohens_d_ci95'][1]:+.2f}] | "
                  f"{r['p_raw']:.3g} | {r['p_holm']:.3g} | "
                  f"{'✓' if r['sig_holm_05'] else ''} |")
    md += ["",
           "## Interpretation",
           "",
           "Features with p_holm < 0.05 survive multiple-testing correction and",
           "represent the defensible quantitative within-contrast PH vs nonPH",
           "endotype signature. Cohen's d ≥ 0.5 = medium effect; ≥ 0.8 = large.",
           ""]
    (OUT / "endotype_corrected.md").write_text("\n".join(md), encoding="utf-8")
    print(f"saved → {OUT}/endotype_corrected.{{json,md}}")
    print(f"Holm-significant features: {sum(1 for r in rows if r['sig_holm_05'])}/{len(rows)}")


if __name__ == "__main__":
    main()
