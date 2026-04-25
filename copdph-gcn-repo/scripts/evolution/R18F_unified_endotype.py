"""R18.F — Unified endotype analysis: graph morphometrics + TDA + lung features.

Combines:
  - R17 per-structure morphometrics (132 features: artery/vein/airway × edge-attr distributions)
  - R17.5 TDA persistence (18 features: artery/vein/airway × persH0/persH1 × max/n/total)
  - R16 lung parenchyma (~50 features: paren_LAA, paren_std_HU, etc.)

Total ~200 features. Within-contrast Holm-Bonferroni at α=0.05.

Output: outputs/r18/unified_endotype.{json,md} + fig_r18_unified_forest.png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent
OUT = ROOT / "outputs" / "r18"
FIG = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
MORPH = ROOT / "outputs" / "r17" / "per_structure_morphometrics.csv"
TDA = ROOT / "outputs" / "r17" / "per_structure_tda.csv"
LUNG = ROOT / "outputs" / "lung_features_v2.csv"
LABELS = ROOT / "data" / "labels_extended_382.csv"
PROTO = ROOT / "data" / "case_protocol_extended.csv"
SEG_FAILS = ROOT / "outputs" / "r13" / "seg_failures_real.json"


def cohens_d(a, b):
    pooled = np.sqrt(((len(a)-1)*np.var(a, ddof=1) + (len(b)-1)*np.var(b, ddof=1))
                      / max(len(a)+len(b)-2, 1))
    return float((a.mean()-b.mean())/pooled) if pooled > 0 else 0.0


def holm(pvals, alpha=0.05):
    n = len(pvals); order = sorted(range(n), key=lambda i: pvals[i])
    out = [None]*n; prev = 0
    for rank, i in enumerate(order):
        p_corr = pvals[i] * (n - rank)
        p_corr = max(p_corr, prev); p_corr = min(p_corr, 1.0)
        out[i] = (rank+1, pvals[i], p_corr, p_corr < alpha); prev = p_corr
    return out


def main():
    morph = pd.read_csv(MORPH); tda = pd.read_csv(TDA); lung = pd.read_csv(LUNG)
    lab = pd.read_csv(LABELS); pro = pd.read_csv(PROTO)
    df = lab.merge(pro[["case_id", "protocol"]], on="case_id") \
        .merge(morph, on="case_id", suffixes=("", "_dup1")) \
        .merge(tda, on="case_id", suffixes=("", "_dup2")) \
        .merge(lung, on="case_id", how="left", suffixes=("", "_dup3"))
    df = df.loc[:, ~df.columns.str.contains("_dup")].copy()
    fails = set()
    if SEG_FAILS.exists():
        sf = json.loads(SEG_FAILS.read_text(encoding="utf-8"))
        fails = {r["case_id"] for r in sf.get("real_fails", []) + sf.get("lung_anomaly", [])}
    df = df[~df["case_id"].isin(fails)]
    contrast = df[df["protocol"].str.lower() == "contrast"].copy()
    print(f"contrast n={len(contrast)} (PH={int((contrast['label']==1).sum())} nonPH={int((contrast['label']==0).sum())})")

    # Feature categorization
    cat_morph = [c for c in df.columns if c.startswith(("artery_", "vein_", "airway_"))
                  and "persH" not in c and pd.api.types.is_numeric_dtype(df[c])]
    cat_tda = [c for c in df.columns if "persH" in c and pd.api.types.is_numeric_dtype(df[c])]
    cat_lung = [c for c in df.columns if (c.startswith(("paren_", "whole_", "apical_", "basal_", "middle_"))
                                            or c in ("lung_vol_mL", "vessel_airway_over_lung",
                                                     "artery_vol_mL", "vein_vol_mL", "airway_vol_mL"))
                 and pd.api.types.is_numeric_dtype(df[c])]
    print(f"  morph_features: {len(cat_morph)}; tda_features: {len(cat_tda)}; lung_features: {len(cat_lung)}")

    rows = []
    for cat_name, feats in [("morph", cat_morph), ("tda", cat_tda), ("lung", cat_lung)]:
        for c in feats:
            a = contrast.loc[contrast["label"] == 1, c].dropna().values
            b = contrast.loc[contrast["label"] == 0, c].dropna().values
            if len(a) < 5 or len(b) < 5: continue
            try: u, p = mannwhitneyu(a, b, alternative="two-sided")
            except Exception: p = 1.0
            d = cohens_d(a, b)
            rows.append({"category": cat_name, "feature": c,
                         "PH_mean": float(a.mean()), "PH_sd": float(a.std(ddof=1)),
                         "nonPH_mean": float(b.mean()), "nonPH_sd": float(b.std(ddof=1)),
                         "delta": float(a.mean()-b.mean()),
                         "cohens_d": d, "p_raw": float(p)})
    pvals = [r["p_raw"] for r in rows]
    holm_out = holm(pvals)
    for r, h in zip(rows, holm_out):
        r["p_holm"] = float(h[2]); r["sig_holm"] = bool(h[3])

    # Per-category top
    summary = {}
    for cat in ["morph", "tda", "lung"]:
        s_rows = [r for r in rows if r["category"] == cat]
        n_sig = sum(1 for r in s_rows if r["sig_holm"])
        top = max(s_rows, key=lambda r: abs(r["cohens_d"]) if r["sig_holm"] else 0,
                  default={"feature": None, "cohens_d": 0, "p_holm": 1})
        summary[cat] = {"n_features": len(s_rows), "n_sig": n_sig,
                        "top_d_feature": top.get("feature"),
                        "top_d": top.get("cohens_d"),
                        "top_p_holm": top.get("p_holm")}
        print(f"  {cat}: {len(s_rows)} feats, {n_sig} Holm-sig, top: "
              f"{top.get('feature')} d={top.get('cohens_d', 0):+.2f}")

    out = {"n": int(len(contrast)), "n_total_features": len(rows),
           "n_holm_sig": sum(1 for r in rows if r["sig_holm"]),
           "per_category_summary": summary, "rows": rows}
    (OUT / "unified_endotype.json").write_text(json.dumps(out, indent=2),
                                                  encoding="utf-8")

    sorted_rows = sorted(rows, key=lambda r: -abs(r["cohens_d"]))
    md = ["# R18.F — Unified endotype analysis (morph + TDA + lung)",
          "",
          f"Within-contrast n={out['n']}; total features={out['n_total_features']}; "
          f"Holm-significant={out['n_holm_sig']} ({out['n_holm_sig']/out['n_total_features']*100:.0f}%).",
          "",
          "## Per-category summary",
          "",
          "| category | n_features | n_sig_holm | top feature | Cohen's d | p_holm |",
          "|---|---|---|---|---|---|"]
    for cat, s in summary.items():
        d_val = f"{s['top_d']:+.2f}" if s['top_d'] is not None else 'NA'
        p_val = f"{s['top_p_holm']:.3g}" if s['top_p_holm'] is not None else 'NA'
        md.append(f"| {cat} | {s['n_features']} | {s['n_sig']} | "
                  f"{s['top_d_feature'] or 'NA'} | {d_val} | {p_val} |")
    md += ["",
           "## Top 30 features overall by |Cohen's d|",
           "",
           "| rank | category | feature | Cohen's d | p_holm | sig |",
           "|---|---|---|---|---|---|"]
    for i, r in enumerate(sorted_rows[:30]):
        md.append(f"| {i+1} | {r['category']} | {r['feature']} | "
                  f"{r['cohens_d']:+.2f} | {r['p_holm']:.3g} | "
                  f"{'✓' if r['sig_holm'] else ''} |")
    (OUT / "unified_endotype.md").write_text("\n".join(md), encoding="utf-8")

    # Forest plot — top 30 colored by category
    top30 = sorted_rows[:30]
    fig, ax = plt.subplots(figsize=(10, 9))
    color_map = {"morph": "#ef4444", "tda": "#8b5cf6", "lung": "#10b981"}
    for i, r in enumerate(reversed(top30)):
        c = color_map[r["category"]]
        marker = "o" if r["sig_holm"] else "x"
        ax.plot(r["cohens_d"], i, marker, c=c, markersize=8,
                markeredgecolor="black", markeredgewidth=0.5)
        sig_label = " ✓" if r["sig_holm"] else ""
        ax.text(2.0, i, f"{r['category']}/{r['feature']}{sig_label}",
                va="center", fontsize=8)
    ax.axvline(0, c="black", lw=0.8)
    ax.axvline(0.5, ls=":", c="grey"); ax.axvline(-0.5, ls=":", c="grey")
    ax.axvline(0.8, ls="--", c="grey"); ax.axvline(-0.8, ls="--", c="grey")
    ax.set_xlabel("Cohen's d (PH − nonPH within contrast)")
    ax.set_xlim(-2.5, 4.0); ax.set_yticks([]); ax.grid(axis="x", alpha=0.3)
    ax.set_title("R18.F — Unified endotype top 30 features\n"
                 "RED=graph morphometrics; PURPLE=TDA persistence; GREEN=lung parenchyma")
    plt.tight_layout()
    plt.savefig(FIG / "fig_r18_unified_forest.png", dpi=140); plt.close()
    print(f"saved → {OUT}/unified_endotype.{{json,md}} + fig_r18_unified_forest.png")


if __name__ == "__main__":
    main()
