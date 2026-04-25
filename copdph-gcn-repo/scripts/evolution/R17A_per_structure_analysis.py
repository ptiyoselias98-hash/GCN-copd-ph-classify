"""R17.A — Analysis on per-structure topology fingerprints.

Reads outputs/r17/per_structure_morphometrics.csv (282 cases × ~150 cols).

Three deliverables:
  1) Per-structure within-contrast PH-vs-nonPH Holm-Bonferroni forest plot
     (artery vs vein vs airway — which structure shows strongest PH signature?)
  2) Per-structure UMAP+KMeans (k=2..6 stability sweep) + endotype enrichment
  3) Per-structure within-nonPH protocol probe (lung-only baseline beats this?)

Outputs:
  outputs/r17/per_structure_endotype.{json,md}
  outputs/r17/per_structure_protocol_probe.{json,md}
  outputs/figures/fig_r17_per_structure_forest.png
  outputs/figures/fig_r17_per_structure_volcano.png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent.parent
OUT = ROOT / "outputs" / "r17"
FIG = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
CSV = OUT / "per_structure_morphometrics.csv"
LABELS = ROOT / "data" / "labels_extended_382.csv"
PROTO = ROOT / "data" / "case_protocol_extended.csv"
SEG_FAILS = ROOT / "outputs" / "r13" / "seg_failures_real.json"


def cohens_d(a, b):
    pooled_sd = np.sqrt(((len(a)-1)*np.var(a, ddof=1) + (len(b)-1)*np.var(b, ddof=1))
                         / max(len(a)+len(b)-2, 1))
    if pooled_sd == 0: return 0.0
    return float((a.mean() - b.mean()) / pooled_sd)


def boot_d_ci(a, b, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed); arr = []
    for _ in range(n_boot):
        ai = rng.choice(len(a), len(a), replace=True)
        bi = rng.choice(len(b), len(b), replace=True)
        arr.append(cohens_d(a[ai], b[bi]))
    return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]


def holm(pvals, alpha=0.05):
    n = len(pvals); order = sorted(range(n), key=lambda i: pvals[i])
    out = [None]*n; prev_corr = 0
    for rank, i in enumerate(order):
        p_corr = pvals[i] * (n - rank)
        p_corr = max(p_corr, prev_corr); p_corr = min(p_corr, 1.0)
        out[i] = (rank+1, pvals[i], p_corr, p_corr < alpha)
        prev_corr = p_corr
    return out


def oof_lr(X, y, seed=42):
    if len(np.unique(y)) < 2: return None, None
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0)
        clf.fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return float(roc_auc_score(y, oof)), oof


def boot_auc_ci(y, p, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    if not len(pos) or not len(neg): return [float("nan"), float("nan")]
    boots = []
    for _ in range(n_boot):
        bp = rng.choice(pos, size=len(pos), replace=True)
        bn = rng.choice(neg, size=len(neg), replace=True)
        idx = np.concatenate([bp, bn])
        try:
            boots.append(roc_auc_score(y[idx], p[idx]))
        except ValueError: continue
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def main():
    if not CSV.exists():
        raise SystemExit(f"missing {CSV} — wait for R17 extractor to finish")
    df = pd.read_csv(CSV)
    lab = pd.read_csv(LABELS); pro = pd.read_csv(PROTO)
    df = df.merge(lab[["case_id", "label"]], on="case_id", how="inner", suffixes=("", "_dup")) \
        .merge(pro[["case_id", "protocol"]], on="case_id", how="inner")
    if "label_dup" in df.columns: df = df.drop(columns=["label_dup"])
    fails = set()
    if SEG_FAILS.exists():
        sf = json.loads(SEG_FAILS.read_text(encoding="utf-8"))
        fails = {r["case_id"] for r in sf.get("real_fails", []) + sf.get("lung_anomaly", [])}
    df = df[~df["case_id"].isin(fails)].copy()
    df["is_contrast"] = (df["protocol"].astype(str).str.lower() == "contrast").astype(int)
    print(f"loaded {len(df)} cases (after seg-failure exclusion)")

    structs = ["artery", "vein", "airway"]
    feature_cols = {s: [c for c in df.columns
                          if c.startswith(f"{s}_") and pd.api.types.is_numeric_dtype(df[c])]
                     for s in structs}
    for s in structs:
        print(f"  {s}: {len(feature_cols[s])} features")

    contrast = df[df["protocol"].str.lower() == "contrast"].copy()
    print(f"within-contrast: n={len(contrast)} "
          f"(PH={int((contrast['label']==1).sum())} nonPH={int((contrast['label']==0).sum())})")

    # ---- Per-structure univariate forest plot (Holm-corrected) ----
    rows = []
    for s in structs:
        for c in feature_cols[s]:
            a = contrast.loc[contrast["label"] == 1, c].dropna().values
            b = contrast.loc[contrast["label"] == 0, c].dropna().values
            if len(a) < 5 or len(b) < 5: continue
            try:
                _, p = mannwhitneyu(a, b, alternative="two-sided")
            except Exception: p = float("nan")
            d = cohens_d(a, b)
            rows.append({"structure": s, "feature": c,
                         "PH_mean": float(a.mean()), "PH_sd": float(a.std(ddof=1)),
                         "nonPH_mean": float(b.mean()), "nonPH_sd": float(b.std(ddof=1)),
                         "delta": float(a.mean()-b.mean()),
                         "cohens_d": d, "p_raw": float(p) if p == p else 1.0})
    if not rows:
        raise SystemExit("no rows produced")
    pvals = [r["p_raw"] for r in rows]
    holm_out = holm(pvals)
    for r, h in zip(rows, holm_out):
        r["p_holm"] = float(h[2]); r["sig_holm"] = bool(h[3])

    # Per-structure top-feature counts
    top_per = {}
    for s in structs:
        s_rows = [r for r in rows if r["structure"] == s]
        n_sig = sum(1 for r in s_rows if r["sig_holm"])
        max_d = max([r for r in s_rows if r["sig_holm"]],
                    key=lambda r: abs(r["cohens_d"]),
                    default={"feature": None, "cohens_d": 0, "p_holm": 1})
        top_per[s] = {"n_features": len(s_rows), "n_sig_holm": n_sig,
                      "top_d_feature": max_d.get("feature"),
                      "top_d": max_d.get("cohens_d"),
                      "top_p_holm": max_d.get("p_holm")}
        print(f"  {s}: {len(s_rows)} features, {n_sig} Holm-sig, top: "
              f"{max_d.get('feature')} d={max_d.get('cohens_d',0):+.2f}")

    out = {"n_total_features": len(rows),
           "n_holm_sig": sum(1 for r in rows if r["sig_holm"]),
           "per_structure_summary": top_per,
           "rows": rows}
    (OUT / "per_structure_endotype.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    # MD report — top 25 by |Cohen's d| with Holm sig
    sorted_rows = sorted(rows, key=lambda r: -abs(r["cohens_d"]))
    md = ["# R17.A — Per-structure topology endotype (within-contrast PH vs nonPH)",
          "",
          f"n_contrast = {len(contrast)} | total features = {len(rows)} | "
          f"Holm-significant = {sum(1 for r in rows if r['sig_holm'])} ({sum(1 for r in rows if r['sig_holm'])/len(rows)*100:.0f}%)",
          "",
          "## Per-structure summary",
          "",
          "| structure | n_features | n_sig_holm | top feature | Cohen's d | p_holm |",
          "|---|---|---|---|---|---|"]
    for s in structs:
        t = top_per[s]
        d_val = f"{t['top_d']:+.2f}" if t['top_d'] is not None else 'NA'
        p_val = f"{t['top_p_holm']:.3g}" if t['top_p_holm'] is not None else 'NA'
        md.append(f"| {s} | {t['n_features']} | {t['n_sig_holm']} | "
                  f"{t['top_d_feature'] or 'NA'} | "
                  f"{d_val} | {p_val} |")
    md += ["",
           "## Top 25 features by |Cohen's d| (across all structures)",
           "",
           "| rank | structure | feature | PH μ±SD | nonPH μ±SD | d | p_holm | sig |",
           "|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(sorted_rows[:25]):
        md.append(f"| {i+1} | {r['structure']} | {r['feature']} | "
                  f"{r['PH_mean']:.3f}±{r['PH_sd']:.3f} | "
                  f"{r['nonPH_mean']:.3f}±{r['nonPH_sd']:.3f} | "
                  f"{r['cohens_d']:+.2f} | {r['p_holm']:.3g} | "
                  f"{'✓' if r['sig_holm'] else ''} |")
    (OUT / "per_structure_endotype.md").write_text("\n".join(md), encoding="utf-8")

    # Forest plot (top 30 by abs Cohen's d, color by structure)
    top30 = sorted_rows[:30]
    fig, ax = plt.subplots(figsize=(11, 9))
    color_map = {"artery": "#ef4444", "vein": "#3b82f6", "airway": "#10b981"}
    for i, r in enumerate(reversed(top30)):
        c = color_map[r["structure"]]
        marker = "o" if r["sig_holm"] else "x"
        ax.errorbar(r["cohens_d"], i, fmt=marker, c=c, markersize=8,
                    markeredgecolor="black", markeredgewidth=0.5)
        sig_label = " ✓" if r["sig_holm"] else ""
        ax.text(2.0, i, f"{r['structure']}/{r['feature'].replace(r['structure']+'_', '')}{sig_label}",
                va="center", fontsize=8)
    ax.axvline(0, c="black", lw=0.8)
    ax.axvline(0.5, ls=":", c="grey", alpha=0.5); ax.axvline(-0.5, ls=":", c="grey", alpha=0.5)
    ax.axvline(0.8, ls="--", c="grey", alpha=0.4); ax.axvline(-0.8, ls="--", c="grey", alpha=0.4)
    ax.set_xlabel("Cohen's d (PH − nonPH within contrast)")
    ax.set_xlim(-2.5, 4.0)
    ax.set_yticks([]); ax.grid(axis="x", alpha=0.3)
    ax.set_title("R17.A — Per-structure topology endotype (top 30 by |Cohen's d|)\n"
                 "Red=artery; Blue=vein; Green=airway. ○=Holm-sig at α=0.05; ✕=NS.")
    plt.tight_layout()
    plt.savefig(FIG / "fig_r17_per_structure_forest.png", dpi=140); plt.close()

    # Volcano: x=Cohen's d, y=-log10(p_holm), color=structure
    fig, ax = plt.subplots(figsize=(10, 7))
    for s in structs:
        s_rows = [r for r in rows if r["structure"] == s]
        x_vals = [r["cohens_d"] for r in s_rows]
        y_vals = [-np.log10(max(r["p_holm"], 1e-15)) for r in s_rows]
        ax.scatter(x_vals, y_vals, c=color_map[s], label=s, alpha=0.6, s=40,
                   edgecolor="black", linewidth=0.3)
    ax.axhline(-np.log10(0.05), ls="--", c="red", alpha=0.5, label="α=0.05 (Holm)")
    ax.axvline(0, c="black", lw=0.5)
    ax.set_xlabel("Cohen's d (PH − nonPH)"); ax.set_ylabel("-log10(p_holm)")
    ax.set_title("R17.A — Per-structure volcano plot (within-contrast PH vs nonPH)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "fig_r17_per_structure_volcano.png", dpi=140); plt.close()

    # ---- Per-structure within-contrast disease classifier ----
    disease_results = {}
    for s in structs:
        feat = [c for c in feature_cols[s] if c in contrast.columns]
        sub = contrast.dropna(subset=feat)
        if len(sub) < 30 or sub["label"].nunique() < 2: continue
        X = sub[feat].values; y = sub["label"].values.astype(int)
        auc, oof = oof_lr(X, y, seed=42)
        if auc is None: continue
        ci = boot_auc_ci(y, oof)
        disease_results[s] = {"n": int(len(sub)), "n_features": len(feat),
                              "auc": auc, "ci95": ci}
        print(f"  disease({s}): n={len(sub)} feat={len(feat)} AUC={auc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")

    # All-structures combined
    all_feat = sum(feature_cols.values(), [])
    sub = contrast.dropna(subset=all_feat)
    if len(sub) >= 30 and sub["label"].nunique() == 2:
        X = sub[all_feat].values; y = sub["label"].values.astype(int)
        auc, oof = oof_lr(X, y, seed=42)
        if auc is not None:
            disease_results["all_three"] = {
                "n": int(len(sub)), "n_features": len(all_feat),
                "auc": auc, "ci95": boot_auc_ci(y, oof)}

    (OUT / "per_structure_disease_aucs.json").write_text(
        json.dumps(disease_results, indent=2), encoding="utf-8")

    print(f"\nsaved → {OUT}/per_structure_endotype.{{json,md}} + "
          f"per_structure_disease_aucs.json")


if __name__ == "__main__":
    main()
