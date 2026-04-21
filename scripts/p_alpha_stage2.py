"""P-α Stage 2: align clusters with clinical, per-variable p-values, report top drivers."""
from __future__ import annotations
import sys, io, json, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

try:
    from pypinyin import lazy_pinyin
except ImportError:
    print("installing pypinyin ...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pypinyin"], check=True)
    from pypinyin import lazy_pinyin

ROOT = Path(__file__).resolve().parent.parent
CLINICAL_XLSX = ROOT / "copd-ph患者113例0331.xlsx"
CLUSTER_CSV   = ROOT / "outputs" / "cluster_topology" / "cluster_assignments.csv"
OUT_DIR       = ROOT / "outputs" / "p_alpha_cluster_clinical"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load
ca = pd.read_csv(CLUSTER_CSV)
df_clin = pd.ExcelFile(CLINICAL_XLSX).parse("Sheet1")

# ── extract pinyin name from case_id
# e.g. nonph_caochenglin_g02017953_... → caochenglin
def extract_name_pinyin(case_id: str) -> str:
    parts = case_id.split("_")
    # parts[0] = ph/nonph, parts[1] = name pinyin
    return parts[1] if len(parts) >= 2 else ""

ca["name_pinyin"] = ca["case_id"].apply(extract_name_pinyin)

# ── convert clinical names to pinyin
def chinese_to_pinyin(name) -> str:
    if not isinstance(name, str) or not name.strip():
        return ""
    return "".join(lazy_pinyin(name.strip()))

df_clin["name_pinyin"] = df_clin["name"].apply(chinese_to_pinyin)

# ── join
joined = ca.merge(df_clin, on="name_pinyin", how="inner", suffixes=("_cl", "_clin"))
print(f"Merged on name_pinyin: {len(joined)}/{len(ca)} cluster rows matched "
      f"({len(joined)}/{len(df_clin)} clinical rows matched)")

# dump for debugging
joined.to_csv(OUT_DIR / "_merged.csv", index=False, encoding="utf-8-sig")

# ── figure out numeric columns (potential drivers)
clinical_cols = [c for c in df_clin.columns if c not in ("name", "name_pinyin", "patient_sn", "id_card_no_x", "ct文件名", "分割索引", "导管时间")]
# try converting each to numeric
numeric_cols = []
for c in clinical_cols:
    s = pd.to_numeric(joined[c], errors="coerce")
    frac_num = s.notna().mean()
    if frac_num >= 0.5:  # at least 50% numeric → treat as numeric
        joined[f"_num_{c}"] = s
        numeric_cols.append(c)
print(f"Numeric-ish clinical columns: {len(numeric_cols)}")

categorical_cols = [c for c in clinical_cols if c not in numeric_cols
                    and joined[c].nunique(dropna=True) <= 10
                    and joined[c].nunique(dropna=True) >= 2]
print(f"Categorical (≤10 uniq) clinical columns: {len(categorical_cols)}")

# ── run per-clustering-method test
cluster_methods = [c for c in ca.columns if c.startswith(("gmm_", "kmeans_", "spectral_"))]
print(f"Cluster methods: {cluster_methods}")

pvalue_rows = []
for method in cluster_methods:
    groups = joined[method].dropna().astype(int)
    if groups.nunique() < 2:
        continue
    # numeric vars: Kruskal-Wallis (non-parametric ANOVA)
    for c in numeric_cols:
        vals_by_group = [joined.loc[groups.index[groups==g], f"_num_{c}"].dropna().values
                         for g in sorted(groups.unique())]
        vals_by_group = [v for v in vals_by_group if len(v) >= 3]
        if len(vals_by_group) < 2:
            continue
        try:
            stat, p = stats.kruskal(*vals_by_group)
        except Exception:
            continue
        pvalue_rows.append({
            "cluster_method": method, "variable": c, "type": "numeric",
            "test": "kruskal", "stat": stat, "pvalue": p,
            "n": int(sum(len(v) for v in vals_by_group)),
            "n_groups": len(vals_by_group),
        })
    # categorical vars: chi-square
    for c in categorical_cols:
        ct = pd.crosstab(joined[c], groups)
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        try:
            chi2, p, dof, _ = stats.chi2_contingency(ct)
        except Exception:
            continue
        pvalue_rows.append({
            "cluster_method": method, "variable": c, "type": "categorical",
            "test": "chi2", "stat": chi2, "pvalue": p,
            "n": int(ct.values.sum()), "n_groups": ct.shape[1],
        })

pv = pd.DataFrame(pvalue_rows)
pv = pv.sort_values(["cluster_method", "pvalue"])
pv.to_csv(OUT_DIR / "cluster_vs_feature_pvalues.csv", index=False, encoding="utf-8-sig")
print(f"\nWrote {len(pv)} p-values to {OUT_DIR/'cluster_vs_feature_pvalues.csv'}")

# ── top-10 per best clustering
# "best clustering" = the one with lowest min p-value (most separable)
method_best = pv.groupby("cluster_method")["pvalue"].min().sort_values()
print("\nClustering methods ranked by min p-value across variables:")
print(method_best.head(10).to_string())

best_method = method_best.index[0]
print(f"\n>>> Most separable clustering: {best_method}")
top_vars = pv[pv["cluster_method"]==best_method].head(30)
print("\nTop 30 variables driving the clusters (p < ... ):")
print(top_vars[["variable","type","test","pvalue","n"]].to_string(index=False))

# Write a markdown summary
md = [f"# P-α : cluster × clinical cross-table  (2026-04-21)", ""]
md.append(f"**Matched** {len(joined)}/{len(ca)} cluster rows to clinical via pinyin(name).")
md.append(f"**Numeric variables tested**: {len(numeric_cols)}")
md.append(f"**Categorical variables tested**: {len(categorical_cols)}")
md.append("")
md.append("## Clustering methods ranked by their best-separated variable")
md.append("")
md.append("| cluster_method | min_pvalue |")
md.append("|---|---|")
for m, p in method_best.head(8).items():
    md.append(f"| `{m}` | {p:.3e} |")
md.append("")
md.append(f"## Top 30 drivers of `{best_method}`")
md.append("")
md.append("| rank | variable | type | test | pvalue | n |")
md.append("|---|---|---|---|---|---|")
for i, r in enumerate(top_vars.itertuples(index=False), 1):
    md.append(f"| {i} | `{r.variable}` | {r.type} | {r.test} | {r.pvalue:.3e} | {r.n} |")
md.append("")

# Significance summary: how many variables pass Bonferroni-corrected threshold?
n_tests = len(numeric_cols) + len(categorical_cols)
alpha = 0.05
bonf = alpha / n_tests
md.append(f"## Significance summary")
md.append(f"- Uncorrected alpha=0.05: variables passing per method")
for m in method_best.index[:6]:
    n_uncorr = int((pv[pv["cluster_method"]==m]["pvalue"] < alpha).sum())
    n_bonf   = int((pv[pv["cluster_method"]==m]["pvalue"] < bonf).sum())
    md.append(f"  - `{m}`: {n_uncorr} uncorrected, {n_bonf} Bonferroni (α={bonf:.3e})")
md.append("")

# PH label sanity
if "label" in joined.columns:
    md.append("## Sanity check — PH label vs best clustering")
    md.append("")
    for m in method_best.index[:3]:
        ct = pd.crosstab(joined["label"], joined[m].astype(int),
                         margins=True, margins_name="total")
        md.append(f"### `{m}`")
        md.append("")
        try:
            md.append(ct.to_markdown())
        except ImportError:
            md.append("```")
            md.append(ct.to_string())
            md.append("```")
        md.append("")

(OUT_DIR / "p_alpha_summary.md").write_text("\n".join(md), encoding="utf-8")
print(f"\nWrote markdown summary: {OUT_DIR/'p_alpha_summary.md'}")
