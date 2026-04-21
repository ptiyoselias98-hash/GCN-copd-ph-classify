"""P-α: Cluster subtype × clinical variable cross-table.

Goal: explain why clusters are PH-orthogonal — what variable(s) actually
drive the clustering?

Inputs:
  - outputs/cluster_topology/cluster_assignments.csv  (per-case cluster id)
  - E:/桌面文件/copd-ph患者113例0331.xlsx              (clinical metadata, 113 rows)
  - copdph-gcn-repo/data/copd_ph_radiomics.csv          (radiomics, 113 rows)

Outputs (under outputs/p_alpha_cluster_clinical/):
  - cluster_clinical_crosstab.xlsx   (per-cluster means/counts)
  - cluster_vs_feature_pvalues.csv   (chi2 / Kruskal-Wallis per variable)
  - significant_features.md          (features with p<0.05 per best clustering)
"""
from __future__ import annotations
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
CLINICAL_XLSX = ROOT / "copd-ph患者113例0331.xlsx"
CLUSTER_CSV   = ROOT / "outputs" / "cluster_topology" / "cluster_assignments.csv"
RADIOMICS_CSV = ROOT / "data" / "copd_ph_radiomics.csv"
OUT_DIR       = ROOT / "outputs" / "p_alpha_cluster_clinical"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("P-α : cluster × clinical cross-table")
print("=" * 60)

# ── 1. Load cluster assignments
ca = pd.read_csv(CLUSTER_CSV)
print(f"\n[cluster] shape={ca.shape}, cols={list(ca.columns)[:10]}")
print(ca.head())

# ── 2. Load clinical xlsx
print(f"\n[clinical] loading {CLINICAL_XLSX.name}")
xl = pd.ExcelFile(CLINICAL_XLSX)
print(f"  sheets: {xl.sheet_names}")
# First sheet is the annotation; Sheet1 is the actual data
data_sheet = "Sheet1" if "Sheet1" in xl.sheet_names else xl.sheet_names[-1]
df_clin = xl.parse(data_sheet)
print(f"  data sheet = {data_sheet}")
print(f"  shape={df_clin.shape}")
print(f"  columns[:20]: {list(df_clin.columns)[:20]}")
print(df_clin.head(3))

# ── 3. Load radiomics as fallback clinical (has label + some features)
df_rad = pd.read_csv(RADIOMICS_CSV)
print(f"\n[radiomics] shape={df_rad.shape}, columns head: {list(df_rad.columns)[:6]}")

# Save raw inspection for debugging
(OUT_DIR / "_schema_cluster.json").write_text(
    json.dumps({"shape": list(ca.shape), "columns": list(ca.columns),
                "head": ca.head(5).to_dict(orient="records")},
               ensure_ascii=False, indent=2), encoding="utf-8")
(OUT_DIR / "_schema_clinical.json").write_text(
    json.dumps({"shape": list(df_clin.shape), "columns": list(df_clin.columns),
                "head": df_clin.head(5).astype(str).to_dict(orient="records")},
               ensure_ascii=False, indent=2), encoding="utf-8")
(OUT_DIR / "_schema_radiomics.json").write_text(
    json.dumps({"shape": list(df_rad.shape), "columns": list(df_rad.columns)[:30]},
               ensure_ascii=False, indent=2), encoding="utf-8")

print(f"\nSchemas dumped to {OUT_DIR}")
print("Stage 1 (schema-check) complete. Next stage will align keys and cross-tabulate.")
