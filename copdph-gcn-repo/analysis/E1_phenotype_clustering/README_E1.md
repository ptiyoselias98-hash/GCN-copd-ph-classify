# Phase E1 — Phenotype Clustering on Corrected Signatures

_2026-04-26_

## Honest finding

**Clean within-contrast signature space (top-58 features by C1 importance) does NOT reveal natural phenotype splits beyond a few outliers.**

Cluster sweep across KMeans/GMM × k=2..5:

| k | algo | silhouette | DB | ARI vs PH | ARI perm-p | Stab ARI mean | Stab std |
|---|---|---|---|---|---|---|---|
| 2 | KMeans/GMM | 0.988 | 0.005 | 0.052 | 0.155 | 0.840 | 0.367 |
| **3 (winner)** | **GMM** | **0.913** | 0.128 | 0.001 | 0.652 | **0.969** | 0.126 |
| 3 | KMeans | 0.913 | 0.128 | 0.001 | 0.652 | 0.969 | 0.126 |
| 4 | KMeans/GMM | 0.918 | 0.115 | -0.006 | 1.000 | 0.81/0.64 | 0.36/0.41 |
| 5 | KMeans | 0.209 | 0.796 | 0.059 | 0.008 | 0.585 | 0.328 |
| 5 | GMM | 0.241 | 0.760 | 0.133 | 0.001 | 0.764 | 0.227 |

**Pre-specified winner**: max(silhouette × stability_ARI_mean) = k=3 GMM (0.913 × 0.969 = 0.884)

## Cluster composition (winner k=3 GMM)

| Cluster | n | PH% | n_PH | n_nonPH | mPAP mean | mPAP median | n_mpap_resolved |
|---|---|---|---|---|---|---|---|
| C0 | **6** | 100% | 6 | 0 | 49.3 | 53.0 | 3 |
| C1 | **1** | 0% | 0 | 1 | 18.0 | 18.0 | 1 |
| C2 | **183** | 86% | 157 | 26 | 29.0 | 28.0 | 98 |

## Interpretation (post-hoc inspection per codex pass-1)

- **C0 = severe-PH outlier cluster**: 6 cases, all PH, all measured mPAP ≥35 — these are the high-extreme PH cases. Likely a "severe vascular remodeling outlier" phenotype, not a separate disease subtype.
- **C1 = single-case outlier**: probably a feature-extreme case unrelated to disease groups.
- **C2 = the bulk**: 183 cases (96% of cohort), 86% PH (matches cohort prevalence). NO sub-structure resolved within this main mass.

## Verdict

**Per spec**: "Do not require cluster == PH label. Require interpretable structural clusters with clinical enrichment."

**Honest**: at the current signature panel + cohort size + feature engineering, clustering does not reveal **interpretable phenotype subtypes** beyond a small severe-PH outlier group. Proposed "phenotype names" (dense-lung, arterial-pruning, venous-remodeling, etc.) are NOT recoverable from this analysis. Two possible reasons:
1. Cohort is too dominated by PH (86% prevalence within-contrast) for natural sub-clustering
2. n_nonPH=27 is too small to define a separate "near-normal COPD" phenotype mass
3. Feature space (top-58) emphasizes severity gradient, not multi-axis phenotype heterogeneity

## Cohort discipline

- Within-contrast n=190 (asserted at script start; codex pass-1 mitigation against silent cohort drift)
- PH label used ONLY as outcome (ARI null) — never as clustering input
- Cluster names assigned post-hoc after inspecting cluster_signature_heatmap.png

## Codex DUAL REVIEW history

- Pre-execution: REVISE (add `assert len(df)==190` for cohort drift safety; fixed)
- Post-execution: pending — clustering produces honest negative on phenotype subtype claim

## Files

- `cluster_assignments.csv` — n=190 with cluster column + top-20 features
- `cluster_summary.csv` — 8 configs sweep with silhouette/DB/ARI/perm-p/stability
- `cluster_per_cluster_summary.csv` — winner per-cluster: n, PH%, mPAP stats
- `cluster_signature_heatmap.png` — winner clusters × top-30 features (z-score)
- `umap_clusters.png` — PCA-2D scatter + heatmap on top-15 features
- `medoid_cases.csv` — medoid case_id per cluster (closest to centroid)
