# E2 — Parenchyma phenotype cluster (protocol-robust v2 features)

Cases: 252 (label=0: 93, label=1: 159)
Features (n=9): paren_mean_HU, paren_std_HU, paren_LAA_950_frac, paren_LAA_910_frac, paren_LAA_856_frac, apical_LAA_950_frac, middle_LAA_950_frac, basal_LAA_950_frac, apical_basal_LAA950_gradient
Best GMM k by BIC: **k=5**  (BIC: {2: -2917.842160280515, 3: -4203.559803520613, 4: -4246.342033053889, 5: -4529.070833067886})

## Cluster composition

| Cluster | Size | PH count | PH proportion | 95% Wilson CI | Protocol mix | Top-3 |
|---|---|---|---|---|---|---|
| 0 | 46 | 37 | 80.43% | [0.67, 0.89] | contrast=42, plain_scan=4 | apical_LAA_950_frac=+1.19, paren_LAA_910_frac=+1.13, paren_LAA_950_frac=+1.10 |
| 1 | 96 | 67 | 69.79% | [0.60, 0.78] | contrast=76, plain_scan=20 | paren_LAA_910_frac=-0.73, paren_LAA_856_frac=-0.66, paren_LAA_950_frac=-0.49 |
| 2 | 19 | 0 | 0.00% | [0.00, 0.17] | plain_scan=19 | paren_mean_HU=+3.37, paren_std_HU=+3.18, paren_LAA_856_frac=-1.57 |
| 3 | 2 | 0 | 0.00% | [0.00, 0.66] | contrast=1, plain_scan=1 | middle_LAA_950_frac=+8.31, paren_LAA_950_frac=+7.75, apical_LAA_950_frac=+6.59 |
| 4 | 89 | 55 | 61.80% | [0.51, 0.71] | contrast=67, plain_scan=22 | paren_LAA_856_frac=+0.60, paren_mean_HU=-0.40, paren_LAA_910_frac=+0.30 |

**Baseline PH proportion across all clustered cases: 63.10%.**
A cluster is 'PH-enriched' if its proportion CI excludes the baseline.

## Protocol-stratified check

PH proportion within `contrast` cases only (should be ~163/189 ≈ 86% if protocol is balanced):

| Cluster | n_contrast | PH_contrast | PH% (contrast) |
|---|---|---|---|
| 0 | 42 | 37 | 88.10% |
| 1 | 76 | 67 | 88.16% |
| 2 | 0 | 0 | nan% |
| 3 | 1 | 0 | 0.00% |
| 4 | 67 | 55 | 82.09% |

## Figures

![UMAP by label](E2_paren_umap_label.png)

![UMAP by protocol](E2_paren_umap_protocol.png)

## Interpretation

A protocol-robust cluster should show (a) a non-trivially PH-enriched cluster with 95% CI above
the baseline PH rate, and (b) balanced protocol mix within that cluster. If instead the PH-enriched
clusters are dominated by `contrast` cases, this means the parenchyma features still leak protocol —
in which case the vessel-graph contribution (E1) becomes the primary driver of any disease claim.