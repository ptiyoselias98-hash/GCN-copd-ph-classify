# R14.B — Multi-structure phenotype clusters

Joins per-case graph aggregates (artery + vein + airway from
`cache_v2_tri_flat`) with lung parenchyma features (R14.A) to
build a single multi-modal feature vector per case, then UMAP +
KMeans clusters them. Each cluster represents a 'typical structural
pattern' on the COPD ↔ COPD-PH spectrum.

## Cohort: full (n=226)

Projection: UMAP | KMeans k=4 | Fig: `E:\桌面文件\图卷积-肺小血管演化规律探索\copdph-gcn-repo\outputs\r14\multistruct_clusters_full.png`

| cluster | n | nonPH | PH | PH% | top features (z-score) |
|---|---|---|---|---|---|
| C0 | 64 | 5 | 59 | 92.2% | l_lung_vol_mL=-1.06; l_paren_HU_p95=+0.93; l_paren_HU_p75=+0.92; l_vessel_airway_over_lung=+0.89; l_paren_mean_HU=+0.87 |
| C1 | 14 | 3 | 11 | 78.6% | g_e2_mean=+2.44; g_x2_mean=+2.39; g_x9_std=+2.20; g_x9_p90=+2.18; g_x2_p90=+2.18 |
| C2 | 76 | 11 | 65 | 85.5% | g_n_nodes=+0.85; g_n_edges=+0.85; g_e0_mean=-0.66; g_x0_std=-0.63; g_x1_std=-0.62 |
| C3 | 72 | 49 | 23 | 31.9% | g_e1_mean=+1.11; g_x1_mean=+1.11; g_x1_p90=+1.06; g_e1_p90=+1.06; l_paren_HU_p95=-0.91 |

## Cohort: contrast_only (n=184)

Projection: UMAP | KMeans k=3 | Fig: `E:\桌面文件\图卷积-肺小血管演化规律探索\copdph-gcn-repo\outputs\r14\multistruct_clusters_contrast_only.png`

| cluster | n | nonPH | PH | PH% | top features (z-score) |
|---|---|---|---|---|---|
| C0 | 54 | 17 | 37 | 68.5% | g_x1_p90=+1.22; g_e1_mean=+1.21; g_x1_mean=+1.21; g_e1_p90=+1.21; l_paren_HU_p95=-1.12 |
| C1 | 60 | 4 | 56 | 93.3% | g_n_nodes=+1.01; g_n_edges=+1.01; l_artery_vol_mL=+0.72; g_x0_std=-0.70; g_e0_p90=-0.65 |
| C2 | 70 | 5 | 65 | 92.9% | l_lung_vol_mL=-0.89; l_paren_mean_HU=+0.66; l_paren_LAA_856_frac=-0.66; l_paren_HU_p75=+0.66; l_paren_HU_p95=+0.65 |
