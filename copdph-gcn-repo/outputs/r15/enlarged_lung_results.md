# R15.G — Enlarged-stratum lung-feature analyses (n=360 ingested cohort)

After seg-failure exclusion: 322 cases. Feature set: 40 lung-parenchyma features.

## A — Within-nonPH protocol probe (lung features, n=151)

Contrast: 27, Plain-scan: 124

- **LR AUC = 0.908 [0.819, 0.968]**
- **MLP AUC = 0.914 [0.820, 0.978]**

_R12 baseline (n=80): LR=0.853 [0.722, 0.942]_

## B — Within-contrast disease classifier (lung-only, n=186)

PH: 159, nonPH: 27

- **LR AUC = 0.847 [0.755, 0.923]**

_R14 baseline (n=184): LR=0.844 [0.754, 0.917]_

## C — Endotype replication (within-contrast PH vs nonPH)

| feature | PH (μ±SD) | nonPH (μ±SD) | Δ | MWU p |
|---|---|---|---|---|
| paren_mean_HU | -807.736 ± 55.293 | -844.737 ± 59.463 | +37.001 | 0.001262156174430707 |
| apical_basal_LAA950_gradient | -0.027 ± 0.136 | 0.041 ± 0.136 | -0.068 | 0.005374330935766128 |
| paren_LAA_950_frac | 0.072 ± 0.094 | 0.109 ± 0.199 | -0.037 | 0.5936585733747652 |
| vessel_airway_over_lung | 0.275 ± 1.166 | 0.860 ± 3.782 | -0.585 | 0.0007375350946647481 |
| lung_vol_mL | 3380.326 ± 1516.326 | 4352.524 ± 1661.573 | -972.198 | 0.005247588252976914 |
| artery_vol_mL | 260.961 ± 98.306 | 202.930 ± 45.663 | +58.031 | 0.0008966196357216711 |
| vein_vol_mL | 201.910 ± 74.624 | 2139.266 ± 10039.205 | -1937.356 | 0.526038934237977 |

## New-ingestion sanity check

- 78 new cases ingested
- All label=0: True
- All plain_scan: True
- Mean paren_mean_HU: -878.0
- Mean paren_LAA_950_frac: 0.2380
