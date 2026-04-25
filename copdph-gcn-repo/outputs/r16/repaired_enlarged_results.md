# R16.D — Repaired-mask enlarged-stratum probe (vs R15.G inflated)

After lung-mask repair (HU<-300 + top-2-CC; median vol 10.8L → 7.7L).
Cohort n=322; feature set 35 cols.

## A — Within-nonPH protocol probe (REPAIRED vs R15.G)

| metric | R15.G (oversegmented) | R16.D (repaired) | Δ |
|---|---|---|---|
| n | 151 | 151 | +0 |
| LR AUC | 0.908 [0.819, 0.968] | **0.958 [0.924, 0.983]** | +0.050 |
| MLP AUC | 0.914 [0.820, 0.978] | **0.958 [0.923, 0.984]** | +0.044 |

## B — Within-contrast disease (REPAIRED vs R15.G)

| metric | R15.G | R16.D (repaired) | Δ |
|---|---|---|---|
| n | 186 | 186 | +0 |
| LR AUC | 0.847 | **0.816 [0.706, 0.909]** | -0.031 |

## C — Endotype replication (REPAIRED data, within-contrast)

| feature | PH μ±SD | nonPH μ±SD | Δ | MWU p |
|---|---|---|---|---|
| paren_mean_HU | -807.736 ± 55.293 | -844.737 ± 59.463 | +37.001 | 0.00126 |
| paren_std_HU | 94.283 ± 13.745 | 78.631 ± 16.926 | +15.652 | 1.23e-08 |
| paren_LAA_950_frac | 0.072 ± 0.094 | 0.109 ± 0.199 | -0.037 | 0.594 |
| paren_LAA_910_frac | 0.174 ± 0.160 | 0.246 ± 0.229 | -0.072 | 0.129 |
| paren_LAA_856_frac | 0.376 ± 0.221 | 0.516 ± 0.252 | -0.140 | 0.0068 |
| apical_basal_LAA950_gradient | -0.027 ± 0.136 | 0.041 ± 0.136 | -0.068 | 0.00537 |
| lung_vol_mL | 3356.657 ± 1530.834 | 4352.524 ± 1661.573 | -995.867 | 0.00447 |
| vessel_airway_over_lung | 0.301 ± 1.153 | 0.860 ± 3.782 | -0.559 | 0.000398 |
| artery_vol_mL | 350.724 ± 580.968 | 202.930 ± 45.663 | +147.794 | 0.000512 |
| vein_vol_mL | 194.568 ± 82.476 | 2139.266 ± 10039.205 | -1944.698 | 0.36 |

## Verdict

If the repaired-mask LR is **near 0.91** → R15.G's enlarged protocol
decoder is real, not artifact.
If it drops back to **~0.85** → the inflation was driven by oversegmentation.
If it stays HIGH but lower → mixed effect.
