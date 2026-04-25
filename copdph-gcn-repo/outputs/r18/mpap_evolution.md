# R18.B — mPAP 5-stage evolution analysis (REAL mPAP, R18.C resolved)

**User clinical input 2026-04-25**: plain-scan COPDNOPH ≈ no PH = mPAP 0-10.
**R18.C resolution**: mpap_lookup_gold.json provides case_id-keyed mPAP for 106 PH cases.

Stages:
- 0=plain-scan nonPH (mPAP 0-10 default, n=125)
- 1=contrast nonPH (mPAP 10-20 borderline, n=27)
- 2=PH borderline (real mPAP <25, n=8)
- 3=PH early-moderate (real mPAP 25-35, n=44)
- 4=PH moderate-severe (real mPAP ≥35, n=27)

Excluded: PH cases without resolved mPAP (typically because patient_sn not in mpap_lookup_gold).

## Trend tests across 5 ordered stages

| feature | Spearman ρ | p_spearman | Jonckheere z | p_JT |
|---|---|---|---|---|
| artery_tort_p10 | -0.619 | 6.7e-17 | -7.354 | 1.92e-13 |
| artery_len_p25 | -0.767 | 8.97e-30 | -10.072 | 0 |
| artery_len_p50 | -0.753 | 3.5e-28 | -9.807 | 0 |
| vein_len_p25 | -0.613 | 2.67e-16 | -7.700 | 1.35e-14 |
| vein_tort_p10 | -0.295 | 0.000321 | -2.749 | 0.00598 |
| paren_std_HU | +0.629 | 1.45e-17 | +8.390 | 0 |
| paren_mean_HU | +0.265 | 0.00116 | +3.110 | 0.00187 |
| paren_LAA_950_frac | +0.218 | 0.00785 | +2.614 | 0.00895 |
| lung_vol_mL | -0.353 | 8.84e-06 | -4.221 | 2.43e-05 |
| apical_basal_LAA950_gradient | -0.117 | 0.158 | -1.327 | 0.185 |

## Stage-wise means (mean ± 1.96·SE)

### artery_tort_p10

| stage | n | mean | 95% CI |
|---|---|---|---|
| 0 | 42 | 1.023 | [1.020, 1.027] |
| 1 | 27 | 1.021 | [1.017, 1.026] |
| 2 | 8 | 1.016 | [1.006, 1.026] |
| 3 | 44 | 1.008 | [1.004, 1.012] |
| 4 | 26 | 1.001 | [0.999, 1.002] |

### artery_len_p25

| stage | n | mean | 95% CI |
|---|---|---|---|
| 0 | 42 | 6.309 | [6.024, 6.593] |
| 1 | 27 | 5.319 | [5.038, 5.600] |
| 2 | 8 | 5.195 | [4.825, 5.566] |
| 3 | 44 | 4.570 | [4.326, 4.813] |
| 4 | 26 | 3.916 | [3.701, 4.130] |

### artery_len_p50

| stage | n | mean | 95% CI |
|---|---|---|---|
| 0 | 42 | 10.474 | [10.020, 10.927] |
| 1 | 27 | 8.980 | [8.525, 9.435] |
| 2 | 8 | 8.719 | [8.085, 9.354] |
| 3 | 44 | 7.784 | [7.380, 8.187] |
| 4 | 26 | 6.735 | [6.369, 7.102] |

### vein_len_p25

| stage | n | mean | 95% CI |
|---|---|---|---|
| 0 | 42 | 5.515 | [5.302, 5.727] |
| 1 | 27 | 5.216 | [4.930, 5.503] |
| 2 | 8 | 4.949 | [4.258, 5.639] |
| 3 | 43 | 4.662 | [4.494, 4.830] |
| 4 | 25 | 4.201 | [4.014, 4.387] |

### vein_tort_p10

| stage | n | mean | 95% CI |
|---|---|---|---|
| 0 | 42 | 1.005 | [1.002, 1.008] |
| 1 | 27 | 1.008 | [1.004, 1.012] |
| 2 | 8 | 1.008 | [1.000, 1.015] |
| 3 | 43 | 1.000 | [1.000, 1.001] |
| 4 | 25 | 1.000 | [1.000, 1.000] |

### paren_std_HU

| stage | n | mean | 95% CI |
|---|---|---|---|
| 0 | 46 | 85.688 | [75.535, 95.841] |
| 1 | 27 | 78.631 | [72.247, 85.015] |
| 2 | 8 | 83.379 | [76.714, 90.044] |
| 3 | 41 | 91.001 | [88.404, 93.597] |
| 4 | 25 | 105.765 | [97.678, 113.852] |

### paren_mean_HU

| stage | n | mean | 95% CI |
|---|---|---|---|
| 0 | 46 | -799.890 | [-854.154, -745.626] |
| 1 | 27 | -844.737 | [-867.166, -822.307] |
| 2 | 8 | -849.194 | [-890.840, -807.548] |
| 3 | 41 | -831.981 | [-847.215, -816.747] |
| 4 | 25 | -790.759 | [-811.224, -770.294] |

### paren_LAA_950_frac

| stage | n | mean | 95% CI |
|---|---|---|---|
| 0 | 46 | 0.052 | [0.009, 0.094] |
| 1 | 27 | 0.109 | [0.034, 0.184] |
| 2 | 8 | 0.159 | [0.037, 0.281] |
| 3 | 41 | 0.107 | [0.068, 0.145] |
| 4 | 25 | 0.076 | [0.043, 0.109] |

### lung_vol_mL

| stage | n | mean | 95% CI |
|---|---|---|---|
| 0 | 47 | 5861.067 | [3604.879, 8117.255] |
| 1 | 27 | 4352.524 | [3725.775, 4979.273] |
| 2 | 8 | 4290.577 | [2887.642, 5693.512] |
| 3 | 42 | 3717.530 | [3224.531, 4210.530] |
| 4 | 27 | 2747.581 | [2189.749, 3305.413] |

### apical_basal_LAA950_gradient

| stage | n | mean | 95% CI |
|---|---|---|---|
| 0 | 46 | 0.008 | [-0.005, 0.021] |
| 1 | 27 | 0.041 | [-0.010, 0.093] |
| 2 | 8 | 0.027 | [-0.069, 0.123] |
| 3 | 41 | -0.029 | [-0.083, 0.025] |
| 4 | 25 | 0.008 | [-0.053, 0.068] |

## Caveats

1. PH cases without resolved mPAP (in xlsx 113 cases minus 106 resolved + those not in extended cohort) are EXCLUDED from analysis.
2. Stage 1 (contrast nonPH n=27) is small.
3. Stage 2 (PH borderline mPAP<25, n=8) is the smallest PH bin.
4. Spearman/Jonckheere are NON-PARAMETRIC trend tests for ordered alternatives — robust to non-normal distributions.
