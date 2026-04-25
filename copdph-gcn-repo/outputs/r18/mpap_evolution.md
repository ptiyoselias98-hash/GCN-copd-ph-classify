# R18.B — mPAP 5-stage evolution analysis (PROXY, mPAP-xlsx-resolution PENDING)

**User clinical input 2026-04-25**: plain-scan COPDNOPH ≈ no PH = mPAP 0-10.

Stages: 0=plain-scan nonPH (n=125), 1=contrast nonPH (n=27), 2/3/4=PH split (proxy: random 1/3).

**CAVEAT**: stages 2-4 use random 1/3 split of PH cases as proxy until `mpap_lookup.json` resolved to case_ids via `copd-ph患者113例0331.xlsx`. R19 will redo with true mPAP per case.

## Trend tests across 5 ordered stages

| feature | Spearman ρ | p_spearman | Jonckheere z | p_JT |
|---|---|---|---|---|
| artery_tort_p10 | -0.552 | 2.36e-20 | -7.995 | 1.33e-15 |
| artery_len_p25 | -0.608 | 1.83e-25 | -9.800 | 0 |
| artery_len_p50 | -0.601 | 9.81e-25 | -9.646 | 0 |
| vein_len_p25 | -0.534 | 1.45e-18 | -8.404 | 0 |
| vein_tort_p10 | -0.297 | 3.92e-06 | -3.301 | 0.000964 |
| paren_std_HU | +0.499 | 5.02e-16 | +7.946 | 2e-15 |
| paren_mean_HU | +0.296 | 4.53e-06 | +4.384 | 1.17e-05 |
| paren_LAA_950_frac | +0.121 | 0.0651 | +1.833 | 0.0668 |
| lung_vol_mL | -0.298 | 2.74e-06 | -4.510 | 6.47e-06 |
| apical_basal_LAA950_gradient | -0.230 | 0.000423 | -3.459 | 0.000542 |

## Stage-wise means (mean ± 1.96·SE)

### artery_tort_p10

| stage | n | mean | 95% CI |
|---|---|---|---|
| 0 | 42 | 1.023 | [1.020, 1.027] |
| 1 | 27 | 1.021 | [1.017, 1.026] |
| 2 | 56 | 1.009 | [1.006, 1.012] |
| 3 | 56 | 1.005 | [1.002, 1.008] |
| 4 | 57 | 1.003 | [1.001, 1.005] |

### artery_len_p25

| stage | n | mean | 95% CI |
|---|---|---|---|
| 0 | 42 | 6.309 | [6.024, 6.593] |
| 1 | 27 | 5.319 | [5.038, 5.600] |
| 2 | 56 | 4.583 | [4.357, 4.809] |
| 3 | 56 | 4.390 | [4.219, 4.561] |
| 4 | 57 | 4.191 | [4.018, 4.364] |

### artery_len_p50

| stage | n | mean | 95% CI |
|---|---|---|---|
| 0 | 42 | 10.474 | [10.020, 10.927] |
| 1 | 27 | 8.980 | [8.525, 9.435] |
| 2 | 56 | 7.765 | [7.412, 8.118] |
| 3 | 56 | 7.484 | [7.180, 7.787] |
| 4 | 57 | 7.196 | [6.913, 7.480] |

### vein_len_p25

| stage | n | mean | 95% CI |
|---|---|---|---|
| 0 | 42 | 5.515 | [5.302, 5.727] |
| 1 | 27 | 5.216 | [4.930, 5.503] |
| 2 | 55 | 4.601 | [4.419, 4.783] |
| 3 | 56 | 4.532 | [4.397, 4.667] |
| 4 | 53 | 4.308 | [4.166, 4.450] |

### vein_tort_p10

| stage | n | mean | 95% CI |
|---|---|---|---|
| 0 | 42 | 1.005 | [1.002, 1.008] |
| 1 | 27 | 1.008 | [1.004, 1.012] |
| 2 | 55 | 1.002 | [1.000, 1.004] |
| 3 | 56 | 1.001 | [1.000, 1.002] |
| 4 | 53 | 1.001 | [1.000, 1.002] |

### paren_std_HU

| stage | n | mean | 95% CI |
|---|---|---|---|
| 0 | 46 | 85.688 | [75.535, 95.841] |
| 1 | 27 | 78.631 | [72.247, 85.015] |
| 2 | 54 | 91.736 | [88.270, 95.201] |
| 3 | 55 | 95.649 | [91.545, 99.752] |
| 4 | 50 | 95.531 | [92.136, 98.926] |

### paren_mean_HU

| stage | n | mean | 95% CI |
|---|---|---|---|
| 0 | 46 | -799.890 | [-854.154, -745.626] |
| 1 | 27 | -844.737 | [-867.166, -822.307] |
| 2 | 54 | -809.510 | [-825.880, -793.140] |
| 3 | 55 | -812.895 | [-826.412, -799.378] |
| 4 | 50 | -800.144 | [-814.833, -785.454] |

### paren_LAA_950_frac

| stage | n | mean | 95% CI |
|---|---|---|---|
| 0 | 46 | 0.052 | [0.009, 0.094] |
| 1 | 27 | 0.109 | [0.034, 0.184] |
| 2 | 54 | 0.070 | [0.044, 0.095] |
| 3 | 55 | 0.081 | [0.052, 0.110] |
| 4 | 50 | 0.063 | [0.042, 0.084] |

### lung_vol_mL

| stage | n | mean | 95% CI |
|---|---|---|---|
| 0 | 47 | 5861.067 | [3604.879, 8117.255] |
| 1 | 27 | 4352.524 | [3725.775, 4979.273] |
| 2 | 55 | 3487.120 | [3083.243, 3890.997] |
| 3 | 56 | 3406.332 | [2978.336, 3834.327] |
| 4 | 54 | 3172.264 | [2790.739, 3553.789] |

### apical_basal_LAA950_gradient

| stage | n | mean | 95% CI |
|---|---|---|---|
| 0 | 46 | 0.008 | [-0.005, 0.021] |
| 1 | 27 | 0.041 | [-0.010, 0.093] |
| 2 | 54 | -0.000 | [-0.028, 0.027] |
| 3 | 55 | -0.040 | [-0.080, -0.001] |
| 4 | 50 | -0.042 | [-0.084, -0.000] |

## Caveats

1. PH stages 2-4 are RANDOM 1/3 splits, not true mPAP. Real evolution trends require resolving `mpap_lookup.json` → case_ids (R19 task).
2. Stage 1 (contrast nonPH n=27) is small.
3. Spearman/Jonckheere on ordinal stages with random PH binning can still detect Stage 0/1 → PH structure shifts (the major within-cohort evolution signal), but cannot separate early/moderate/severe PH.
