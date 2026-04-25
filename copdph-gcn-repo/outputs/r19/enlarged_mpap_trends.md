# R19.E — Enlarged 360-cohort mPAP 5-stage trends

Cohort n = 231 (vs R18.B 282 cohort).
Stage counts:

| stage | n |
|---|---|
| 0 | 125 |
| 1 | 27 |
| 2 | 8 |
| 3 | 44 |
| 4 | 27 |

## Trend tests (sorted by |Spearman ρ|)

| feature | n | Spearman ρ | p_spearman | Jonckheere z | p_JT |
|---|---|---|---|---|---|
| lung_vol_mL | 229 | -0.622 | 7.05e-26 | -9.695 | 0 |
| artery_total_len_mm | 225 | +0.577 | 2.1e-21 | +8.638 | 0 |
| artery_len_p90 | 225 | -0.558 | 8.66e-20 | -8.919 | 0 |
| artery_len_mean | 225 | -0.535 | 4.63e-18 | -8.591 | 0 |
| artery_len_p75 | 225 | -0.446 | 2.1e-12 | -7.190 | 6.46e-13 |
| vein_total_len_mm | 223 | +0.430 | 1.79e-11 | +5.950 | 2.67e-09 |
| paren_mean_HU | 225 | +0.377 | 5.43e-09 | +5.654 | 1.57e-08 |
| artery_len_p50 | 225 | -0.320 | 9.28e-07 | -5.230 | 1.7e-07 |
| artery_len_p10 | 225 | -0.301 | 4.2e-06 | -4.963 | 6.93e-07 |
| artery_len_p25 | 225 | -0.282 | 1.78e-05 | -4.686 | 2.78e-06 |
| paren_LAA_950_frac | 225 | -0.258 | 9.19e-05 | -3.804 | 0.000143 |
| paren_std_HU | 225 | -0.251 | 0.000142 | -2.770 | 0.00561 |
| artery_tort_p10 | 225 | -0.159 | 0.0168 | -2.328 | 0.0199 |
| vein_len_p25 | 223 | -0.139 | 0.0381 | -2.517 | 0.0118 |
| vein_tort_p10 | 223 | -0.127 | 0.058 | -1.500 | 0.134 |
| vein_len_p50 | 223 | -0.124 | 0.064 | -2.407 | 0.0161 |
| apical_basal_LAA950_gradient | 225 | +0.110 | 0.101 | +1.643 | 0.1 |

## Comparison to R18.B (282-cohort)

R18.B at n=282 reported (top trends):
- artery_len_p25 ρ=−0.767 p=9e-30
- artery_len_p50 ρ=−0.753
- paren_std_HU ρ=+0.629
- artery_tort_p10 ρ=−0.619

Enlarged cohort tests above. Trends should be at least as strong
(more nonPH-plain Stage 0 cases tighten the lower-mPAP anchor).
