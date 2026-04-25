# R18.E — Covariate-adjusted endotype effects (R17 reviewer must-fix)

Within-contrast cohort n=197 (PH=170, nonPH=27). Residualize each
endotype feature against `year` (scanner-era proxy from case_id),
then re-test PH vs nonPH on the residual.

## Covariate-adjustment table

| feature | raw d | raw p | year-adj d | year-adj p | year-rho | LR R² | Δd (raw−adj) |
|---|---|---|---|---|---|---|---|
| artery_tort_p10 | -1.42 | 2e-09 | -1.40 | 1.24e-07 | -0.11 | 0.025 | -0.02 |
| artery_len_p25 | -1.25 | 1.03e-07 | -1.22 | 1.56e-07 | -0.15 | 0.025 | -0.02 |
| artery_len_p50 | -1.23 | 1.02e-07 | -1.21 | 1.92e-07 | -0.15 | 0.026 | -0.02 |
| vein_len_p25 | -1.19 | 2.2e-06 | -1.16 | 2.67e-06 | -0.15 | 0.032 | -0.03 |
| vein_tort_p10 | -1.15 | 5.67e-08 | -1.14 | 0.00271 | -0.22 | 0.005 | -0.02 |
| paren_std_HU | +1.10 | 1.23e-08 | +1.08 | 8.39e-08 | +0.14 | 0.020 | +0.02 |
| paren_mean_HU | +0.66 | 0.00126 | +0.65 | 0.00163 | -0.00 | 0.002 | +0.01 |
| lung_vol_mL | -0.64 | 0.00447 | -0.61 | 0.00538 | -0.19 | 0.045 | -0.03 |

## Interpretation

- If raw d ≈ adjusted d, the endotype effect is INDEPENDENT of
  scanner-era / year confound (robust finding).
- If |raw d − adjusted d| > 0.2, the effect is partially confounded.
- Year-correlation rho gives the magnitude of the confound on the
  raw feature; LR R² gives variance explained by year alone.

## Caveats

1. `year` is the only available scanner-era proxy from case_id.
   True scanner-model / kernel / dose / reconstruction metadata not
   available in this cohort.
2. Within-contrast restriction (n=197) means
   we cannot use is_contrast as covariate (constant=1).
3. mPAP not used as covariate to avoid label-correlation collinearity
   (PH-vs-nonPH is part of the mPAP gradient by definition).
