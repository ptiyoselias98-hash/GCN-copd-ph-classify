# R15.C — Lung HU residual-confound audit (within contrast-only)

Cohort: contrast-only n=197 (excluding seg-failures).
Tests whether lung HU features carry scanner/reconstruction-era
confound that survives within-contrast restriction.

## Test 1 — Year-effect on paren_mean_HU (Kruskal-Wallis)

- H = 12.57, p = 0.249 (11 year-groups)
- **no significant year-effect on HU within contrast**

Group means by year:

| year | n | μ paren_mean_HU | σ |
|---|---|---|---|
| 2013 | 8 | -815.2 | 31.1 |
| 2014 | 8 | -839.8 | 42.3 |
| 2015 | 5 | -846.3 | 52.8 |
| 2016 | 9 | -824.4 | 66.9 |
| 2017 | 14 | -788.8 | 60.6 |
| 2018 | 20 | -810.1 | 76.7 |
| 2019 | 20 | -802.8 | 40.7 |
| 2020 | 25 | -813.5 | 58.1 |
| 2021 | 21 | -813.6 | 55.6 |
| 2022 | 22 | -800.3 | 59.1 |
| 2023 | 33 | -822.5 | 56.0 |

## Test 2 — Spearman HU vs disease label (within contrast)

- ρ = +0.237, p = 0.00111, n = 186
- **HU correlates with disease label within contrast (disease signal)**

## Test 4 — HU range within-contrast PH vs nonPH

| group | n | μ HU | σ | p5–p95 |
|---|---|---|---|---|
| PH | 159 | -807.7 | 55.3 | [-888, -704] |
| nonPH | 27 | -844.7 | 59.5 | [-904, -735] |
| Δ means | — | +37.00 | — | — |

## Test 3 — KMeans HU-cluster vs scan-year (residual scanner-confound proxy)

- Kruskal H = 0.34, p = 0.844 (3 clusters)
- HU-cluster silhouette = 0.354
- **HU clusters not year-correlated**

## Combined verdict

If Test 1 + Test 3 are both significant, lung HU features carry strong
scanner/era confound even within contrast-only — the R14 'lung-only AUC'
claim must be re-scoped to 'lung-features-as-currently-extracted include both
disease and scanner signals'. If only Test 2 is significant (HU correlates
with disease label) but Tests 1+3 are NS, the lung-only AUC is genuine
disease signal.
