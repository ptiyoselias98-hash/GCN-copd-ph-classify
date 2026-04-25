# R13.4 — CORAL within-nonPH protocol probe (vs corrected-GRL R11)

Excluded 38 cases on seg-failure list (R13.2b).

## Full cohort (legacy R11/R12 denominator, uncorrected)

| λ | n | LR AUC [95% CI] | MLP AUC [95% CI] |
|---|---|---|---|
| 0.0 | 80 | 0.878 [0.788, 0.950] | 0.864 [0.773, 0.937] |
| 1.0 | 80 | 0.772 [0.656, 0.873] | 0.827 [0.723, 0.910] |
| 5.0 | 80 | 0.803 [0.696, 0.897] | 0.845 [0.754, 0.920] |
| 10.0 | 80 | 0.818 [0.716, 0.906] | 0.815 [0.714, 0.900] |

## Corrected cohort (excluding seg-failures)

| λ | n_full → n_corrected (excluded) | LR AUC [95% CI] | MLP AUC [95% CI] |
|---|---|---|---|
| 0.0 | 80 → 68 (12) | 0.891 [0.796, 0.967] | 0.878 [0.777, 0.961] |
| 1.0 | 80 → 68 (12) | 0.791 [0.673, 0.892] | 0.793 [0.673, 0.893] |
| 5.0 | 80 → 68 (12) | 0.827 [0.719, 0.920] | 0.825 [0.720, 0.916] |
| 10.0 | 80 → 68 (12) | 0.801 [0.684, 0.905] | 0.817 [0.704, 0.912] |

## Comparison vs corrected-GRL R11 baseline

R11 corrected-GRL @ seed=42 (within-nonPH protocol LR, n=80, full):
  λ=0: 0.840 | λ=1: 0.894 | λ=5: 0.842 | λ=10: 0.790

R12 cross-seed pooled (n=80, full):
  λ=0: 0.867 | λ=1: 0.902 | λ=5: 0.886 | λ=10: 0.873

If CORAL drives the corrected-cohort LR AUC ≤0.60 with upper-CI ≤0.65,
we have broken the protocol floor and R14 expands to seeds {1042, 2042}.
Otherwise, the negative-result is sharper than R12 (now confirmed across
two distinct deconfounder families on the same cohort), and the path to
≥9.5 requires 345-cohort ingestion + segmentation re-runs.
