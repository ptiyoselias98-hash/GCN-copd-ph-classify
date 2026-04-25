# R14.E — Multi-seed CORAL + MMD within-nonPH protocol probe (corrected n=68)

Excluded 38 cases on seg-failure list (R13.2b). All AUCs are
5-fold OOF LR/MLP on per-fold encoder embeddings; disease AUC is the per-fold
5-fold mean from `sprint6_results.json` on remote.

## CORAL multi-seed aggregation (corrected, n=68 within-nonPH)

| λ | n_seeds | Protocol LR (μ ± SD) | Protocol MLP (μ ± SD) | Disease AUC (μ ± SD) |
|---|---|---|---|---|
| 0.0 | 3 | 0.862 ± 0.043 | 0.870 ± 0.033 | 0.934 ± 0.002 |
| 1.0 | 3 | 0.710 ± 0.084 | 0.752 ± 0.052 | 0.932 ± 0.001 |
| 5.0 | 3 | 0.738 ± 0.082 | 0.770 ± 0.080 | 0.933 ± 0.001 |
| 10.0 | 3 | 0.754 ± 0.066 | 0.758 ± 0.078 | 0.931 ± 0.001 |

## Per-seed breakdown

| config | n | LR | LR CI95 | MLP | MLP CI95 | disease μ |
|---|---|---|---|---|---|---|
| coral_l0.0_s42 | 68 | 0.891 | [0.796, 0.967] | 0.878 | [0.777, 0.961] | 0.9340002970885324 |
| coral_l0.0_s1042 | 68 | 0.812 | [0.701, 0.908] | 0.833 | [0.730, 0.920] | 0.9362589126559715 |
| coral_l0.0_s2042 | 68 | 0.882 | [0.796, 0.952] | 0.898 | [0.813, 0.964] | 0.9316971182412359 |
| coral_l1.0_s42 | 68 | 0.791 | [0.673, 0.892] | 0.793 | [0.673, 0.893] | 0.9330020796197267 |
| coral_l1.0_s1042 | 68 | 0.714 | [0.577, 0.842] | 0.694 | [0.552, 0.824] | 0.9319741532976826 |
| coral_l1.0_s2042 | 68 | 0.624 | [0.471, 0.761] | 0.769 | [0.635, 0.887] | 0.9319318181818183 |
| coral_l5.0_s42 | 68 | 0.827 | [0.719, 0.920] | 0.825 | [0.720, 0.916] | 0.9330087641117053 |
| coral_l5.0_s1042 | 68 | 0.723 | [0.588, 0.848] | 0.806 | [0.690, 0.902] | 0.9318917112299465 |
| coral_l5.0_s2042 | 68 | 0.665 | [0.524, 0.789] | 0.679 | [0.536, 0.805] | 0.9335635769459298 |
| coral_l10.0_s42 | 68 | 0.801 | [0.684, 0.905] | 0.817 | [0.704, 0.912] | 0.9310895721925133 |
| coral_l10.0_s1042 | 68 | 0.679 | [0.538, 0.816] | 0.670 | [0.528, 0.801] | 0.9300935828877005 |
| coral_l10.0_s2042 | 68 | 0.783 | [0.661, 0.889] | 0.788 | [0.660, 0.894] | 0.9324413250148545 |

## MMD pilot (single seed)

| config | n | LR | LR CI95 | MLP | disease μ |
|---|---|---|---|---|---|
| mmd_l1.0_s42 | 68 | 0.860 | [0.752, 0.951] | 0.895 | 0.9260814022578728 |
| mmd_l5.0_s42 | 68 | 0.644 | [0.498, 0.785] | 0.781 | 0.8545424836601307 |

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
