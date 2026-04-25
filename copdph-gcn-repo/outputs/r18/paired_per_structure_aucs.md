# R18.A — Same-case paired per-structure disease AUC

Addresses R17 reviewer flag: 'different-N artery 0.749 vs vein 0.777 vs airway 0.797 not interpretable'.
Same-case cohort: drop any row with NaN across ANY structure (all three required to compare).

**Cohort**: n = 190 (PH=163, nonPH=27). 5-fold OOF LR, paired bootstrap 5000 iter.

## Structure AUCs (identical n)

| structure | n_features | AUC | 95% CI |
|---|---|---|---|
| artery | 44 | 0.741 | [0.640, 0.833] |
| vein | 44 | 0.801 | [0.701, 0.886] |
| airway | 44 | 0.790 | [0.697, 0.872] |
| all_three | 132 | 0.769 | [0.652, 0.873] |

## Paired Δ-AUCs (same case set)

| comparison | AUC_a | AUC_b | Δ (a−b) | 95% CI | p (two-sided) |
|---|---|---|---|---|---|
| artery_minus_vein | 0.741 | 0.801 | -0.059 | [-0.169, +0.047] | 0.276 |
| artery_minus_airway | 0.741 | 0.790 | -0.049 | [-0.168, +0.067] | 0.41 |
| vein_minus_airway | 0.801 | 0.790 | +0.010 | [-0.066, +0.089] | 0.811 |
| all_three_minus_artery | 0.769 | 0.741 | +0.028 | [-0.037, +0.098] | 0.403 |
| all_three_minus_vein | 0.769 | 0.801 | -0.031 | [-0.125, +0.057] | 0.517 |
| all_three_minus_airway | 0.769 | 0.790 | -0.021 | [-0.125, +0.075] | 0.721 |

## Interpretation

If `airway-{artery|vein}` Δ has CI excluding 0 with airway > vessel,
airway truly has higher disease AUC at fixed n (not artifact). If CI
spans 0, the R17 different-N apparent airway-LR=0.797 was a sub-cohort
selection artifact. If `all_three-X` is positive sig, multi-structure
fusion adds real value over any single structure.
