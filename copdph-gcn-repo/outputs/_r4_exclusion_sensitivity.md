# R4.4 Exclusion sensitivity — retain 27 placeholder nonPH with degraded features

Cohort A = current behavior (placeholder vessels dropped).
Cohort B = include them; parenchyma features still valid because lung.nii.gz
          is intact; paren_* just equals whole_lung for placeholder cases
          (no vessels to subtract).

| Cohort | Feat set | n | disease LR full (CI) | disease LR contrast (CI) |
|---|---|---|---|---|
| `A_excluded_paren_only` | - | 231 | 0.862 [0.835, 0.886] | 0.858 [0.784, 0.932] |
| `A_excluded_paren_plus_spatial` | - | 231 | 0.871 [0.854, 0.888] | 0.851 [0.770, 0.932] |
| `B_included_paren_only` | - | 252 | 0.870 [0.826, 0.906] | 0.860 [0.763, 0.950] |
| `B_included_paren_plus_spatial` | - | 252 | 0.879 [0.827, 0.913] | 0.855 [0.759, 0.951] |

## Delta (B − A)

| Feat set | Δn_cases | Δ disease full LR | Δ disease contrast LR |
|---|---|---|---|
| `paren_only` | +21 | +0.008 | +0.002 |
| `paren_plus_spatial` | +21 | +0.008 | +0.004 |

**Max |Δ| on disease-contrast AUC: 0.004**. If this is smaller
than the bootstrap CI half-width (~0.05 on contrast-only 186-case subset),
the exclusion choice is not driving the disease claim — i.e. the claim is
robust to the exclusion rule.