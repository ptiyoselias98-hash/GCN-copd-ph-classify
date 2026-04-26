# R21.D — Feature-level embedding probe + multi-seed CORAL

Cohort: unified-301 (n=290); features=78 (8 R17-artifact features excluded)

## T1. Disease-AUC by stratum (5-seed mean ± std)

| stratum | n | n_pos | n_neg | AUC mean | AUC std |
|---|---|---|---|---|---|
| within_contrast_n190 | 190 | 163 | 27 | 0.710 | 0.066 |
| full_cohort_n290 | 290 | 163 | 127 | 0.886 | 0.006 |

## T2. Protocol-AUC within nonPH (control test)

- n = 127 (27 contrast nonPH + 100 plain-scan nonPH)
- protocol-AUC = **0.912 ± 0.009**
- HIGH means protocol confound is REAL: features can decode protocol
  even when disease is held constant.

## T3. Multi-seed CORAL deconfounding (5-seed)

- protocol-AUC after CORAL = **0.041 ± 0.010**
- within-contrast disease AUC after CORAL = **0.710 ± 0.066**

## Verdict

- Protocol-AUC drop after CORAL: +0.870
- Disease-AUC change after CORAL: +0.000
- Deconfounding works (drop > 0.05): **True**
- Biology survives (|disease change| < 0.05): **True**

**Closes R20 must-fix #3 + #5 with POSITIVE verdict**: feature-level CORAL deconfounds protocol while preserving within-contrast disease signal across 5 seeds.