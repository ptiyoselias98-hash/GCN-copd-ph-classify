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

## T3. Multi-seed CORAL deconfounding (5-seed) — ORIENTATION-FREE

Per R21 codex review feedback: report orientation-free leakage `max(AUC, 1-AUC)` because below-chance signed AUC means classifier learned *inverted* protocol signal, NOT that protocol information was *removed*.

- protocol-AUC after CORAL (SIGNED) = 0.041
- protocol-AUC after CORAL (orientation-free max(AUC,1-AUC)) = **0.959**
- within-contrast disease AUC after CORAL = **0.710 ± 0.066**

## Verdict

- Pre-CORAL protocol-AUC: 0.912 (already > 0.5)
- Post-CORAL signed AUC: 0.041
- Post-CORAL orientation-free leakage: 0.959
- Information-leakage drop: -0.047
- Disease-AUC change after CORAL: +0.000
- Deconfounded to chance (oriented < 0.6): **False**
- Biology survives (|disease change| < 0.05): **True**

**HONEST NEGATIVE**: orientation-free leakage post-CORAL = 0.959 — feature-level CORAL OVER-CORRECTED, creating an inverted-direction artifact. Information content largely preserved (just sign-flipped). #5 NOT cleanly closed.