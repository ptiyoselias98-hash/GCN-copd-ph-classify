# R20.H — Unified-pipeline R18.B verification

Cohort (CONTRAST-ONLY): n=190 (Simple_AV_seg legacy contrast 163 PH + 27 nonPH)
(Full unified-301 cohort had n=290; restricted to contrast to avoid plain-scan vs contrast pipeline-mixing artifact from R19.E)

## Contrast-only PH vs nonPH (within-pipeline within-protocol)

| feature | Cohen's d | MWU p | PH mean | nonPH mean | n_PH | n_nonPH |
|---|---|---|---|---|---|---|
| artery_len_p25 | -0.298 | 0.013 | 5.330 | 5.897 | 163 | 27 |
| artery_len_p50 | -0.473 | 0.00096 | 9.396 | 10.735 | 163 | 27 |
| artery_tort_p10 | -0.370 | 0.032 | 1.017 | 1.026 | 163 | 27 |
| vein_len_p25 | -0.712 | 0.12 | 5.815 | 8.069 | 163 | 27 |


| feature | n_resolved | Spearman ρ | p | n_PH | n_nonPH |
|---|---|---|---|---|---|
| artery_len_p25 | 102 | -0.211 | 0.033 | 75 | 27 |
| artery_len_p50 | 102 | -0.281 | 0.0043 | 75 | 27 |
| artery_tort_p10 | 102 | -0.136 | 0.17 | 75 | 27 |
| vein_len_p25 | 102 | -0.129 | 0.19 | 75 | 27 |

## Verdict

Legacy R18.B (HiPaS pipeline, n=147 within-contrast): ρ = -0.767

**Closes R18 must-fix #2 with PARTIAL/MIXED verdict**: DIRECTION preserved across pipelines (all 4 flagship features show PH < nonPH within Simple_AV_seg unified contrast cohort, artery_len_p25 d=-0.298, artery_len_p50 d=-0.473, vein_len_p25 d=-0.712); BUT magnitude reduced (ρ -0.211 unified vs -0.767 legacy). The vascular remodeling signal is pipeline-independent in DIRECTION but the legacy ρ=-0.767 effect SIZE was Pipeline-specific (HiPaS-style masks).