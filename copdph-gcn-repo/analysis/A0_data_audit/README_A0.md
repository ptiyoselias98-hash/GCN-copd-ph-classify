# Phase A0 — Data Audit & Clean Cohort Definition

_2026-04-26_

## Cohorts defined (clean_cohort_table.csv)

| Cohort | n | PH | nonPH | mPAP-resolved | Protocol |
|---|---|---|---|---|---|
| C1_all_available | 290 | 163 | 127 | 102 | 190 contrast + 100 plain |
| **C2_within_contrast_only** (PRIMARY) | **190** | **163** | **27** | **102** | all contrast |
| C3_borderline_mPAP_18_22 | 12 | 1 | 11 | 12 | all contrast |
| C4_clear_low_high (<20 vs ≥35) | 51 | 26 | 25 | 51 | all contrast |
| C5_early_COPD_no_PH_proxy | 127 | 0 | 127 | 27 | 27 contrast + 100 plain |

## Cache QC (unified-301)

- 290 cases × 87 morph columns (after R20.G build with R19.C binary-mask patch)
- Per-structure n_nodes p50: artery 182, vein 182, airway 182 (similar scale)
- 0 empty graphs in unified-301
- 22 single-node airway cases — flagged for B1 sensitivity
- 282 cases overlap with legacy R17 morph (132 features) + R17.5 TDA (18 features) + lung_features_v2 (51 features) → 145D extended feature universe (used in R25/R26)

## STOP RULE check

✅ No severe cache/mask failure detected. Proceed to B1.

## Decisions for downstream phases

1. **C2 within-contrast n=190** is the primary disease-classification cohort
2. **C4 clear_low_high n=51 (mPAP<20 vs ≥35)** for secondary cleaner-signal classifier
3. **C3 borderline n=12** is too small for standalone classifier; use for descriptive deep-dive only
4. **C5 nonPH-only** for severity-axis projection ("early high-risk") in Phase G
5. mPAP defaults (plain=5.0, contrast=15.0) FORBIDDEN in inferential analyses; only n=102 measured mPAP cases for severity correlation
