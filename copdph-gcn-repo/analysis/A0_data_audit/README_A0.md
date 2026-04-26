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

## Cache QC (unified-301) — corrected after codex DUAL-REVIEW pass-1

- 290 cases × 87 morph columns (after R20.G build with R19.C binary-mask patch)
- **Artery**: n_nodes p10/p50/p90 = 83/167/606, n_edges p10/p50/p90 = 150/319/1200, 0 empty / 0 single-node — VALID
- **Vein**: n_nodes p10/p50/p90 = 58/150/374, n_edges p10/p50/p90 = 95/271/698, 0 empty / 0 single-node — VALID
- **Airway**: n_nodes = 1 / n_edges = 0 / n_branches = 0 for **ALL 290 cases** — TRIVIAL placeholder graph (Simple_AV_seg pipeline only segments artery + vein + lung; airway is NOT segmented in unified pipeline). See critical finding section below.
- 282 cases overlap with legacy R17 morph (132 features, 44 real airway features) + R17.5 TDA (18 features, airway H0/H1 also all-zero) + lung_features_v2 (51 features) → 145D extended feature universe (used in R25/R26)

## STOP RULE check — CRITICAL FINDING (2026-04-26 codex DUAL REVIEW pass-1)

⚠️ **Airway graph in unified-301 is STRUCTURALLY TRIVIAL** — codex flagged inconsistency with my initial summary. Verification confirms:
- airway_n_nodes = 1 for ALL 290 cases (mean=1.0, std=0.0, p10/p50/p90=1/1/1)
- airway_n_edges = 0 for ALL 290 cases
- airway_n_branches = 0 for ALL 290 cases

**Root cause**: Simple_AV_seg pipeline's released checkpoints are `lung.pth` + `main_AV.pth` (artery+vein only) — airway is NOT segmented. The "airway_*" columns in `morph_unified301.csv` are placeholders from a 1-node dummy graph.

**Implication for prior rounds**:
- R24.D "airway too sparse n=6 features" was actually based on trivial-graph dummy features
- R26.A modality ablation airway contribution was junk (TDA airway H0/H1 also all-zero per R20.D audit)
- Any "airway" claim in R24-R27 should be retracted or relabelled

**Implication for Phase B1 onward**:
- Use **legacy R17 morphometrics** (`outputs/r17/per_structure_morphometrics.csv`, 282 cases × 132 features, **44 real airway features**) for ANY airway analysis
- Use **unified-301 Simple_AV_seg** (`outputs/r20/morph_unified301.csv`) for artery + vein only
- Hybrid feature panel: artery/vein from unified-301 + airway from legacy R17 + lung from `lung_features_v2.csv` + TDA from R17.5
- Cohort intersection: 212 cases (per R25.A); within-contrast subset of those = 190
- Pipeline-mixing risk acknowledged: airway-from-legacy + artery/vein-from-unified mixes HiPaS-style and Simple_AV_seg masks for different structures. This is acceptable IFF airway analyses are reported separately and not combined with artery/vein in shared latent space without explicit caveat.

**Decision**: proceed to B1 with the hybrid strategy + airway-from-legacy clearly labelled. STOP RULE not fully tripped (artery + vein + lung are valid; airway has a viable fallback path).

## Decisions for downstream phases

1. **C2 within-contrast n=190** is the primary disease-classification cohort
2. **C4 clear_low_high n=51 (mPAP<20 vs ≥35)** for secondary cleaner-signal classifier
3. **C3 borderline n=12** is too small for standalone classifier; use for descriptive deep-dive only
4. **C5 nonPH-only** for severity-axis projection ("early high-risk") in Phase G
5. mPAP defaults (plain=5.0, contrast=15.0) FORBIDDEN in inferential analyses; only n=102 measured mPAP cases for severity correlation
