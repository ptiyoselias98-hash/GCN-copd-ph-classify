# Phase B1 — Graph Signature Panel

_2026-04-26_

## Output

`outputs/supplementary/B1_graph_signature/graph_signatures_patient_level.csv` (n=290 cases × 172 features)

## Verified counts (from `graph_signature_dictionary.json`)

| | n |
|---|---|
| Total features (clean) | **172** |
| R17 + placeholder artifacts hard-blacklisted | **13** (8 R17 numerical + 5 placeholder) |
| Near-zero-SD / constant features dropped after audit | **0** |

### Source breakdown

| Source | n features | Pipeline |
|---|---|---|
| unified_301_Simple_AV_seg | **78** | artery + vein (290 cases, R20.G build with R19.C binary-mask patch) |
| legacy_R17_HiPaS_style_pipeline | **42** | airway only, suffix `_legR17` (282 cases, vendor HiPaS-style mask) |
| lung_features_v2 | **40** | parenchyma HU + LAA + regional + apical/basal (282 cases) |
| R17.5_TDA_gudhi | **12** | artery + vein H0/H1 persistence (airway pers all-zero, dropped) |

### Category breakdown (Phase B1 spec categories 1-8)

| Category | n |
|---|---|
| 1_basic_graph_size | 17 |
| 2_branching_topology | 18 |
| 3_diameter_length_distribution | 87 |
| 7_lung_parenchyma | 38 |
| 8_TDA_persistence | 12 |
| 0_other | 0 (after 2 codex iterations) |

## Hybrid pipeline strategy (per A0 critical finding)

Airway in unified-301 is structurally trivial (n_nodes=1 across all 290 cases). Hybrid sourcing per phase B1+:
- artery + vein = unified Simple_AV_seg (R20.F+R20.G)
- airway = legacy R17 HiPaS-style (real graph features, suffix `_legR17`)
- lung parenchyma = lung_features_v2 (computed from CT lung mask only, pipeline-independent)
- TDA = R17.5 gudhi persistence (graph topology only, mostly pipeline-independent)

This mixes pipelines for different STRUCTURES. All downstream phase reports must label airway analyses as "from legacy HiPaS-style pipeline" and not co-cluster airway with artery+vein in shared latent space without explicit caveat.

## Cohort flags preserved

5 cohort flags from A0 (C1_all_available / C2_within_contrast_only / C3_borderline_mPAP_18_22 / C4_clear_low_high / C5_early_COPD_no_PH_proxy) + fold_id + measured_mpap + measured_mpap_flag are preserved as columns alongside features for downstream Phase C/D/E filtering.

## Codex DUAL REVIEW history

- Pre-execution: GREEN_LIGHT
- Post-execution iteration 1: REVISE (28 features in 0_other due to lowercase HU/LAA regex bug + TDA mis-attributed to unified due to prefix-order bug)
- Post-execution iteration 2: REVISE (3 placeholder features survived in 0_other from lung_features_v2 source not filtered)
- Post-execution iteration 3: REVISE (commit message numbers 80/38/8 didn't match JSON 78/40/13) — corrected here
- Final state: 172 features, 0 in 0_other, source breakdown verified
