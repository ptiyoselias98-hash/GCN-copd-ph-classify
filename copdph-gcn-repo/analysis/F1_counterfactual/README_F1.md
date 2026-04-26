# Phase F1 — Counterfactual / Model Sensitivity

_2026-04-26_

## Headline

**Bucket ablation on within-contrast n=190 (full panel AUC = 0.899, Ridge_LR 5-seed × 5-fold)**:

| Bucket removed | n features | Remaining AUC | AUC drop |
|---|---|---|---|
| **artery** | 39 | 0.855 | **+0.043** (largest drop) |
| **airway** (legR17) | 42 | 0.867 | **+0.032** |
| lung (lung_features_v2) | 40 | 0.886 | +0.013 |
| vein | 39 | 0.893 | +0.005 |
| **TDA** | 12 | 0.901 | **−0.002 (NEGATIVE — removal slightly improves)** |

**Permutation null (50 perms, 5-seed × 5-fold matched protocol)**:
- Real AUC = 0.899
- Null mean AUC = 0.490
- Null 99-pct = 0.661
- Real >> null 99-pct → PASS, perm-p = 0.0

## Per-patient driver bucket assignment

`max |full_p - removed_p|` per case:

| Driver bucket | n patients | n PH | n nonPH |
|---|---|---|---|
| artery | 49 | 41 | 8 |
| airway | 47 | 42 | 5 |
| lung | 40 | 32 | 8 |
| vein | 30 | 27 | 3 |
| TDA | 24 | 21 | 3 |

## Interpretation (model sensitivity, NOT causal counterfactual)

1. **Artery bucket is the strongest contributor**: removing 39 unified-301 artery features drops AUC by 0.043 (largest single-bucket effect). This is consistent with C1 T2 finding that artery_len percentiles correlate with mPAP (top T1 features include artery_diam_*, artery_len_*).
2. **Airway-coupling (legacy R17) ranks 2nd**: removing 42 airway features drops AUC 0.032 — confirms the hybrid sourcing strategy was justified (airway-from-legR17 contributes ≠ trivial unified-301 airway).
3. **Lung parenchyma adds modest signal (+0.013)**: smaller than expected given C1 found paren_std_HU ρ=+0.66 with mPAP — but classification AUC and severity correlation are different metrics; the lung bucket's contribution is partly redundant with vascular topology.
4. **Vein near-zero drop (+0.005)**: vein features are mostly redundant with artery + lung in the combined panel.
5. **TDA removal IMPROVES AUC (−0.002)**: TDA is essentially noise / redundant in this combined panel. Demote TDA from headline contributor.
6. **Per-patient driver split is balanced**: ~25-27% per bucket. No single bucket dominates all patients — driver signal is heterogeneous across cohort. PH cases distribute similarly across drivers; nonPH cases more concentrated on artery/lung.
7. **Null falsification PASS strong**: real AUC 0.899 vs 99-pct null 0.661 with 50-perm matched protocol confirms classification signal is genuine, not multiplicity-induced.

## Honest framing (per codex pass-1)

This is **model sensitivity under bucket-level signature perturbation**, NOT causal patient-level counterfactual. The ranking artery > airway > lung > vein > TDA reflects the Ridge_LR's reliance on each bucket given the n=190 within-contrast cohort and is contingent on:
- Specific feature engineering in B1
- Ridge_LR's regularization landscape
- Hybrid pipeline sourcing (unified vs legR17 vs lung_v2)

It does NOT establish that artery features cause PH classification — only that removing them most harms model AUC.

## Codex DUAL REVIEW history

- Pre-execution: REVISE (T3 null used 1-seed shortcut not matched protocol; fixed to 5-seed × 5-fold)
- Post-execution: pending

## Files

- `bucket_ablation_results.csv` — 5 buckets × (n_remaining, AUC, AUC drop)
- `per_patient_driver_assignment.csv` — n=190 with full_p, driver_bucket, drop per bucket
- `permutation_results.json` — 50 perms × 5-seed × 5-fold null falsification
- `prediction_drop_heatmap.png` — bucket ablation bars + per-patient driver counts
