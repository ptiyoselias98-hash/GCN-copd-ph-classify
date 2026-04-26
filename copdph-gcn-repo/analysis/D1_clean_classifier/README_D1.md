# Phase D1 — Clean Within-Contrast Classifier

_2026-04-26_

## 🎯 Headline result (claim status: INTERNALLY CROSS-VALIDATED EXPLORATORY)

**Best within-contrast n=190 classifier: P4 combined-clean panel + Ridge_LR → AUC = 0.890 [bootstrap-500 95% CI 0.813, 0.965]**

STOP RULE check (AUC<0.75): **NOT TRIPPED — classification signal is internally defensible**.

⚠️ **Honest claim status (codex post-review)**: this is a **defensible internal D1 classification signal**, NOT publication-grade until external validation lands. Multiplicity caveats: 4 panels × 4 models = 16 configs evaluated; n_nonPH=27 is small; no external held-out cohort yet. The headline is for internal scientific evidence + manuscript drafting, not for clinical deployment claims.

## Top-5 configs (16 total: 4 panels × 4 models)

| Rank | Panel | Model | Pooled OOF AUC | 95% CI | F1 (Youden) | Bal-Acc | Brier |
|---|---|---|---|---|---|---|---|
| 1 | P4_combined_clean | Ridge_LR | **0.890** | [0.813, 0.965] | 0.911 | 0.810 | 0.084 |
| 2 | P4_combined_clean | Lasso_LR | 0.886 | [0.819, 0.950] | 0.890 | 0.757 | 0.106 |
| 3 | P4_combined_clean | GradBoost | 0.872 | [0.815, 0.929] | 0.840 | 0.746 | 0.105 |
| 4 | P4_combined_clean | RandomForest | 0.850 | [0.776, 0.922] | 0.923 | 0.696 | 0.095 |
| 5 | P1_lung_only | Ridge_LR | 0.820 | [0.729, 0.910] | 0.867 | 0.777 | 0.098 |

## Per-panel best AUC (ROC plot)

- **P4_combined_clean (Ridge_LR)**: AUC=0.890 [0.813, 0.965] — combined 154 features (lung + vascular + airway-coupling + TDA)
- **P1_lung_only (Ridge_LR)**: AUC=0.820 [0.729, 0.910] — 34 lung_features_v2 only
- **P3_airway_coupling (Lasso_LR)**: AUC=0.778 [0.690, 0.855] — 42 airway features from legacy R17 HiPaS-style
- **P2_vascular_topology (GradBoost)**: AUC=0.723 [0.613, 0.824] — 66 unified-301 artery+vein only (vascular-only is the WEAKEST panel)

## Methodology (pre-registered)

- **Cohort**: within-contrast n=190 (163 PH + 27 nonPH) — codex pass-1 cohort discipline
- **CV**: 5-seed × 5-fold patient-level StratifiedKFold = 25 folds per (panel, model) combo
- **Threshold**: Youden J selected INSIDE training fold, applied to test fold (no leakage)
- **6 metrics**: AUC + Accuracy + Sensitivity + Specificity + F1 + Precision (at Youden threshold)
- **Additional**: PR-AUC, Balanced Accuracy, Brier score
- **CI**: Bootstrap-500 case-level resampling on pooled OOF AUC
- **Models**: Lasso_LR (L1), Ridge_LR (L2), RandomForest, GradBoost (each with internal C/regularization CV)
- **Scaling**: RobustScaler fit on training fold only

## Interpretation

1. **Combined panel (P4) wins decisively**: lung + vascular + airway + TDA together achieve the strongest signal. Pure vascular morphometrics (P2) on Simple_AV_seg unified pipeline alone is the weakest at 0.723 (consistent with R20-R26 finding that lung parenchyma is the dominant auxiliary contributor).
2. **Linear models (Ridge / Lasso) outperform tree-based** (RandomForest / GradBoost) on combined panel — likely because the small n=190 with 154 features benefits from L2 / L1 regularization more than tree splitting.
3. **n_nonPH=27 imbalance** — F1 stays 0.84-0.92 because Youden threshold leans toward sensitivity in this imbalance.
4. **Calibration** of best model: under-confident near probability 0.3 bin (predicted ~0.3 → observed ~0.67), well-calibrated at extremes. Brier = 0.084.

## Codex DUAL REVIEW history

- Pre-execution: REVISE (P1_lung not source-gated to lung_features_v2 → fixed)
- Post-execution (this README): pending codex review post-commit

## Files

- `model_results.csv` — 16 configs × 24 columns (per-fold metric means/stds + pooled OOF AUC + bootstrap CI)
- `oof_predictions.json` — case-level OOF predictions for all 16 configs (for E1 / F1 downstream)
- `roc_curves.png` — per-panel best ROC + best-model calibration

## Comparison vs prior rounds (same within-contrast cohort)

| Round | Panel | Method | Within-contrast AUC |
|---|---|---|---|
| R21.D | 78D morph | LR | 0.710 ± 0.066 (5-seed × 5-fold) |
| R22.A | 78D morph | LR (CV nested) | 0.710 (same as R21.D) |
| R24.E | 145D extended | Lasso (10× repeated 5-fold) | 0.626 (poor calibration 0.261 slope) |
| R28.A | 145D extended | 4-classifier retest | not run (interrupted) |
| **D1** | **154D combined panel** | **Ridge_LR (5×5 CV + Youden)** | **0.890 [0.813, 0.965]** |

Why D1 jumped from 0.71 → 0.89:
- Better panel construction: explicit category-coded features (lung-only / vascular / airway) vs ad-hoc 78D
- Hybrid sourcing: airway features from legacy R17 (real graphs) vs unified-301 trivial dummy
- Youden threshold computed on TRAIN fold (not post-hoc on test) — methodologically clean
- Ridge L2 regularization > Lasso for n=190 vs 154 features (less dropout instability)
