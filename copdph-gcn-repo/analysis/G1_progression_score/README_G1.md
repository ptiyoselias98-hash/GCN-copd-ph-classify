# Phase G1 — PH-like Structural Progression Score

_2026-04-26_

## 🎯 Headline (cross-sectional severity ordering, NOT longitudinal)

**Within-contrast n=190 OOF severity score (Ridge_LR L2, P4 154-feature panel, 5-seed × 5-fold)**:
- AUC = **0.890** (matches D1 best)
- Spearman ρ vs measured mPAP (n=102) = **+0.710, p=6.9e-17** — **strongest severity correlation in entire project**
- Score saturates near 1.0 for severe PH (mPAP ≥35), drops near 0 for nonPH (mPAP <20)

## Threshold metrics (within-contrast n=190)

| Threshold | thr | Sens | Spec | PPV | NPV | LR+ | LR− |
|---|---|---|---|---|---|---|---|
| Youden_J | 0.826 | 0.828 | 0.926 | 0.985 | 0.472 | 11.18 | 0.186 |
| Sens_90pct | 0.718 | 0.914 | 0.815 | 0.968 | 0.611 | 4.94 | 0.105 |
| Spec_90pct | 0.826 (=Youden) | 0.828 | 0.926 | 0.985 | 0.472 | 11.18 | 0.186 |

Note: Spec_90pct converges to Youden threshold because Youden already achieves 0.926 specificity. NPV=0.47 reflects n_nonPH=27 small.

## ⚠️ Critical honest negative on nonPH C5 projection

Per codex pass-1 stratification by protocol:
- **contrast nonPH (n=27)**: mean projected score = 0.016, top-25% high-risk = 0/27 — **sensible** (these were the training-set nonPH)
- **plain-scan nonPH (n=100)**: mean projected score = 1.000, top-25% high-risk = **99/100** — **PROTOCOL SHIFT ARTIFACT, NOT BIOLOGY**

**This is the exact protocol confound R20-R22 retired**. The contrast-trained score does NOT generalize to plain-scan via simple RobustScaler transform. Plain-scan features sit in a different feature distribution and the model assigns near-1.0 to all of them.

**Implication for paper**: nonPH plain-scan projections ARE NOT VALID early-COPD high-risk identification. Cross-protocol projection is an open problem (out of scope per R22 retirement). Within-contrast nonPH (n=27 → 0/27 high-risk) shows the score is well-calibrated within its training distribution — useful for hypothesis-generating only, not deployment.

## T4 Borderline subgroup (mPAP 18-22, n=12 descriptive only)

| case_id (snippet) | label | mPAP | Score |
|---|---|---|---|
| nonph_huichunyi | 0 | 20 | 0.49 |
| nonph_huangzutian | 0 | 19 | 0.81 |
| nonph_liufanggao | 0 | 19 | 0.80 |
| nonph_lilele | 0 | 18 | 1.00 |
| nonph_lixiangqing | 0 | 19 | 1.00 |
| nonph_shenjuhua | 0 | 18 | 4e-30 |
| nonph_caochenglin | 0 | 18 | 0.68 |
| nonph_xugaofeng | 0 | 20 | 0.09 |
| **ph_zhangxuelin** | **1** | **22** | **0.14** |
| nonph_gushenggui | 0 | 19 | 0.82 |
| nonph_huangkunquan | 0 | 18 | 0.47 |
| nonph_chenxijun | 0 | 19 | 0.45 |

**Notable**: 1 PH borderline case (mPAP=22) has score 0.14 — model would call it nonPH despite the PH diagnosis. 5/11 nonPH borderline cases score >0.6 — model flags them as PH-like. Borderline is genuinely hard for the model at this n=12.

## Cohort discipline

- T1/T3: within-contrast n=190 (asserted at script start)
- Score calibration on C2 within-contrast only
- T2 projection separates contrast vs plain-scan per codex pass-1 → exposes protocol shift
- Borderline C3 n=12 descriptive only

## Codex DUAL REVIEW history

- Pre-execution: REVISE (90%-sens / 90%-spec ROC index selection reversed; fixed — `roc_curve` returns thr DESC, tpr ASC, so first-hit = max-spec given sens≥0.9 etc.)
- Post-execution: pending

## Files

- `severity_score_patient_level.csv` — within-contrast n=190 with OOF severity score
- `early_copd_projection.csv` — all 290 cases with progression score + nonPH high-risk top-25 flag
- `threshold_metrics.csv` — 3 thresholds (Youden + 90%-sens + 90%-spec) × 6 metrics
- `g1_summary.json` — AUC + ρ + thresholds + framing statement
- `severity_axis_plot.png` — 3 panels: score-vs-mPAP scatter + by-label histogram + nonPH C5 projection by protocol
