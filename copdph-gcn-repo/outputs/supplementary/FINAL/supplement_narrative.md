# Phase A→H Supplement Narrative

_Auto-generated 2026-04-26 — protocol-controlled clean signature analysis pipeline_

## 1. Old high AUC was confounded and retired
Sprint 5/6 era reported all-cohort AUC ≈ 0.92-0.94. R20-R23 codex hostile review revealed these contained
~0.91 protocol-decoding signal (within-nonPH plain-vs-contrast AUC = 0.912). All-cohort AUC was retired
in R22 paper repositioning and is NOT cited as biology in this work.

## 2. In clean within-contrast setting, disease discrimination is harder but defensible
Phase D1 best within-contrast n=190 (163 PH + 27 nonPH) classifier (P4 combined-clean panel + Ridge_LR,
5-seed × 5-fold + Youden train-fold threshold): **AUC = 0.890 [bootstrap-500 95% CI 0.813, 0.965]**.
Per-fold range [~0.7, ~1.0] is honest given n_nonPH=27. STOP RULE NOT tripped.

## 3. Simple volume / PA-Ao do not explain COPD-PH
F1 bucket ablation: removing artery features drops AUC by 0.043 (largest single bucket effect).
But per-patient driver split is heterogeneous (artery 49 / airway 47 / lung 40 / vein 30 / TDA 24 cases).
No single anatomical bucket dominates; multi-structure signature is required.

## 4. Lung heterogeneity + artery/vein topology track mPAP severity
C1 T2 measured-mPAP within-contrast n=102 Spearman + 500-perm null:
- paren_std_HU ρ = +0.664 (perm-p = 0)
- whole_std_HU ρ = +0.635
- vein_persH1_total ρ = -0.594 (TDA loop topology decreases with mPAP)
- vein_persH0_total ρ = -0.561
- paren_HU_p95 ρ = +0.558
22 Holm-significant + 58 FDR-significant features after multiplicity correction.

## 5. Corrected signature panel improves clean AUC
Hybrid panel: 78 unified-301 artery+vein (Simple_AV_seg) + 42 legacy R17 airway (HiPaS-style;
unified airway is structurally trivial — 1-node dummy graph) + 40 lung_features_v2 + 12 TDA = **172 features**
on within-contrast n=190. A0 audit + B1 dictionary + C1 ranking → D1 lift from R21.D 0.710 to 0.890.

## 6. Structural clusters reveal phenotypes ≠ PH labels
E1 winner k=3 GMM (silhouette 0.913, stability ARI 0.969) splits 190 into 183/6/1 clusters
(mainly one large mass + 6 severe-PH outliers + 1 single-case outlier). ARI vs PH label = 0.001
(perm-p = 0.65). Honest finding: corrected signature space does NOT reveal multi-axis phenotype subtypes
beyond a small severe-PH outlier group, given 86% PH prevalence + n_nonPH=27.

## 7. Cross-sectional severity ordering and PH-like structural progression score
G1 score on within-contrast: AUC 0.890; Spearman ρ vs measured mPAP = **+0.710 p = 6.9e-17** (n=102) —
strongest severity correlation in the project. Youden threshold 0.826 gives Sens 0.83 / Spec 0.93 /
PPV 0.985 / NPV 0.47 / LR+ 11.18 / LR− 0.186. Cross-sectional only — NOT incident PH prediction
(no longitudinal scans).

## 8. Critical honest negatives + future work
- **Cross-protocol projection FAILS**: contrast-trained score applied to plain-scan nonPH (n=100) gives
  mean 1.000 (99/100 high-risk top-25%) vs contrast nonPH (n=27) mean 0.016 (0/27). Protocol shift
  artifact dominates. Cross-protocol generalization remains future work.
- **TDA bucket NOT useful**: F1 ablation shows TDA removal IMPROVES AUC by 0.002 (negligible). Demote
  TDA from headline contributor.
- **Phenotype clustering negative**: no robust subtypes beyond severity gradient.
- **External validation absent**: D1 0.890 + G1 ρ=+0.71 are internally cross-validated only.
- **Borderline mPAP 18-22 (n=12)**: model genuinely struggles; 1 PH at mPAP 22 scored 0.14 (predicted nonPH);
  5/11 nonPH at mPAP 18-20 scored >0.6 (model flagged PH-like).
