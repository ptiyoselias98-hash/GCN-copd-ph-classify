# Phase A→H Limitations

1. **Cohort size**: within-contrast n=190 (163 PH + 27 nonPH). n_nonPH=27 fundamentally limits
   the precision of any classifier metric and the power of phenotype clustering. Bootstrap CIs are
   wide as a result, not a methodological flaw.

2. **PH prevalence imbalance**: 86% PH within-contrast distorts cluster discovery and inflates
   Brier-style calibration. NPV at Youden threshold = 0.47 reflects this prevalence saturation.

3. **No external validation cohort**: all D1, G1, F1 results are internally cross-validated.
   Codex post-execution review explicitly demoted "publication-grade" → "internally cross-validated
   exploratory".

4. **No paired longitudinal scans**: cross-sectional severity ordering only. The G1 progression
   score should NOT be cited as longitudinal evolution or incident PH prediction.

5. **Protocol heterogeneity**: cross-protocol projection (G1 T2) FAILS. Plain-scan vs contrast
   feature distribution differences swamp disease signal. R22 retired all cross-protocol classifier
   claims; this remains the case.

6. **Hybrid pipeline mixing**: airway features come from legacy R17 HiPaS-style (282 cases),
   artery/vein from unified-301 Simple_AV_seg (290 cases), lung from lung_features_v2 (282 cases).
   This is functionally necessary because Simple_AV_seg has no airway model, but means airway analyses
   are tied to a different segmentation pipeline than artery/vein.

7. **Pruning slope alpha (C1 T4)**: estimated from 3-5 diameter percentile bins per patient, not
   from full skeleton diameter histogram. Sensitivity across 3 bin schemes shows alpha is stable
   1.5-1.7, but this is not a high-resolution pruning measurement.

8. **mPAP defaults forbidden in inferential analyses**: only n=102 measured-mPAP cases used in
   T2 / G1 ρ correlations. Plain-scan nonPH (default mPAP=5) and contrast nonPH without measured
   mPAP (default 15) excluded from correlation tables — they appear only in qualitative cohort
   stratification (A0 cohorts C3/C4/C5).

9. **Model-selection multiplicity**: D1 evaluated 4 panels × 4 models = 16 configs. Best AUC 0.890
   is the post-hoc maximum; the unbiased estimate is closer to mean of within-panel best (~0.85).

10. **TDA from legacy R17**: only artery + vein H0/H1 persistence (12 features). Airway TDA was
    all-zero in R17.5 audit; retired. F1 ablation shows TDA removal slightly IMPROVES AUC, suggesting
    these features add noise rather than signal in the combined panel. They survive in C1 individual
    correlation analyses (vein_persH1_total ρ=-0.594) but should not be cited as classifier-relevant.
