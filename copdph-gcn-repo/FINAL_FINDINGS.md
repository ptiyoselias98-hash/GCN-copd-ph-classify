# Final Findings — COPD → COPD-PH Structural Evolution

_Last updated 2026-04-26 R19. Within-pipeline evidence consolidated; cross-pipeline (legacy + new100) results require pipeline unification (HiPaS re-segmentation, R20 priority)._

The user's four scientific questions, mapped to currently-verifiable evidence on the **legacy 282-cohort within-pipeline** (R19.F confirmed extractor identity to R18.B → R17 morphometrics → R16 endotype panel). Honest scoring: codex 9.1 / honest 8.7.

---

## Q1. 肺血管影像表型如何演化 (vascular phenotype evolution)?

**~65% answered cross-sectionally; longitudinal pairs absent (cohort limitation).**

### Cross-sectional 5-stage mPAP-ordered evolution (R18.B legacy 282-cohort, verified by R19.F)

Stage definition (n_total = 261 in legacy stratum):
- Stage 0 (n=125): plain-scan nonPH, mPAP 0-10 default per user
- Stage 1 (n=27): contrast nonPH, mPAP 10-20 borderline default
- Stage 2 (n=8): PH borderline real mPAP <25
- Stage 3 (n=44): PH early-moderate real mPAP 25-35
- Stage 4 (n=27): PH moderate-severe real mPAP ≥35

Top monotonic trends (Spearman ρ, Jonckheere-Terpstra p):

| feature | ρ | p (Spearman) | Jonckheere z |
|---|---|---|---|
| **artery_len_p25** | **−0.767** | 9e-30 | −10.07 |
| artery_len_p50 | −0.753 | 4e-28 | −9.81 |
| paren_std_HU | +0.629 | 1e-17 | +8.39 |
| artery_tort_p10 | −0.619 | 7e-17 | +8.00 |
| vein_len_p25 | −0.613 | 3e-16 | +7.70 |
| paren_LAA_950_frac | +0.218 | 0.008 | +1.83 |
| apical_basal_LAA950_gradient | NS p=0.16 | — | — |

**Mechanistic interpretation** (constrained to "consistent with"-style language per R17 reviewer):

- **Artery edge-length percentiles shift uniformly downward** (p10/p25/p50 ALL d ≈ −1.0 to −1.4 in within-contrast PH-vs-nonPH, R16.B Holm-Bonferroni). Combined with **artery_tort_p10 d=−1.42 (LARGEST single-feature effect in project)**, this is consistent with **artery segment-length downshift + low-tortuosity-edge loss**, i.e., vascular remodeling/pruning where small straight arterial branches are lost or shortened.
- **Vein edge-length parallel** (vein_len_p25 d=−1.19), suggesting venous remodeling co-occurs with arterial.
- **TDA H1 vein-loop persistence drops** (vein_persH1_total d=−1.21, R17.5): persistent loop topology in vein networks decreases in PH — independent confirmation of vessel-network simplification at topological level (NOT just edge-length statistics).

### Critical correction
- R16/R17 earlier reported `apical_basal_LAA950_gradient` significant — that was a **random-PH-stage-binning artifact**. With real mPAP staging it is **NS (p=0.16)**. R18.E covariate-adjusted endotype reaffirms.

### Limitations (explicit)
- All evidence is **cross-sectional**. No paired same-patient longitudinal scans available.
- Stage 0/1 nonPH bins use protocol/default mPAP assignments (plain=5, contrast=15), not measured.
- Stage 2 (PH borderline mPAP<25) has only n=8 — early-stage PH characterization is power-limited.

---

## Q2. 肺实质 + 气道辅助作用 (parenchyma & airway auxiliary roles)?

**~55% answered.**

### Parenchyma (R16.B + R18.E covariate-adjusted, n=197 within-contrast)

| feature | Cohen's d (raw / year-adj) | p_holm | direction |
|---|---|---|---|
| **paren_std_HU** | **+1.10 / +1.08** | 1.7e-7 | PH lungs +15.7 HU more heterogeneous (LARGEST endotype effect) |
| paren_mean_HU | +0.66 / +0.65 | 0.013 | PH +37 HU denser |
| lung_vol_mL | −0.64 / −0.61 | 0.036 | PH 996 mL smaller |
| paren_LAA_856_frac | −0.62 | 0.041 | PH actually less mild emphysema |
| paren_LAA_950_frac | NS | 1.0 | total severe emphysema NOT differential |

**Year-residualization** (R18.E proxy for scanner/era confound) shows ALL effects survive with Δd<0.05 — confounds NOT contaminating findings (year-rho all in [−0.22, +0.14]).

### Airway (R17.A within-contrast Holm-Bonferroni)

**0/44 airway features Holm-significant** within contrast cohort. Within-contrast COPD-PH is a **vascular phenotype**; airway changes happen in COPD itself (already shared between PH and nonPH within contrast).

### Lung-vs-graph paired AUC (R18.A, same-case n=190)

Paired bootstrap on identical case set, no protocol confound:
- artery 0.741, vein 0.801, airway 0.790, all_three 0.769, all paired Δ NS (p>0.27)
- (lung+graph) − graph: Δ=+0.085 [+0.029, +0.148], **p=0.0008** ← lung adds genuinely complementary signal
- lung-only > graph-only: Δ=+0.062 [−0.031, +0.160], p=0.19 NS ← marginal-CI artifact (R14 claim retracted)

**Defensible claim**: parenchyma adds significant complementary signal to vascular graph for disease classification, but neither modality dominates standalone.

---

## Q3. 早期 PH (early-stage 25-35 mPAP) 的肺血管改变?

**~40% answered.**

Stage 2 (PH borderline mPAP<25, n=8) and Stage 3 (PH early-moderate mPAP 25-35, n=44) are partially separable from nonPH on the same monotonic features as severe PH (Stage 4, n=27). The Spearman trend across all 5 stages survives Jonckheere-Terpstra; this means **the same vessel/parenchyma direction of change starts early** (Stage 2 effect partial) and intensifies through Stage 4.

**Cannot yet say**: which feature break-points distinguish "pre-PH" from "early PH" most cleanly. Stage 2 n=8 is too small for change-point detection. Need either (a) larger early-PH cohort or (b) longitudinal pairs.

---

## Q4. 系统化定量 + 纵向演化分析?

**Quantification ~75%; longitudinal ~5%.**

### Quantification done
- 282 cases × 132 per-structure morphometric features (R17, audited at R18.A)
- 282 cases × 18 TDA persistence features (R17.5, gudhi 3.10)
- 282 cases × ~50 lung parenchyma features (R16/lung_features_v2)
- Multiplicity-corrected effect sizes (R16.B, R18.F unified panel: 26 Holm-sig features across 3 modalities)
- Covariate-adjusted (R18.E year-residualization, all effects survive Δd<0.05)
- Cross-sectional 5-stage mPAP trajectory (R18.B/C, real-mPAP from `mpap_lookup_gold.json` for 106 PH; user-default for 158 plain-scan)

### Longitudinal
- **0% available** — no paired same-patient scans in current cohort. Cohort-level limitation, not solvable by code.

---

## Score-debt audit (post-R19, codex 9.1 / honest 8.7)

5 honest-debt items remaining (R20+ priority):
1. ✅ R19.A lung overlay gallery — DONE (R19)
2. 🔄 R19 DDPM training in flight (epoch ~14/30); R19.G inference script staged
3. ❌ Embedding-level enlarged-stratum probe — blocked on pipeline unification
4. ❌ HiPaS re-segmentation of 38 + new100 — pending
5. ❌ Multi-seed CORAL on enlarged stratum — blocked on pipeline unification

**Path to 9.5**: complete DDPM evaluation + unify pipelines (HiPaS or principled stratification) + close ≥2 must-fix in R20.

---

## What R20 must do (priority-ordered)

1. **Auto-trigger R19.G DDPM inference** when training completes (~3h ETA). Will close R18 must-fix #1.
2. **HiPaS re-segmentation of new100** — most-actionable pipeline-unification task. Requires HiPaS model checkpoint accessible (may need to scp + run on remote GPU 1). Closes #4 + unblocks #3 + #5.
3. **Embedding-level enlarged probe** after pipeline unification — closes #3.
4. **Multi-seed CORAL on enlarged stratum** — closes #5.
5. **Codex final review** for ≥9.5 promotion.

If all 5 close: estimated honest score **9.4-9.6**, contingent on external/temporal validation (still missing — cohort-level constraint).

---

## Reading guide

- README.md: round-by-round score table + figure embeds
- HONEST_SCORE_AUDIT.md: debt-corrected scoring methodology
- RESEARCH_ROADMAP.md: 5-axis gap analysis
- This document (FINAL_FINDINGS.md): consolidated science answers to user's 4 questions
- outputs/r{N}/*.md: per-round detailed artifacts with full statistics
