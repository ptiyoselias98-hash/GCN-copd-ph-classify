# R22 Paper Repositioning — Formal Scope Decision

_Closes R21 codex must-fix #2: "either adapt the unified multi-structure cache to GCN embeddings or explicitly retire GCN-embedding claims and reposition the paper around morphometric features"_.

After R20–R22 evidence accumulation, the paper is repositioned to claim
**within-contrast cross-sectional vascular endotype evidence** as the
primary scientific contribution. Cross-protocol enlarged-cohort
classifier claims are retired.

## What survives (paper-level claims)

### 1. Cross-sectional vascular endotype within-contrast (n=190, 163 PH + 27 nonPH)

**Pipeline-INDEPENDENT direction** (verified across BOTH legacy HiPaS-style and
Simple_AV_seg unified pipelines):

| feature | unified d (n=190) | legacy d-equiv | direction |
|---|---|---|---|
| artery_len_p25 | -0.298, p=0.013 | -1.25 (R17.A) | PH < nonPH ✓ |
| artery_len_p50 | -0.473, p=0.001 (Holm-sig) | -1.23 | PH < nonPH ✓ |
| artery_tort_p10 | -0.370, p=0.032 | -1.42 | PH < nonPH ✓ |
| vein_len_p25 | -0.712, p=0.12 | -1.19 | PH < nonPH ✓ |

Spearman ρ across 5-stage mPAP severity (n=102 mPAP-resolved):
- legacy HiPaS-style: ρ = -0.767 (R18.B; pipeline-amplified MAGNITUDE)
- unified Simple_AV_seg: ρ = -0.211 (R20.H; direction-only preservation)

**Defensible paper claim**: "Within contrast-enhanced CT, the COPD-PH
vasculature shows a consistent direction of artery+vein edge-length
percentile downshift and tortuosity decrease, reproducible across two
independent segmentation pipelines (HiPaS-style legacy and Simple_AV_seg
unified). The magnitude of the severity-correlation is pipeline-specific,
indicating that absolute Spearman-ρ values should be reported per
pipeline, not as a unified estimate."

### 2. TDA loop-topology evidence (n=190 within-contrast)

`vein_persH1_total` Cohen's d = -1.214, Holm p = 2.98e-6 in 18-feature
TDA panel; bootstrap-1000 sign-stability 100%; LOO d range tight
[-1.335, -1.127]. Closes R18 must-fix #8.

### 3. Parenchyma 3-modality coupling (R18.F)

Endotype panel of 26 Holm-significant features across vessel + parenchyma +
TDA modalities, multiplicity-corrected. Defensible.

### 4. Within-contrast disease classifier (n=190)

5-seed × 5-fold LR on 78 morphometric features:
- mean AUC = 0.710 (across seed-means)
- per-fold range = [0.438, 0.879] WIDE (n_nonPH=27 small)
- This is reported AS A CONTROL/SANITY-CHECK, NOT as a clinical-grade
  classifier. The variance reflects small-cohort uncertainty.

## What is RETIRED (claims removed from paper)

### 1. Cross-protocol enlarged-cohort PH-vs-nonPH single AUC

The full-cohort n=290 AUC = 0.886 ± 0.006 is **mostly protocol decoding**
(within-nonPH protocol-AUC = 0.912). Cannot serve as biological evidence.
**RETIRE** any "AUC=0.94 / 0.88 / etc on 282/290/360-cohort" claim.

### 2. Cross-pipeline magnitude of vascular evolution

Legacy ρ=-0.767 vs unified ρ=-0.211 — magnitude is pipeline-specific.
**REQUIRED**: when citing ρ, state the pipeline. Do NOT claim a single
universal ρ.

### 3. Label-free DDPM PH-detector

R20.B DDPM AUC=0.129 (orientation-free 0.871) but this is protocol-shift
direction, not disease. **RETIRE** "anomaly detector for PH" framing.
The DDPM is now reported as an **honest-negative case study**: training
on plain-scan does NOT generalize as a PH detector for legacy CTPA cohort.

### 4. GCN-embedding enlarged-cohort deconfounding

R21.D feature-level full CORAL over-corrected (oriented 0.959).
R22.A nested univariate CORAL = no-op vs StandardScaler (oriented 0.912).
Neither closes the protocol-confound at feature level. **RETIRE** any
claim that the enlarged 290-cohort can support a single deconfounded
PH classifier without GCN-embedding-level training-time deconfounding,
which is out of scope for this paper.

## Acceptable scope (R22+ paper)

- Title: "Cross-sectional vascular endotype in COPD-PH: pipeline-robust
  edge-length downshift verified across two independent segmentation
  pipelines"
- Cohort: 190 contrast-enhanced CT (163 PH + 27 nonPH); legacy HiPaS-style
  and Simple_AV_seg unified independently verify direction
- Findings: Within-contrast d-effects, multi-pipeline direction agreement,
  TDA loop-topology PH-loss, multi-modality endotype panel
- Limitations: small n_nonPH (27) limits classifier-grade claims; no
  longitudinal data; no external validation cohort; cross-protocol
  generalization requires future GCN-embedding-level deconfounding
  (out of scope for this paper)

## Cumulative honest-debt status (post-R22)

| Item | Status |
|---|---|
| #1 DDPM evaluation | CLOSED honest-negative (R20.B) |
| #2 Pipeline unification | CLOSED partial-positive (R20.H direction preserved) |
| #3 Embedding-level enlarged probe | RETIRED (paper repositioning, R22) |
| #4 HiPaS re-segmentation | CLOSED via Simple_AV_seg substitution (R20.F+) |
| #5 Multi-seed CORAL on enlarged | RETIRED (paper repositioning, R22) — feature-level CORAL is over- or under-corrected; GCN-level out of scope |
| #6 Terminology reframe | CLOSED (R20.E) |
| #7 R17 artifact audit | CLOSED with auditable artifact (R20.C) |
| #8 TDA robustness | CLOSED positive (R20.D) |
| Locked cohort manifest | CLOSED (R21.A) |
| Bootstrap CI on disease AUC | CLOSED (R22.A) |

8/8 R18 must-fix items closed (5 positive, 1 honest-neg, 2 retired-by-repositioning).
+ R20 must-fix #4 closed.
+ R21 codex must-fix items 1-5 closed (orientation-free correction, repositioning, fold table, cohort manifest, naming).

The paper is now scoped. R23 should run final codex review.
