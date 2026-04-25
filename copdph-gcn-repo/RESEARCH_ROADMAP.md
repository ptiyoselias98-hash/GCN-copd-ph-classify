# Research Roadmap — COPD → COPD-PH structural-evolution program

_Last updated 2026-04-25 R14_

## The scientific question (final goal)

在 COPD 向 COPD-PH 转归的过程中，**肺血管影像表型如何演化？肺实质与气
道表型在其中起到怎样的辅助作用？** 现有研究在 COPD-PH **早期阶段**的
肺血管改变上证据有限，缺乏系统化的定量刻画与纵向演化分析。

To answer this, we need:

1. **Quantitative phenotyping** of the artery / vein / airway tri-graph
   on a protocol-balanced multi-stage cohort (early COPD → established
   COPD-PH).
2. **Vascular-graph evolution metrics** (node count, mean diameter,
   tortuosity, branching topology, Strahler order) tracked across stages.
3. **Auxiliary lung-parenchyma + airway phenotypes** (LAA-950 emphysema
   severity, apical-basal gradient, parenchyma HU distribution, airway
   wall thickness) as covariates that constrain the vascular endotype.
4. **Clustering / endotyping** that reveals the *typical structural
   patterns* on the COPD ↔ COPD-PH spectrum (R14.B is a first pass).
5. **Longitudinal validation** ideally — paired same-patient scans at
   different stages. (Currently absent — see "missing data" below.)

## What this codebase has produced (R1 → R14)

### Cohort + segmentation infrastructure
- ✅ **345-case authoritative cohort**: 160 PH (contrast) + 27 nonPH-contrast
  + 158 plain-scan nonPH (24 refill + 76 new + 58 existing) — R13.1
- ✅ **DCM-count audit**: 10 PH + 5 nonph-plain pruned for slice-count
  inconsistency — matches user's manual prune narrative
- ✅ **Segmentation-quality audit**: 34 REAL EMPTY-mask + 4 lung-component
  anomalies identified; effective n drops 80→68 within nonPH stratum (R13.2b)
- ⚠ **100 new plain-scan nonPH cases** queued but DCM→NIfTI pipeline +
  segmentation NOT YET RUN — needed for R15 to enlarge the within-nonPH
  stratum from 68 to ~166 (more statistical power)
- ⚠ **HiPaS-style unified resegmentation** for the 38 failure cases NOT
  YET RUN — the auxiliary path C from §24.6

### Vascular-graph quantification
- ✅ **TEASAR-based tri-structure cache** (`cache_v2_tri_flat`): per-case
  pkls with artery+vein+airway graphs, ~13-15 features per node + ~50
  graph-level aggregates (n_nodes, edge stats, x{0..N}_mean/std/p90)
- ✅ **Per-structure volumes, mean HU, p95 HU**: in `lung_features_v2.csv`
- ⚠ **TEASAR parameter sensitivity** (R6 must-fix) only proxied via
  skimage skeletonize; full kimimaro sweep on a 20-case subset still
  pending GPU availability
- ⚠ **Anatomical QC overlays** (R5 must-fix) not produced — would let
  reviewers verify each case's tri-graph is anatomically correct

### Disease-classifier evidence
- ✅ **Hybrid-GCN (3-layer GAT) on tri-graph**: arm_a baseline 5-fold AUC
  ~0.95 on full 282 → drops to ~0.84 on contrast-only 189 (protocol
  confound flagged by ARIS Round 1 as W1)
- ✅ **Within-nonPH protocol decoder**: LR AUC 0.85+ on graph aggregates,
  0.88+ on GCN embeddings, 0.66 on missingness alone. Confirms the
  protocol confound is encoded in the cache features (R5, R9, R12, R13).
- ✅ **Two deconfounder families tested on legacy 243 + n=80**:
  - GRL (R10-R12): exhausted; best λ=10 hierarchical CI [0.72, 0.94],
    disease AUC crashes 0.73 → 0.64 at high λ
  - CORAL (R13): single-seed λ=1 wins Pareto vs GRL — protocol LR 0.772
    with disease preserved at 0.93. Multi-seed expansion (R14) running.
- ⚠ **HSIC / IRM / propensity-overlap reweighting** not yet attempted

### Phenotype endotyping (R14.B — first pass)

Multi-structure UMAP + KMeans on 66 features (vascular graph + lung
parenchyma) over 226 cases yields a defensible 3-cluster structure
**within contrast-only** (n=184, no protocol confound):

| cluster | n | PH% | endotype description |
|---|---|---|---|
| C0 | 54 | 69 % | **Transition / mixed** — high vessel diameter (g_x1, g_e1), lower parenchyma HU p95 (more emphysematous) |
| C1 | 60 | 93 % | **PH arterial-rich** — high node/edge count + high artery volume; complex vasculature |
| C2 | 70 | 93 % | **PH dense-lung** — small lung volume, high parenchyma mean HU, low LAA-856; less emphysema |

**Interpretation (preliminary, n=184)**: PH manifests as TWO distinct
endotypes — one with extensive vascular remodelling (more nodes/edges,
arterial volume up) and one with restrictive small dense lungs (volume
loss, higher HU, lower emphysema). The transition cluster (C0) carries
COPD-PH-mixed cases distinguished by vessel-diameter-related features
plus emphysema-like parenchyma. **This is a candidate answer to the
final question's first half** (vascular evolution shows two-pathway
heterogeneity into PH), but requires:

- Multi-stage stratification (early vs established PH) — currently
  binary disease label only
- Replication on ingested 345-cohort
- Functional correlation (mPAP, 6MWT, FEV1) — `copd-ph患者113例0331.xlsx`
  has these but only for ~113 cases

## Distance-to-goal assessment (honest)

Quantifying 距离最终目标的差距 in 5 axes:

### Axis A — Cohort completeness  (60 % done)
- ✅ 345 case_id manifest with prune-explanation
- ⚠ 100 new plain-scan nonPH not ingested (DCM→NIfTI→seg pending)
- ⚠ No early-stage-PH stratification (no MPAP-binned subcohort)
- ❌ No longitudinal pairs (same patient at different timepoints)

### Axis B — Segmentation quality  (50 % done)
- ✅ Audit framework + 38-case exclusion list
- ✅ kimimaro-pinned cache (env locked)
- ⚠ 38 failure cases NOT re-segmented yet
- ⚠ No anatomical-overlay QC gallery
- ❌ HiPaS unified-pipeline rebuild pending (would let plain-scan and
  contrast share the same segmentation distribution)

### Axis C — Vascular-graph quantification  (70 % done)
- ✅ Tri-structure (artery + vein + airway) per-case graph in v2 cache
- ✅ 50-feature aggregate per case
- ✅ GCN embedding-level analyses
- ⚠ TEASAR parameter sensitivity only proxied
- ⚠ Strahler order, tortuosity, mean diameter not surfaced as
   first-class features in the aggregate CSV (need explicit extraction
   from the pkl graphs)
- ❌ No paper-quality vascular-evolution figure yet

### Axis D — Lung-parenchyma + airway auxiliary  (60 % done)
- ✅ `lung_features_v2.csv` has LAA-950, apical-basal gradient,
   pure-parenchyma HU stats for 282 cases
- ✅ R14.B multi-modal clustering uses lung features alongside graph
- ✅ **R14.D lung-only vs graph-only ablation** (`outputs/r14/ablation_lung_vs_graph.md`):
   within-contrast n=184 — lung_only AUC **0.844** > graph_only **0.782**
   > graph+lung **0.867** (complementary). Lung parenchyma is the
   primary disease signal carrier; vascular graph adds **+0.085 AUC**.
   ⚠ CIs overlap; reviewer requires paired AUC-difference CI for the
   reversal claim (R15).
- ⚠ Audit needed: lung HU features may retain acquisition/reconstruction
   confound even within-contrast (R15 must-fix)
- ⚠ Airway wall thickness, tapering not extracted (the airway appears in
   the graph as 13-D node features, not as morphometric global features)
- ❌ Lung-CT radiomics (PyRadiomics) on the parenchyma not extracted

### Axis E — Confounder control + reproducibility  (80 % done)
- ✅ Protocol confound flagged + within-nonPH primary endpoint
- ✅ Three deconfounder families tested: GRL (R10-R12), CORAL (R13-R14
   12 configs multi-seed), MMD (R14 2 configs single-seed)
- ✅ **CORAL @ λ=1 mean LR 0.71 (best 0.624) on n=68 corrected with
   disease 0.93 preserved** — first deconfounder to break GRL's 0.80
   floor. Best-deconfounder-so-far; not yet a confirmed Path-B win
   (3 seeds, no hierarchical CI, no paired test against GRL — R15).
- ⚠ Hierarchical seed × case bootstrap CI + paired CORAL-vs-GRL on
   identical corrected n=68 cases pending (R15 must-fix)
- ⚠ Lung HU residual-confound audit pending (R15)
- ⚠ HSIC / IRM / propensity-overlap reweighting still untried
- ❌ External cohort validation absent

### Overall: ~62% of the way to a publishable answer (up from 57% in R13).

The most binding constraints, in priority order:

1. **(Hardest) longitudinal data** — without paired same-patient scans
   we can only infer "evolution" cross-sectionally from endotype
   distributions. This is a fundamental cohort limitation, not solvable
   by code.
2. **(Tractable) 345-cohort ingestion + HiPaS re-seg** — this brings
   within-nonPH from n=68 to ~166 and resolves the protocol-balanced
   evidence stratum.
3. **(Tractable) explicit vascular-evolution morphometrics** — extract
   per-case mean_diameter, tortuosity, max_strahler, branching ratio
   from the v2 graph pkls (already implicitly there, need surfacing).
4. **(Tractable) lung-only vs graph-only ablation** to quantify the
   auxiliary role of lung parenchyma in disease classification.
5. **(Tractable) anatomical-overlay QC gallery** — evidence the
   tri-graph is anatomically correct; required by ARIS reviewer.

R15-R20 plan (high-leverage):
- R15: HiPaS re-seg + 100-case DCM→NIfTI launch (Path A)
- R16: explicit vascular morphometrics + lung-only-vs-graph-only
  ablation
- R17: anatomical overlay gallery + TEASAR sweep finalised
- R18: multi-stage PH stratification using mPAP from
  `copd-ph患者113例0331.xlsx`
- R19: 345-cohort retrain on multi-modal features; report endotype
  composition shift vs legacy 282
- R20: paper-quality figures + final ARIS review

Estimated total wall time: **2-4 weeks of unattended cron + 1 GPU-week**
on RTX 3090 ×2.
