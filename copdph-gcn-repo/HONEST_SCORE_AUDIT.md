# Honest score & gap-to-goal audit (2026-04-25, post-R17)

This document corrects the optimistic ARIS-loop scoring by tallying **all
unfinished must-fix items across rounds** and recomputing the gap-to-final-
goal honestly. The codex hostile reviewer scores each round +0.2 for
local progress, but compounded over 17 rounds the cumulative unclosed
debt has been ignored.

## ARIS-claimed score trajectory vs honest score

| round | reviewer score | unclosed cumulative debt | honest score |
|---|---|---|---|
| R10 | 6.2 | — | 6.2 |
| R11 | 5.0 | 1 (regressed) | 5.0 |
| R12 | 7.0 | 0 | 7.0 |
| R13 | 8.0 | 1 (HiPaS pending) | 7.5 |
| R14 | 8.4 | 4 (paired CIs, lung confound, k-stability, morphometrics queued) | 7.6 |
| R15 | 8.8 | 5 (above + 100-case ingestion partial) | 7.8 |
| R16 | 9.0 | 7 (above + embedding probe, multi-seed CORAL on enlarged, paired DeLong, overlay gallery) | 8.0 |
| **R17** | **9.2** | **12 (R16 unclosed + R17 same-case paired, extraction artifacts, covariate adjustment, scanner sensitivity, mPAP stage, narrative tightening)** | **~8.0** |

**Adjustment rule going forward**: each unclosed must-fix carries a −0.1
score penalty. R17 reviewer-claimed 9.2 − 12 × 0.1 = **8.0** honest score.

## Cumulative unclosed must-fix items (12 as of R17)

### Method validation debt

1. ❌ **Embedding-level enlarged-stratum protocol probe** (R16 must-fix #1) — R15.G/R16.D are scalar lung-features only; GCN inputs/embeddings on the n=151 enlarged stratum NOT tested
2. ❌ **Multi-seed CORAL/MMD on enlarged cohort** with hierarchical seed × case CIs (R16 #2)
3. ❌ **Repaired-vs-unrepaired paired DeLong** for disease + protocol AUC on same cases (R16 #3)
4. ❌ **Same-case paired artery/vein/airway/all_three AUCs** with identical n + paired DeLong (R17 #1) — current different-N comparison invalidates the airway-LR=0.797 paradox
5. ❌ **R17 extraction audit** — `n_terminals=0` for ALL cases is a bug (degree counting on doubled edges); Lap eig0 numerical zero degenerate; edge_attr `[::2]` de-doubling unverified (R17 #2)
6. ❌ **Lung-mask blinded overlay gallery** — repaired masks have no Dice/coverage/anatomical QC (R16 #4)
7. ❌ **Covariate-adjusted models** for artery_tort_p10 + artery/vein length percentiles + paren_std_HU (scanner/age/sex if metadata) (R17 #3)
8. ❌ **Scanner-stratified sensitivity** + leave-one-site-out + robust regression for paren_std_HU and topology features (R17 #4)

### Cohort completeness debt

9. ❌ **HiPaS re-segmentation of 38 legacy seg-failure cases** — R13.2b identified, never repaired (R16 #5)
10. ❌ **v2 cache rebuild on new100 cases** — required to extend per-structure morphometrics + GCN inputs from 282 to 360 (cohort-side dependency)

### Scientific-narrative debt

11. ❌ **mPAP 5-stage stratified evolution analysis** (R18) — user provided plain-scan = mPAP 0-10 default 2026-04-25; without this NO real "evolution" answer (Q3 + Q4 of final goal essentially 0% answered)
12. ❌ **Mechanistic claim tightening** — README/REPORT_v2 say "vessel pruning"; actual evidence supports "segment-length percentile downshift + low-tort downshift consistent with remodeling/pruning" (artery total_len_mm not reduced; n_nodes/n_edges higher) (R17 #5)

## Final-goal completion audit (downward-corrected)

| user question | prior claim | honest |
|---|---|---|
| **Q1**: 肺血管影像表型如何演化? | 62% | **30-40%** (cross-sectional only, no stage-binning, no trajectory) |
| **Q2**: 肺实质 + 气道辅助作用? | inferred | **35%** (paren_std_HU is good, no scanner-correction, no vessel-parenchyma coupling, no cross-modal attention) |
| **Q3**: 早期 PH (early-stage) 改变? | unaddressed | **<10%** (no mPAP stage 2 vs 4 differentiation; R18 never run) |
| **Q4**: 系统化定量 + 纵向演化分析? | inferred | quantification ~40% (univariate only, no multivariate endotype model); longitudinal ~5% (no paired same-patient scans, R18 stage-binned cross-section is the proxy) |

**Weighted average (Q1=40%, Q2/3/4=20% each)**: **~30%** to publishable answer (vs. RESEARCH_ROADMAP.md's claimed 62%).

## What was NEVER STARTED (out of original task list)

These are explicitly queued tasks that have ZERO progress:

- **R17.5 TDA persistence diagrams** (Wasserstein vessel distribution distance, gudhi/ripser) — 0%
- **R18 mPAP 5-stage stratified endotype** with Spearman + Jonckheere-Terpstra + per-stage UMAP — 0%
- **R18 enhanced**: stage-wise GP regression trajectory + change-point detection — 0%
- **R19 lung parenchyma diffusion model anomaly detection** (DDPM on 48³ patches) — 0%
- **R20 multi-branch joint model** (artery + vein + airway + parenchyma cross-attention + multi-task disease/endotype/protocol-adversarial) — 0%
- **R20 attention rollout / GNNExplainer** for "PH 模型在 vessel 拓扑哪里看" interpretability heatmap — 0%
- **External cohort or temporal validation** — 0% (no available data)

## Corrective actions taken in R18

1. ✅ Score corrected to 8.0/10 in README (honest, not codex-inflated 9.2)
2. ✅ This audit document committed
3. 🔄 R18 immediately starts with: same-case paired AUCs, R17 extraction audit (fix n_terminals bug), mPAP 5-stage manifest, stage-wise trajectory analysis
4. 🔄 R17.5 TDA + R19 diffusion will launch on remote 24-core + GPU 0 in next 1-2 fires (parallel)
5. 🔄 codex review prompt template updated: each unclosed must-fix is −0.1 penalty; reviewer not allowed to give >score increase if cumulative debt grew
