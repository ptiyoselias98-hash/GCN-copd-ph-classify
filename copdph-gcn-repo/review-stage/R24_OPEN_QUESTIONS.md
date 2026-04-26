# R24+ Open Science Questions — target raised to 9.9

_2026-04-26 user reopened loop with the explicit research framing:_

> 在 COPD 向 COPD-PH 转归过程中，肺血管影像表型如何演化？肺实质与气道
> 表型在其中起到怎样的辅助作用？目前 COPD-PH 早期肺血管改变研究有限，
> 缺乏系统化的定量刻画和纵向演化分析。本研究将以肺血管演化规律为核心，
> 结合肺实质破坏和气道重塑的辅助信息，探究不同结构影像表型在疾病进展
> 中的演化轨迹、相互作用及协同模式，为早期识别高危患者提供定量影像学
> 依据。

R23 reached codex 9.6 on **manuscript scope**. The user clarified that
the 9.6 did NOT yet prove the four core science questions. Target raised
to **9.9**. Manuscript scope decisions from R23 stand; R24+ must add new
evidence for these questions:

## Q1. 血管表型演化轨迹 (current ~65% answered)
- **Have**: cross-sectional 5-stage mPAP severity ordering (legacy ρ=-0.767,
  unified ρ=-0.211); 4 flagship features direction-preserved across pipelines
- **Need (R24)**: pseudotime trajectory model that orders patients along a
  continuous severity axis using all 78 morphometric features jointly; export
  per-feature evolution profile (which feature changes EARLIEST, which LATEST).
  Use diffusion-map or PHATE pseudotime; defensible substitute for missing
  repeated-scan longitudinal data.

## Q2. 肺实质 + 气道辅助作用 (current ~55%)
- **Have**: paren_std_HU d=+1.10 (year-residualized robust); airway 0/44 Holm-sig;
  R18.A (lung+graph)−graph Δ=+0.085 p=0.0008
- **Need (R24)**: multi-structure cross-attention weight visualization from R20
  joint encoder, plus structure-leadership analysis ("does parenchyma change
  precede vessel change in pseudotime, or follow?"); answers "辅助 vs 主导".

## Q3. 早期 PH (mPAP 25-35) 血管改变 (current ~40%)
- **Have**: Stage 2 (n=8) + Stage 3 (n=44) partially separable on flagship features
- **Need (R24)**: piecewise-linear or changepoint analysis on continuous mPAP
  axis; identify the mPAP threshold where each flagship feature first becomes
  detectable. Likely 22-28 mmHg per literature; required for "early identification
  of high-risk patients".

## Q4. 系统化定量 + 纵向演化分析 (quantification ~75%, longitudinal ~5%)
- **Have**: 132 morphometric + 18 TDA + 50 lung features; R18.E covariate-adjusted
- **Need (R24)**:
  - (a) systematic feature panel ranked by pseudotime-monotonicity AND
        Holm-corrected effect size — "evolution feature panel"
  - (b) pseudotime trajectory as the longitudinal-substitute deliverable
  - (c) explicit acknowledgment in the paper that true longitudinal data is
        a cohort-level limitation; pseudotime + changepoint together provide
        a defensible substitute analysis

## Q5. (implied by user) 高危患者识别定量影像学依据
- **Need (R24)**: validated risk score = β₁·artery_len_p25 + β₂·paren_std_HU
  + β₃·vein_persH1_total + ... fit on within-contrast n=190; report
  Hosmer-Lemeshow calibration, decision-curve analysis, and threshold for
  "high-risk" classification at clinically-meaningful sensitivity/specificity.

## R24+ workplan (closes path to 9.9)

| Sub-round | Action | Closes question | Effort |
|---|---|---|---|
| R24.A | Pseudotime trajectory (diffusion-map or PHATE) on unified-301 morph | Q1 + Q4(b) | ~30 min local |
| R24.B | Per-feature mPAP-axis changepoint detection | Q3 | ~20 min local |
| R24.C | R20 cross-attention weight extraction + Sankey visualization | Q2 (interaction & 协同) | ~40 min |
| R24.D | Pseudotime structure-leadership analysis (which structure changes earliest along trajectory) | Q2 (辅助 vs 主导) | ~30 min |
| R24.E | Risk-score nomogram + calibration + DCA | Q5 | ~60 min |
| R24.F | "Evolution feature panel" Holm-corrected ranked by pseudotime monotonicity | Q4(a) | ~20 min |
| **R24.G** | **Disease Progression Space**: self/weakly-supervised representation learning (contrastive on 152D feature space) → low-dim latent → PH-anchor centroid → per-patient progression-percentile via Mahalanobis distance to PH manifold. Validate vs mPAP. Identify borderline-PH candidates (50-75% percentile) | Q1 (continuous trajectory) + Q3 (early identification) + Q5 (risk score) | **~60 min** |

R24.G is the explicit user-requested experiment (2026-04-26): "依托大样本横断面COPD CT...自监督/弱监督表示学习...连续疾病进展空间...COPD-PH患者作为终点锚点...通过相似度及进展轴相对距离推断阶段". Method: contrastive SSL on unified-301 152D features → 32D latent → PH centroid → per-patient progression-percentile.

Each R24 sub-round produces an artifact in `outputs/r24/` and a paragraph
update in FINAL_FINDINGS.md or a new SCIENCE_ANSWERS_R24.md document.

Final R24 codex review: target 9.9, with explicit yes/no on each of
Q1–Q5 + R24.G validation.
