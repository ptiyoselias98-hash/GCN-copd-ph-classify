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

---

## v4 (FINAL) — applied 3 rounds of codex GPT-5.5 hostile review

### Round-1 fixes: protocol leakage, scaling, default-mpap defaults, multimodal PH anchors, out-of-fold validation
### Round-2 fixes: R24.0 prerequisite, AND-gate not OR for protocol, ΔAIC≥10, kill R24.C
### Round-3 REVISE: AND-gate explicit, stratified permutation null (R24.X), R24.B as "informative-when-found" not core claim, side-by-side PCA-32 baseline, exploratory framing visible in figures

### Final sequence (locked, pre-registered)

| sub-round | task | cohort | gates | multiplicity | hardware |
|---|---|---|---|---|---|
| **R24.0** | locked cohort/analysis table | all 290 | structural | n/a | local CPU (1 worker, fast) |
| **R24.A** | PHATE pseudotime | within-contrast n=190 PRIMARY + full n=290 stress test | mPAP \|ρ\|≥0.35 (Holm m=3, boot CI excludes 0) **AND** protocol AUC<0.65 **AND** \|ρ_protocol\|<0.20 | family m=3 + heatmap m=78 | local CPU (joblib n_jobs=14) |
| **R24.B** | piecewise mPAP changepoints | n=102 measured mPAP only | ΔAIC≥10 vs linear AND boot freq≥70% AND CI width≤8 mmHg | Holm m=78 | local CPU joblib |
| **R24.F** | mPAP-association feature panel | n=102 | bootstrap optimism CI; cross-fit pseudotime fold-local | Holm m=78 | local CPU joblib |
| **R24.D** | structure ordering along pseudotime | within-contrast | bootstrap-1000 onset CI; sensitivity τ∈{0.25, 0.5, 0.75 SD} | m=6 pairwise (or CI-only) | local CPU joblib |
| **R24.E** | nomogram + DCA + calibration | within-contrast n=190 | 10× repeated nested 5-fold CV; calibration slope/intercept + Brier + fold AUC range; DCA at 10/25/50/86% prevalence; "EXPLORATORY" stamp visible in figure | feature coefficients across full candidate set | local CPU joblib (LR Lasso) |
| **R24.G** | mPAP-anchored severity embedding (RENAMED) | within-contrast n=190 PRIMARY | OOF severity-percentile vs mPAP ρ ≥ +0.50 (positive orientation, boot CI excludes 0); SSL must beat PCA-32 by ≥0.05 ρ | m=1 primary + m=1 SSL-vs-PCA + m=3 anchor pairs | remote GPU (SSL) + local CPU (PCA baseline + Ledoit-Wolf) |
| **R24.X** | stratified permutation null falsification | R24.A + R24.G | real statistic > 99th percentile of permuted-mPAP null (perm within protocol×fold strata, 1000 permutations) | n/a (null block) | local CPU joblib |
| ~~R24.C~~ | ~~cross-attention Sankey~~ | KILLED per round-2 codex (interpretability liability without scientific value) | - | - | - |
| **R24.Y** | overlay gallery REGENERATION on unified-301 (fixes R19.A blank-placeholder bug — PH source CTs were stub `_source.txt` redirects in R19 era, only nonPH new100 cases rendered) | unified-301 (16 representative + 16 worst-repaired by lung-mask CC fraction) | structural; visual QC only, not inferential | n/a | local CPU (~5 min) |

### Parallel execution DAG (user "尽量多GPU多worker并行" 2026-04-26)

Hardware inventory: local 16-core i5-14400 / 32GB / no GPU; remote IMSS 24-core / 2× RTX 3090 24GB idle.

**Wave 1 — local CPU concurrent (after R24.0 lock):**
- SHARED worker budget = 14 cores (joblib backend with `pre_dispatch='2*n_jobs'`); split across:
  - R24.A PHATE × 2 strata (within-contrast + full): n_jobs=4 each
  - R24.B per-feature changepoint × 78: n_jobs=4
  - R24.F bootstrap optimism × 78 features × 1000 iters: n_jobs=2
- All four read `cohort_locked_table.csv` only (read-only); each writes to unique output path under `outputs/r24/<sub>/`
- Fixed seeds: R24.A=42, R24.B=42, R24.F=42 — committed in script header

**Wave 1.5 — barrier**: commit + push wave-1 artifacts (pseudotime CSV, changepoint CSV, feature ranking CSV) BEFORE wave 2 launches. R24.X cannot run yet.

**Wave 2 — remote GPU concurrent (independent of wave 1; can start in parallel with wave 1):**
- GPU 0 worker α: R24.G SSL contrastive train (CUDA_VISIBLE_DEVICES=0, seed=42, feature-dropout 5%, 100 epoch, latent dim 32, n_neg=64, batch=64) → `outputs/r24/r24g_ssl_d32/`
- GPU 0 worker β: R24.G SSL ablation latent_dim=16 (CUDA_VISIBLE_DEVICES=0, seed=43) → `outputs/r24/r24g_ssl_d16/`
- GPU 1 worker α: R20 multi-branch retrain on within-contrast n=190 (CUDA_VISIBLE_DEVICES=1, seed=42, NOT for attention; feature extraction only) → `outputs/r24/r20_within_contrast_retrain/`
- GPU 1 worker β: R24.G SSL ablation latent_dim=64 (CUDA_VISIBLE_DEVICES=1, seed=44) → `outputs/r24/r24g_ssl_d64/`
- All 4 jobs share VRAM 24GB / 4 ≈ 6GB each — RTX 3090 budget per job
- Pinned env: each job logs to `/tmp/r24_<sub>_<gpu>_<seed>.log`; no shared writes

**Wave 2.5 — barrier**: commit + push wave-2 artifacts (SSL latent CSVs + retrained encoder); GPU jobs done.

**Wave 3 — sequential downstream (after wave 1.5 + 2.5 commits):**
- R24.D structure-onset bootstrap (depends on R24.A pseudotime committed + R24.G latent committed): joblib n_jobs=14
- R24.E 10× nested 5-fold CV: joblib n_jobs=14
- **R24.X stratified permutation null** (1000 perms within protocol×fold strata): consumes COMMITTED R24.A pseudotime + R24.G OOF severity CSVs; recomputes statistic on permuted mPAP only; joblib n_jobs=14 — **MUST be sequential, never parallel with R24.A/R24.G** (per round-4 codex barrier)
- All matplotlib renders (Agg backend, parallelizable)
- README inline-embed assembly

**Wave 4 — codex final review + manifest:**
- R24.Z aggregator commits unified `outputs/r24/RUN_MANIFEST.json` with:
  - **Inputs**: `cohort_locked_table.csv` SHA256 + producing commit SHA
  - **Inter-wave artifacts** (every CSV/PNG/weights consumed downstream): SHA256 + producing commit SHA — including:
    - Wave 1: R24.A pseudotime CSVs (within-contrast + full), R24.B changepoint CSV, R24.F feature-rank CSV
    - Wave 2: R24.G OOF severity CSVs (SSL d=32/16/64 + PCA-32 baseline), R20 within-contrast retrain encoder weights
    - Wave 3: R24.D structure-onset bootstrap CSV, R24.E nomogram + DCA CSV, R24.X permutation null distribution CSV
    - All PNGs in `outputs/figures/fig_r24*.png`
  - **Per-job**: seed, command line, output path, GPU id (if applicable), wall time, exit code
  - **Gate pass/fail**: per sub-round, with computed statistic + threshold + verdict
- Final codex GPT-5.5 review for 9.9 verdict

**Sync strategy:** every worker commits its own (script + CSV + PNG) immediately on completion; final R24.Z aggregator commits the unified state + REVIEW_STATE.json update + codex review. Cron remains 10-min cadence.

### Pre-registered gates (frozen before any code runs)
1. fold_id from R24.0 cohort_locked_table.csv
2. PHATE knn=5, decay=40, t=auto, RobustScaler
3. R24.A protocol gate AND-conjunction (not OR)
4. ΔAIC≥10, boot freq≥70%, CI width≤8 mmHg for changepoints
5. R24.G OOF ρ≥+0.50 with positive orientation
6. SSL must beat PCA-32 by ≥0.05 ρ to claim added value
7. Permutation null 99th-percentile threshold
8. Feature universe: 78-morph primary; 152D extended only if R24.0 commits the extended table

### Visualization (mandatory per cron prompt)
Every surviving sub-round produces ≥1 PNG in `outputs/figures/` named `fig_r24<sub>_*.png`, README inline-embed with Chinese explanation, 5-piece sync commit.

Each R24 sub-round produces an artifact in `outputs/r24/` and a paragraph
update in FINAL_FINDINGS.md or a new SCIENCE_ANSWERS_R24.md document.

Final R24 codex review: target 9.9, with explicit yes/no on each of
Q1–Q5 + R24.G validation.
