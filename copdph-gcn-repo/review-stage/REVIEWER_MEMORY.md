# Reviewer Memory (persistent across rounds)

> This file is injected into every subsequent ARIS round's context so the
> reviewer remembers prior concerns and can verify whether they were actually
> addressed (vs. cosmetically mentioned).

## Round 1 key concerns (2026-04-23)

**Central threat: acquisition/source confounding.** All 170 PH cases are
contrast-enhanced CT; 85 of the nonPH cases are plain-scan CT. The reported
arm_a AUC ~0.95 could be protocol classification rather than disease signal.
Any claim of disease performance must survive protocol-balanced controls.

**Airway is out-of-scope for main results.** Only 41/47 rebuilt airways are
valid (2 invalid, 4 errored). No airway-specific QC. arm_b_v2 AUCs unstable.
Airway results must be appendix-only until airway QC is implemented and
all airway segmentations pass blinded visual inspection.

**Statistics are not confirmatory-grade.** Many arms × configs × repeats
without nested CV, multiple-comparison correction, paired DeLong / bootstrap
CIs on the v1→v2 AUC jump. Internal 5-fold CV is not sufficient for top-venue
claims.

**Graph construction not anatomically validated.** TEASAR parameters,
radius estimation, Strahler ordering treated as correct without overlay
audits or morphometric baselines. Known defects: voxel-space centroid coords,
ct_density=0, Strahler-approx without cycle handling.

**QC/exclusions are label-biased.** 27 placeholder nonPH excluded (changes
class balance). Exclusion rules (vox_per_key, mask_vox + num_nodes) are
ad-hoc thresholds.

## Round 18 key concerns (2026-04-25 21:30) — CARRY INTO ROUND 19

**Score: codex 9.3 / honest 8.8** (debt-penalty −0.5 for 5 unclosed must-fix).
**+0.8 honest from R17=8.0** — biggest single-round honest leap.
**Path to 9.5: close 5 remaining debts.**

R18 closed 5 must-fix items (most progress in any round):

1. ✅ **R18.A same-case paired per-structure AUCs**: vein 0.801, airway 0.790,
   artery 0.741 (n=190 same-case). ALL paired Δ NS (p>0.27).
   **Airway 0.797 R17 'highest' was sub-cohort artifact — RETRACTED.**

2. ✅ **R18.C mPAP resolution**: `mpap_lookup_gold.json` resolves case_id →
   mPAP for 106/113 PH cases (no MD5 inversion needed).

3. ✅ **R18.B 5-stage mPAP evolution**: First defensible CROSS-SECTIONAL
   severity-ordered evidence:
   - artery_len_p25 ρ=−0.767 p=9e-30 Jonckheere z=−10.07
   - paren_std_HU ρ=+0.629 p=1.5e-17
   - artery_tort_p10 ρ=−0.619 p=6.7e-17
   - vein_len_p25 ρ=−0.613 p=2.7e-16
   **CRITICAL CORRECTION**: apical_basal_LAA950_gradient earlier R16/R17
   sig was RANDOM-SPLIT ARTIFACT — now NS p=0.16 with real mPAP.
   paren_LAA_950 NOW sig ρ=+0.218 (was NS in proxy).

4. ✅ **R18.E covariate-adjusted endotype**: ALL 8 features survive
   year-residualization (Δd<0.05). artery_tort_p10 raw d=−1.42 →
   year-adj −1.40. Year-rho all in [−0.22, +0.14]. Confounds NOT
   contaminating findings.

5. ✅ **R17.5 TDA persistence** + ✅ **R18.F unified endotype**:
   3-modality Holm-Bonferroni panel (n=197 contrast, 201 features):
   - morph 17/143 sig, top artery_tort_p10 d=−1.42
   - **TDA 3/18 sig, NEW vein_persH1_total d=−1.21** (PH veins lose
     H1-loop topology — independent topology-loss finding)
   - lung 6/40 sig, top paren_HU_p95 d=+1.12
   3-modality LARGE-effect coupling: **vessel-remodeling ×
   parenchyma-densification × topology-loss**

6. ✅ **R20 multi-branch joint** trained end-to-end: 4-branch
   (artery/vein/airway/paren) + cross-attention + multi-task heads.
   Within-contrast n=100 mean AUC 0.859±0.160 (range 0.61-1.00 high
   variance — exploratory, not stable).

7. **R19 DDPM IN FLIGHT GPU 0**: epoch 4/30 loss 0.075 (from 0.81).
   ETA ~6h. NO ANOMALY EVALUATION YET — must complete before R19 counts.

**Reviewer regressions**:
- mPAP "evolution" must be reframed to "cross-sectional severity
  ordering" — no longitudinal repeated measures
- R20 full-cohort AUC 0.943 is mostly protocol decoding
- R20 within-contrast 0.859 has fold variance 0.61-1.00
- mPAP Stage 0/1 nonPH bins assigned by protocol/default — not measured
- Year-only residualization incomplete (need age/sex/scanner/site)
- R17 extraction artifacts still not audited (n_terminals=0 bug)

**5 remaining honest-debt items for R19+ (path to 9.5)**:
1. R19 DDPM full evaluation (inference + anomaly heatmaps)
2. Embedding-level enlarged-stratum probe (v2 cache rebuild on new100)
3. Lung-mask blinded overlay gallery
4. HiPaS re-seg of 38 legacy failures
5. Multi-seed CORAL/MMD on enlarged stratum

## Round 17 key concerns (2026-04-25 19:35) — CARRY INTO ROUND 18

**Score 9.2/10 revise** (up from R16=9.0). **0.3 from target 9.5.**

1. **Strongest topology finding to date** (R17.A): native per-structure
   morphometrics from `cache_tri_v2` reveal artery/vein edge-distribution
   phenotype within contrast (n=197):
   - **artery_tort_p10 d=−1.42 p_holm=2.6e-7** (LARGE, project's strongest
     single feature)
   - artery_len_p25/p50/p10 d=−1.22 to −1.25
   - vein_len_p25 d=−1.19, vein_tort_p10 d=−1.15
   - All top-18 Holm-sig are artery+vein edge-LENGTH+TORTUOSITY
     **DISTRIBUTIONS** — the user's hypothesized vessel-pruning signature
     captured precisely by percentile features that GCN mean-pool destroys.

2. **Airway 0/44 Holm-sig** is a WEAK negative, not proof of vessel-only PH
   phenotype. Airway-only LR AUC=0.797 paradoxically highest (different-N
   artifact). Reviewer requires same-case paired comparison.

3. **Mechanistic claim must be tightened**: artery total_len_mm IS NOT
   lower, n_nodes/n_edges higher — so "vessel pruning/dropout" overclaims.
   Supported claim: **"segment-length percentile downshift + low-percentile
   tortuosity downshift consistent with vascular remodeling/pruning"** —
   NOT proven dropout.

4. **R17 extraction artifacts**: n_terminals=0 for all structures (bug —
   probably edge-doubling double-counting degrees), near-zero nonPH SD on
   some features, Laplacian eig0 numerical zero (degenerate, drop).

5. **R16 must-fix items NOT closed in R17**: only 1/7 closed (per-structure
   morphometrics). Embedding-level enlarged probe, multi-seed CORAL,
   paired DeLong, overlay QC, HiPaS repair, covariate adjustment all
   PENDING.

6. **Path to 9.5**: same-case paired AUCs + covariate/scanner-adjusted
   confirmation of artery_tort_p10/length-percentiles + extraction audit
   + embedding-level enlarged deconfounding + anatomical QC overlay
   gallery.

Round 18 minimum:
- Same-case paired artery/vein/airway/all_three AUCs (identical n)
- R17 extraction audit (n_terminals bug, edge-doubling [::2] check, Lap eig0)
- Scanner/age/sex covariate-adjusted models for artery_tort_p10 + artery/vein
  length percentiles + paren_std_HU
- Sensitivity: scanner-stratified, leave-one-out, robust regression
- Rephrase mechanistic narrative (no "proven dropout")
- Embedding-level enlarged probe (R16 must-fix #1)

## Round 16 key concerns (2026-04-25 18:30) — CARRY INTO ROUND 17

**Score 9.0/10 revise** (up from R15=8.8). Half a point from target 9.5. R16
delivered the strongest methodological progression of the project so far:

1. **Simple_AV_seg plain-scan oversegmentation independently detected** (R16.A):
   79/100 new lung masks >8.5L (median 10.8L), confirming domain-transfer flag.

2. **Holm-Bonferroni endotype** (R16.B): 9/14 features survive at α=0.05.
   **paren_std_HU is the LARGEST effect (Cohen's d=+1.10 [+0.80, +1.48],
   p_holm=1.7e-7)** — PH lungs +15.7 HU more heterogeneous parenchyma.
   This is the highest-quality single endotype finding to date.

3. **Lung-mask repair** (R16.C): HU<-300 + top-2-CC filter → median vol
   10839 mL → 7678 mL (26% drop, 100/100 cases).

4. **Repaired enlarged probe** (R16.D) — KEY INVERSE FINDING:
   - Within-nonPH protocol LR **rises** 0.908 → **0.958** [0.924, 0.983]
     after repair. R15.G was CONSERVATIVE, not inflated. Protocol confound
     is even more severe than reported on the enlarged stratum.
   - Within-contrast disease drops 0.847 → 0.816 [0.706, 0.909]; ~3 AUC
     points lived in over-mask soft-tissue. **0.816 is the cleaner estimate.**
   - Endotype replicates IDENTICALLY on repaired data — confirms robust.

5. **R17 schema discovered**: `tri_structure/cache_tri_v2/<case_id>_tri.pkl`
   has native `{artery: Data, vein: Data, airway: Data}` — direct per-structure
   access without reverse-engineering merged cache_v2_tri_flat.

6. **Outstanding R16 reviewer flags for R17**:
   - Embedding-level enlarged probe NOT yet done (scalar 0.958 insufficient)
   - Multi-seed CORAL/MMD on enlarged n=151 NOT redone
   - Paired DeLong repaired vs unrepaired disease/protocol AUC missing
   - Blinded overlay gallery / Dice / coverage QC missing for repaired masks
   - Per-structure morphometrics from cache_tri_v2 NOT yet extracted
   - HiPaS re-seg of 38 legacy failures pending
   - paren_std_HU not yet covariate-adjusted (age/sex/scanner)
   - vein_vol_mL & vessel_airway_over_lung have extreme variance (Simple_AV_seg
     artery/vein-confusion artifacts; need per-structure QC)

7. **Path to 9.5/10** (codex):
   - Embedding-level protocol AUC reduced AFTER deconfounding while disease
     AUC stays ≥0.80
   - paren_std_HU confirmatory-grade: covariate-adjusted + scanner-robust
   - Final locked cohort manifest + reproducibility from case IDs
   - Anatomical mask validation (overlay gallery, Dice if labels available)

Round 17 minimum:
- v2 cache rebuild on new100 (after lung-mask repair) → enlarged GCN-input cohort
- Embedding-level within-nonPH protocol probe at n≈151
- Multi-seed CORAL on enlarged stratum w/ hierarchical CIs
- Per-structure morphometrics from cache_tri_v2 (artery/vein/airway separate)
- Paired DeLong repaired-vs-unrepaired
- Lung-mask overlay gallery (5-10 representative + 5-10 worst-case repaired)
- Covariate-adjusted paren_std_HU model

## Round 15 key concerns (2026-04-25 16:30) — CARRY INTO ROUND 16

**Score 8.8/10 revise** (up from R14=8.4). Major progress on enlarged
cohort + R14 must-fix items, but new questions raised:

1. **R14 must-fix paired-CI ✅**: lung_only > graph_only is NOT significant
   (paired Δ=+0.062 [-0.031, +0.160] p=0.19). The defensible read is
   COMBINED (lung+graph) > graph alone (Δ=+0.085 p=0.0008) — lung adds
   complementary signal, but neither modality dominates standalone.

2. **CORAL is seed-unstable** ✅: paired vs corrected-GRL only sig at
   seed=2042 (p=0.007); ties at 1042; loses at 42. Reframe to "exploratory
   non-GRL deconfounder", NOT confirmed Path-B win.

3. **Clustering k=2 winner** ✅: consensus ARI 0.943 at k=2, R14 k=3
   unstable (0.827). 3+ subdivisions are seed-dependent.

4. **Lung scanner-era confound NOT detected** ✅: year-effect on
   paren_mean_HU p=0.25, HU-cluster vs year p=0.84. Lung-only disease
   AUC is genuine.

5. **100-case ingestion ✅**: DCM→NIfTI→Simple_AV_seg→lung-features
   pipeline complete. Cohort 282→360, plain-scan nonPH 85→163, total
   nonPH 112→190. 22 of 100 were refills of legacy placeholders.

6. **KEY new finding (concerning)**: enlarged within-nonPH protocol
   probe at n=151 yields LR AUC **0.908 [0.819, 0.968]** vs R12 baseline
   **0.853 [0.722, 0.942]** at n=80. **Protocol confound is MORE
   pronounced on the larger stratum**, not less — current CORAL/GRL
   evidence on legacy 80-case stratum does NOT generalize to the 151
   stratum. R16 must redo deconfounding on enlarged embeddings.

7. **Endotype replication ✅** (within-contrast n=186):
   - paren_mean_HU PH −807.7 vs nonPH −844.7, Δ=+37 (p=0.001)
   - apical_basal_LAA950_gradient PH −0.027 vs nonPH +0.041 (p=0.005;
     SIGN FLIP — PH has basal>apical emphysema)
   - lung_vol_mL PH 3380 vs nonPH 4353, Δ=−972 (p=0.005)
   - artery_vol_mL PH 261 vs nonPH 203, Δ=+58 (p=0.0009)
   - paren_LAA_950_frac NS (p=0.59) — total emphysema unchanged; the
     difference is in DISTRIBUTION (apical-basal gradient) not amount.
   First quantitative answer fragment to "vascular phenotypes evolve
   in COPD→COPD-PH": denser+smaller lungs, more artery volume, basal-
   emphysema redistribution.

8. **Outstanding R15 reviewer flags for R16**:
   - Simple_AV_seg trained on CTPA, applied to plain-scan — domain
     transfer QC missing (Dice/coverage/overlay gallery)
   - Vascular morphometrics CSV lacks longest_path_hops (BFS condition
     never satisfied); per-structure artery/vein split incomplete
   - 38 legacy HiPaS re-segmentation pending
   - Endotype p-values not multiplicity-corrected; covariate-adjusted
     analyses (age/sex/scanner) missing

Round 16 minimum:
- Embedding-level enlarged-stratum protocol probe (n=151)
- Multi-seed CORAL/MMD on enlarged cohort with hierarchical CIs
- Independent Simple_AV_seg QC on plain-scan masks (lung volume/range,
  vessel volume sanity, blinded spot QC)
- HiPaS re-seg outcomes for 38 legacy failures
- Multiplicity-corrected endotype effect sizes + covariate adjustment

## Round 14 key concerns (2026-04-25 12:30) — CARRY INTO ROUND 15

**Score 8.4/10 revise** (up from R13=8.0). R14 added multi-seed CORAL +
lung-vs-graph ablation + multi-structure clustering + research roadmap:

1. **Multi-seed CORAL λ=1 reduces protocol LR to 0.71 (mean of 3 seeds,
   range 0.62-0.79) on n=68 corrected, with disease AUC 0.93 preserved**.
   First deconfounder to break corrected-GRL R11 0.80 floor with intact
   disease signal. NOT confirmed yet — needs hierarchical seed×case CI
   and paired comparison vs GRL on identical n=68 cases (R15 must-fix).

2. **Lung-only AUC 0.844 > graph-only AUC 0.782 (within-contrast n=184)** —
   lung parenchyma carries MORE disease signal than vascular graph
   topology in this cohort. graph+lung 0.867 = complementary (+0.085
   AUC). Reviewer flag: contrast-nonPH n=26 is small; HU features may
   retain residual scanner/reconstruction confound; CIs overlap so
   reversal needs paired AUC-diff CI.

3. **3 within-contrast PH endotypes** from UMAP+KMeans on 66-D feature
   vector: C0 transition (69%PH; vessel-diameter+emphysema), C1
   PH-arterial-rich (93%PH), C2 PH-dense-lung (93%PH). Plausible
   hypothesis-generator but baseline contrast PH prevalence is 85.9%,
   so C1/C2 only modestly enriched. No stability/silhouette/ARI yet.

4. **MMD λ=5 LR=0.644 but disease drops to 0.85** — too aggressive;
   λ=1 MMD has LR 0.86, no advantage over CORAL.

5. **R14_vascular_morphometrics.py scaffolded** but not run (scp blocked
   in transcript). Move to R15.

6. **RESEARCH_ROADMAP.md** scopes 5 axes; overall ~62% to publishable
   answer. Most binding constraint: longitudinal data absence (cohort
   limitation).

Round 15 minimum:
- Hierarchical seed × case bootstrap CI for CORAL λ=1 vs λ=0 vs GRL,
  paired on n=68
- Paired AUC-difference CI for lung_only > graph_only
- Lung-feature residual-confound audit within contrast-only
- Clustering stability sweep (k=2..6, multiple seeds, silhouette,
  consensus ARI)
- Launch DCM→NIfTI pipeline for 100 new plain-scan nonPH cases
- HiPaS re-segmentation status report for 38 failure cases
- Run vascular morphometrics extraction on remote pkls

## Round 13 key concerns (2026-04-25 12:10) — CARRY INTO ROUND 14

**Score 8.0/10 revise** (up from R12=7.0). R13 closed several R12 must_fix
items but CORAL evidence is single-seed and overclaim language was caught:

1. **345-cohort reconciliation closed**: case_id-level diff against
   legacy 282 yields exactly 15 only-legacy = 10 PH + 5 nonph_plain,
   matching the user's manual DCM-count-prune narrative (170→160 PH;
   85→58 plain). PH count discrepancy explained.

2. **38 seg-failure cases identified** (34 REAL EMPTY-mask + 4 lung
   anomalies) in the legacy nii-unified-282 masks. 12 of these were in
   the 80-case within-nonPH stratum used by R1-R12. Effective n drops
   to 68 after interim exclusion. **Final cohort policy must re-segment
   via HiPaS, not exclude permanently** — exclusion is protocol/skew-
   structured (mostly nonph_plain).

3. **CORAL λ=1 single-seed pilot wins Pareto**: protocol LR 0.772
   (vs GRL 0.790; Δ=0.018 with wide overlapping CIs), AND disease AUC
   preserved at ~0.93 (vs GRL crash to 0.64 at λ=10). MUCH better Pareto
   but NOT a confirmed deconfounder win. Multi-seed expansion required.

4. **Don't say "Path B exhausted"** from a single-seed pilot. Reviewer
   flagged §24.6 overclaim; softened. Path B continues with multi-seed
   CORAL + MMD evidence; HSIC/IRM still unattempted.

5. **MMD scaffolded but not evidenced**: `run_sprint6_v2_coral.py --use_mmd`
   exists but no runs. Treat as R14 deliverable.

6. **CORAL alignment-loss printed as 0.00000** every batch — likely
   numeric underflow from /4d² with d=64. Need higher-precision logging
   to confirm CORAL gradient actually flows.

Round 14 minimum:
- Multi-seed CORAL at seeds {1042, 2042} × 4 λ on n=68 corrected
- Hierarchical seed × case bootstrap CIs + paired GRL comparison
- MMD pilot at λ ∈ {1, 5}, seed=42
- Disease AUC + CI in coral_probe.json (not only narrative)
- DCM→NIfTI pipeline launch for 100 new plain-scan nonPH cases
- Re-segmentation outcomes for 38 failure cases via HiPaS

## Round 12 key concerns (2026-04-25 11:30) — CARRY INTO ROUND 13

**Score 7/10 revise** (up from R11=5.0). R12 added auditable artifacts and a
crucial scientific finding, but the framing must be tightened:

1. **Missingness alone leaks protocol within-nonPH at AUC 0.664**
   [0.599, 0.724]. 31 of 32 cache-missing nonPH cases are plain-scan, 1
   contrast. Any "principled missingness" rescue inherits this leak as a
   floor — even perfect imputation cannot recover invariance with the
   missingness pattern intact.

2. **Cross-seed pooled-prob + hierarchical bootstrap CIs** confirm corrected
   GRL is exhausted on legacy 243-cache, n=80 within-nonPH stratum: best
   λ=10 hierarchical CI [0.719, 0.935], lower bound ≫ 0.60 target. MLP
   probe stays ~0.88 → non-linear protocol leakage exceeds linear-probe
   diagnostics.

3. **R12 must NOT be read as general impossibility.** It is impossibility
   only for: corrected-GRL + nonPH-only adversary + n=80 + legacy 243 cache.
   Open paths reviewer requires before claiming general impossibility:
   - non-GRL deconfounder (CORAL/MMD/HSIC kernel independence; conditional
     adversary with environment IRM; propensity-overlap reweighting)
   - 345-cohort ingestion (158 plain-scan nonPH = 24 refill + 76 new + 58
     existing) → properly enlarged within-nonPH stratum
   - unified HiPaS-style segmentation rebuild (removes protocol-induced
     segmentation-quality differential at source)

4. **Cohort manifest unresolved**: legacy 282-cohort has PH=170 but user's
   updated authoritative inventory has PH=160. R13 must do a case_id-level
   diff and freeze a single cohort manifest.

5. **CI inferential rule (going forward)**: use hierarchical (seeds × cases)
   bootstrap for any seed-aggregated AUC; the seed-replicate-pooled n=240
   probe is descriptive only.

6. **Held-out adversary validation curve required**: batch-mean adversary
   AUC saw-tooths in [0.05, 0.95] in run.log, useful but noisy and not
   equivalent to a held-out per-epoch validation AUC. Any new adversarial
   variant must report held-out adversary loss/AUC with fixed early stop.

Round 13 minimum:
- 345-cohort case_id reconciliation + PH-count audit
- 1+ non-GRL deconfounder on legacy 243 cohort
- Reframe REPORT_v2 §23.4 (drop blanket "impossibility")
- Disease AUC reported alongside any protocol-AUC reduction

## Round 5 key concerns (2026-04-24 10:30) — CARRY INTO ROUND 6

**Score 6/10 reject.** Two new important findings:

1. **GCN-input within-nonPH protocol AUC = 0.853 [0.722, 0.942]** on the
   EXACT cache_v2_tri_flat graph aggregates (47 features, 80 nonPH cases).
   The R4.1 finding that "v2 per-structure volumes" had LR AUC 0.529
   does NOT generalize — the richer features the GCN sees DO leak protocol.
   Disease AUC on the same features is also strong (0.858 within-contrast).
   Domain-adversarial debiasing is the principled mitigation path.

2. **Paired DeLong is BLOCKED** by arm_b dataset construction. arm_b uses
   `require_radiomics=True` → 92 cases on contrast-only, while arm_c uses
   189. The fix is to rerun arm_b without the radiomics filter (or use
   arm_a equivalent on contrast-only) to get same-case-set comparison.

3. **Graph-stats coverage is 243/282 (87%)**: 39 cases missing from
   `cache_v2_tri_flat`. Must audit which/why for the W6 exclusion-sensitivity
   demand.

Round 6 minimum:
- Same-case-set arm_b vs arm_c paired DeLong with case_id-anchored OOF dumps
- 39-case audit + GCN exclusion-sensitivity
- Remote env lock + kimimaro version pinned

## Round 3 key concerns (2026-04-23 16:40) — CARRY INTO ROUND 4

**Score 4/10 reject.** Main new insight over Round 2:

**Protocol-AUC should be measured WITHIN label=0 only** (27 contrast nonPH
vs 85 plain-scan nonPH), not across the full 282. Measuring across the
cohort lets the model shortcut via `label → contrast` (since 170/170 PH
are contrast), so "protocol decodability" was entangled with label
decodability. The within-nonPH test is the honest W1 endpoint.

Round 4 minimum-to-reach-8:
1. Protocol-matched primary analysis (matching/weighting OR adequate contrast nonPH).
2. Protocol decodability on *exact GCN inputs* computed **within nonPH only**.
3. Per-case val-prob dumps → paired DeLong on headline deltas (single predefined primary endpoint).
4. Overlay gallery + TEASAR parameter-sensitivity sweep (anatomical validity).
5. Exclusion sensitivity (placeholders retained with degraded handling).
6. Locked reproducibility: conda-lock, kimimaro version from remote, git SHA in cache metadata.

## Round 2 key concerns (2026-04-23 15:45) — CARRY INTO ROUND 3

**Score 3/10 reject.** Key new concerns beyond Round 1:

1. **Protocol/label are near-perfectly entangled in the cohort.** 170 PH cases
   are contrast, 85 nonPH plain-scan, only 27 nonPH are contrast (the only
   "free variation" in protocol within a label). The contrast-only ablation
   (26 nonPH negatives, 3–7 per fold) is underpowered. Contrast-only results
   cannot rescue the claim. Primary endpoint must become protocol-matched.

2. **Protocol decodability must be tested on the EXACT GCN/cache features,
   not separate scalar lung features.** §14.3 proves scalars leak protocol
   perfectly; Round 3 must prove whether the graph-node features do too,
   using the same training pipeline, to know whether the residual 0.87 AUC
   is disease or residual protocol.

3. **Statistical inference is still not confirmatory-grade.** Fold-level
   paired tests are not a substitute for case-level paired AUC (DeLong).
   Per-case val-fold predictions must be persisted and paired DeLong reported.

4. **TEASAR / graph construction remains unvalidated anatomically.** No
   overlay audits, no skeleton-to-mask coverage metrics, no parameter
   sensitivity sweep.

5. **Reproducibility package is still text-only.** No env lockfile, no
   kimimaro version pin, no cache-builder commit hash, no one-command rebuild.

## Verification checklist for subsequent rounds

For each fix claimed in a new round, verify:
- [ ] **W1**: protocol-balanced subset (contrast-only PH vs contrast-only nonPH) reports separate AUC; protocol-prediction control reports an AUC.
- [ ] **W2**: case_id → patient_id mapping audited; fold splits verified patient-disjoint; ideally external or temporal validation present.
- [ ] **W3**: TEASAR parameter sensitivity sweep present; visual overlays for sample cases; skeleton-to-mask coverage reported.
- [ ] **W4**: node coords in mm; ct_density backfilled or explicitly removed; Strahler has_cycles flag.
- [ ] **W5**: exclusion sensitivity analysis present; exclusions broken down by label/protocol.
- [ ] **W6**: paired DeLong or bootstrap CIs on v1 vs v2; multiplicity correction or predefined primary endpoint.
- [ ] **W7**: airway claims scoped to appendix OR airway QC complete + all failed cases fixed.
- [ ] **W8**: env lockfile, kimimaro version pin, cache-builder commit hash, reproducibility one-liner present.
