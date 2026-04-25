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
