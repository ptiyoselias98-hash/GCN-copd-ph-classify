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
