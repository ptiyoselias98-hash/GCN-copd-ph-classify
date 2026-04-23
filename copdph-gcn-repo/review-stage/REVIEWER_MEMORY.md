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
