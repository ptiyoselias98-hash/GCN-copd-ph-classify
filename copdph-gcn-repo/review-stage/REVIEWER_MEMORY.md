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
