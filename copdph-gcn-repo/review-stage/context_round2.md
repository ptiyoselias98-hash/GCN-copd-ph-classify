# ARIS Round 2 Context — COPD-PH GCN v2 Cache

**Round 1 verdict**: 2/10 reject. Eight weaknesses flagged (W1–W8). See
`REVIEWER_MEMORY.md` for the persistent memory.

**This round's claim**: all Round 1 concerns that can be addressed without a
server-side rebuild have been addressed. We are asking whether the evidence
now justifies a revised verdict and, if not, which specific weaknesses remain
blockers.

---

## What changed since Round 1

### W1 (critical) — acquisition-protocol confounding

1. **Protocol-balanced arm_b / arm_c ablation** (§13): retrain both arms on
   the contrast-enhanced-only subset (189 cases, 163 PH + 26 nonPH). Both
   arms drop 0.05–0.08 AUC. arm_c's lung-feature advantage drops from +0.04
   to +0.006 AUC (i.e. disappears).
2. **Paired statistical test** (§14.2, `_compute_ci_fold_level.py`):
   - Wilcoxon p < 0.0001 on arm_c − arm_b under full cohort.
   - Wilcoxon p = 0.49 on the same comparison under contrast-only.
   - Paired bootstrap Δ CI excludes zero for the full cohort, includes zero
     for contrast-only.
3. **Protocol decoder baseline** (§14.3, `_w1_protocol_classifier.py`):
   on 279 cases, scalar lung features predict `is_contrast` with AUC
   **1.000 ± 0.000** under both LR and GB. Single features `mean_HU`,
   `std_HU`, `HU_p5`, `HU_p25` each individually achieve protocol AUC 1.000.
   The same features predict disease at AUC 0.898 full cohort, 0.678
   contrast-only → ~0.22 AUC gap is pure protocol leakage.
4. **Protocol-robust lung phenotype v2** (§14.4,
   `_extract_lung_v2.py`): parenchyma-only features (lung − artery − vein −
   airway) + spatial apical/middle/basal LAA + vessel-lung integration
   features. Extraction completing during this round.
5. **Cohort protocol labels committed** (§14.5,
   `data/case_protocol.csv`): 170 PH contrast / 27 nonPH contrast /
   85 nonPH plain-scan, zero unmatched from the original DCM folders.

### W2 (critical) — patient-level validation

`_audit_patient_leakage.py` parses case_id → (label, pinyin, id) patient key
and verifies per-fold train/val disjointness. Result: 282 cases from 282
unique patients (one scan per patient); zero leakage possible by
construction. External/temporal validation remains an open limitation.

### W6 (major) — statistical reporting

Replaced point-estimate AUC comparisons with 95% percentile bootstrap CIs
and paired Wilcoxon / paired-t on the 15 fold AUCs (5-fold × 3 repeats).
All pairwise deltas between arms × protocols are reported with CI + p.
**Caveat**: fold-level only (per-case probs not persisted); a server rerun
to dump val-fold probs → true DeLong is queued.

### W4 (major) — feature engineering defects

Explicitly listed in §11.3 as limitations, not silently carried. Not yet
fixed in v2 cache because fixing them forces a rebuild that invalidates all
Sprint 6 results. Proposal: freeze v2 for the reviewer-facing submission,
ship a `v3_mm` builder with (a) mm-space centroid coords, (b) ct_density
backfilled from raw CT, (c) Strahler with cycle-handling — run in parallel
on the remote GPU nodes while main results are being written up.

### W7 (critical) — airway claims

Scoped to appendix (§11.5, §10.1 now carries an airway-caveat footnote).
No disease claim is drawn from the airway channel; arm_c's lung-feature
advantage is the one that we retract in §13.5 under protocol balancing.

### W8 (major) — reproducibility

Expanded §12 with explicit TODOs for `environment.yml`, kimimaro version
pin, cache-builder commit hash, one-command rebuild script. No new
artifacts published yet.

### Still not done (explicit list for this round)

- **W3 TEASAR sensitivity** — requires remote rebuild, deferred.
- **W5 exclusion-sensitivity** — 27 nonPH placeholder cases retention is a
  substantial methodology change; plan is to impute empty graphs and rerun
  arm_b / arm_c, but not done in this round.
- **W6 case-level DeLong** — only fold-level replacement so far.
- **W8 environment freeze** — pin/lockfile TODO.

---

## The reviewer question for Round 2

1. Do §13 + §14 materially change the verdict, or do they only document what
   Round 1 already suspected?
2. Are there new confounders we missed that the v2 parenchyma feature set
   should be tested against?
3. What is the minimum set of remaining items (among W3/W5/W6-upgrade/W8)
   that must land before the paper is defensible at a top venue?
4. Is the Round-1 reviewer memory fully satisfied, or does anything need
   sharper documentation?

Please score 1–10 again and list must-fix-before-next-round items.
