# R22.A — Proper deconfounding (nested CORAL + bootstrap CI)

Per R21 codex feedback: nested train-fold CORAL avoids test leakage; univariate per-feature alignment is less aggressive than full-covariance and avoids over-correction artifact.

## Within-contrast disease AUC (baseline, no CORAL needed within contrast)

- 5-seed × 5-fold = 25 fold AUCs
- Mean across seed-means = **0.710**
- Per-fold range = [0.438, 0.879]
- Bootstrap-500 95% CI = [0.729, 0.948] **(OVER-OPTIMISTIC per R22 codex; resample-with-replacement of 27 nonPH cases creates duplicates that inflate the CI relative to the true case-level uncertainty. DO NOT cite as the AUC's true 95% CI in the manuscript. Report only the mean (0.710), per-fold range [0.438, 0.879], and the n_nonPH=27 caveat.)**
- Cohort: n=190 (163 PH + 27 nonPH)
- **Caveat**: n_nonPH=27 small; per-fold AUC variance is real, not an artifact. Each fold has ~5-6 nonPH cases. The honest uncertainty is in the per-fold range, NOT the bootstrap CI.

## Within-nonPH protocol-AUC (orientation-free)

- No CORAL: orientation-AUC = **0.912**
- Nested univariate CORAL: orientation-AUC = **0.912**
- Drop: +0.000

**HONEST NEGATIVE**: nested CORAL still has high orientation-free leakage. Feature-level deconfounding insufficient — need richer approach (training-time GRL, deeper feature learning, or paper repositioning).