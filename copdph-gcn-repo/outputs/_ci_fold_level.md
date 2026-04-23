# Fold-level bootstrap CIs + paired tests (W6, first pass)

Per-arm 95% percentile bootstrap CIs on mean fold-AUC
(n_boot=10000, 15 fold AUCs = 5-fold × 3 repeats):

| Arm | n | mean AUC | 95% CI | SD |
|---|---|---|---|---|
| arm_b_full | 15 | 0.920 | [0.905, 0.934] | 0.031 |
| arm_c_full | 15 | 0.959 | [0.941, 0.975] | 0.034 |
| arm_b_contrast_only | 15 | 0.871 | [0.824, 0.915] | 0.095 |
| arm_c_contrast_only | 15 | 0.877 | [0.832, 0.918] | 0.088 |

Paired deltas (matched by fold index, 15 paired folds):

| Pair | Δ mean | 95% CI | Wilcoxon W | p | paired t | p |
|---|---|---|---|---|---|---|
| arm_c_full − arm_b_full | +0.039 | [+0.028, +0.052] | 0.00 | 0.0001 | 6.21 | 0.0000 |
| arm_c_contrast_only − arm_b_contrast_only | +0.006 | [-0.004, +0.018] | 46.50 | 0.4945 | 1.10 | 0.2909 |
| arm_b_full − arm_b_contrast_only | +0.049 | [+0.016, +0.084] | 27.00 | 0.0637 | 2.72 | 0.0167 |
| arm_c_full − arm_c_contrast_only | +0.082 | [+0.055, +0.109] | 0.00 | 0.0007 | 5.71 | 0.0001 |

**Caveat**: these are fold-level tests, not case-level DeLong. Proper DeLong on
paired AUCs requires per-case predicted probabilities, which are not persisted
in the current `sprint6_results.json`. A small rerun writing val-set probs
per fold is scheduled for Round 2 to replace these intervals with DeLong.