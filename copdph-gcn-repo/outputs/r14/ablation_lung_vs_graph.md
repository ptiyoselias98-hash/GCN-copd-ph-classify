# R14.D — Lung-only vs graph-only disease classifier ablation

Within-contrast cohort (no protocol confound). 5-fold OOF LR, case-level bootstrap CI.

**n_total**: 184 (PH=158, nonPH=26). Excluded 38 seg-failure cases.

## Headline ablation

| feature set | n_feats | AUC [95% CI] |
|---|---|---|
| graph_only | 50 | 0.782 [0.676, 0.877] |
| lung_only | 49 | 0.844 [0.754, 0.917] |
| graph+lung | 99 | 0.867 [0.796, 0.930] |

## Within-graph substructure ablation

Restricts the graph feature set by substring match in feature name.

| substring | n_feats | AUC [95% CI] |
|---|---|---|
| x0 | 3 | 0.746 [0.622, 0.848] |
| x1 | 12 | 0.807 [0.699, 0.899] |
| x2 | 3 | 0.789 [0.694, 0.876] |
| e0 | 2 | 0.602 [0.490, 0.712] |
| e1 | 2 | 0.783 [0.678, 0.873] |

## Interpretation

- If graph_only AUC ≈ graph+lung AUC: lung-parenchyma is largely redundant
  with vascular graph features for disease classification (graph subsumes lung).
- If lung_only AUC > graph_only AUC: lung-parenchyma is the primary disease
  signal carrier (lung dominates); the graph adds little.
- If lung_only ≈ graph_only and graph+lung > both: complementary information.
- Within-graph substring ablation shows which structure (artery/vein/airway)
  carries the most disease-discriminative signal.
