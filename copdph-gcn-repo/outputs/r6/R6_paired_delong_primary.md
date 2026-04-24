# R6.1 — Paired DeLong on arm_c − arm_a contrast-only (PRIMARY ENDPOINT)

Same 189 contrast-only cases (163 PH + 26 nonPH), gcn_only mode, ensembled across 5 folds × 3 repeats.

- AUC arm_a (vessel-only): **0.8391**
- AUC arm_c (vessel + 13 lung scalar globals): **0.8143**
- Δ AUC (arm_c − arm_a): **-0.0248**
- DeLong 95% CI on Δ: **[-0.0887, +0.0391]**  (INCLUDES 0 — NOT significant)
- DeLong z = 0.760, **p two-sided = 0.4474**
- Bootstrap 95% CI on Δ (n=5000): **[-0.0354, +0.0935]**

## Interpretation

- This is the W6 case-level paired confirmatory test the Round-4/5 reviewers required.
- Single pre-specified endpoint, no multiplicity correction needed.
- arm_c lung-feature contribution under protocol balancing: NOT statistically significant.
- Conclusion is consistent with the Round-2 §13.5 retraction of the lung-feature claim.