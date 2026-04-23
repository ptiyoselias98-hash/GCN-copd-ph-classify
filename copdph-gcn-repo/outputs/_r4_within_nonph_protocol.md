# R4.1 Protocol decoder WITHIN nonPH only

Critical methodological fix (Round 3 reviewer): protocol AUC across the full
cohort conflates label↔protocol coupling (all 170 PH are contrast). This
test restricts to **label=0** cases (27 contrast + 85 plain-scan) to isolate
protocol leakage from label signal.

Cases: 112 (contrast=27, plain-scan=85)

5-fold stratified CV, 95% bootstrap CIs on mean CV AUC (2000 resamples):

| Feature set | n_feats | n_cases | Protocol AUC LR (95% CI) | Protocol AUC GB (95% CI) |
|---|---|---|---|---|
| `v1_whole_lung_HU` | 11 | 110 | 0.765 [0.697, 0.833] | 0.757 [0.666, 0.837] |
| `v2_parenchyma_only` | 10 | 93 | 0.794 [0.705, 0.886] | 0.731 [0.652, 0.825] |
| `v2_paren_LAA_only` | 3 | 93 | 0.715 [0.646, 0.789] | 0.673 [0.566, 0.762] |
| `v2_spatial_paren` | 4 | 93 | 0.669 [0.543, 0.795] | 0.652 [0.548, 0.748] |
| `v2_per_structure_volumes` | 4 | 110 | 0.529 [0.429, 0.631] | 0.702 [0.615, 0.771] |
| `v2_vessel_ratios` | 2 | 85 | 0.674 [0.542, 0.805] | 0.632 [0.441, 0.800] |
| `v2_combined_no_HU` | 9 | 73 | 0.731 [0.653, 0.810] | 0.664 [0.590, 0.737] |

## Interpretation

- Lowest LR protocol AUC (within nonPH): `v2_per_structure_volumes` at 0.529 (95% CI [0.429, 0.631]).
- Compare to R3 numbers (across full 282) — if a feature set has similar
  within-nonPH AUC, the R3 signal was real protocol; if within-nonPH drops
  to ~0.5, the R3 signal was mostly label-shortcut.
- Sample size is small (n=112); CIs are wider. A set with upper-CI below
  0.7 is defensible as 'protocol-robust' for the contrast/plain comparison.

**Warning**: 27 vs 85 is imbalanced (class balance 0.24). LR uses
`class_weight=balanced` to mitigate; GB does not. Prefer LR numbers for
the 'protocol-decodability' endpoint.