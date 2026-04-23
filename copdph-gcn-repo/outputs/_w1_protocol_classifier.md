# W1 stress-test — lung-feature protocol-decoder baseline

Question: can the 14 scalar lung features in `lung_features_only.csv` predict
the **acquisition protocol** (contrast vs plain-scan) as accurately as they
predict **disease**? If yes, the residual ~0.87 disease AUC on the
contrast-only subset (§13) is a lower bound on confounding, not an upper bound.

## 5-fold stratified CV AUCs (scalar lung features only)

| Target | Logistic Regression | Gradient Boosting |
|---|---|---|
| **Protocol** (contrast vs plain-scan) | **1.0000 ± 0.0000** | **1.0000 ± 0.0000** |
| Disease (full cohort, label) | 0.8714 ± 0.0303 | 0.8980 ± 0.0409 |
| Disease (contrast-only, label) | 0.6767 ± 0.0620 | 0.6780 ± 0.0596 |

## Single-feature protocol decoders (ranked)

| Feature | CV AUC for protocol |
|---|---|
| `mean_HU` | 1.0000 ± 0.0000 |
| `std_HU` | 1.0000 ± 0.0000 |
| `HU_p5` | 1.0000 ± 0.0000 |
| `HU_p25` | 1.0000 ± 0.0000 |
| `lung_vol_mL` | 0.9005 ± 0.0712 |
| `LAA_950_frac` | 0.8846 ± 0.0516 |
| `LAA_910_frac` | 0.8846 ± 0.0516 |
| `LAA_856_frac` | 0.8846 ± 0.0516 |
| `HU_p50` | 0.8419 ± 0.1098 |
| `largest_comp_frac` | 0.8228 ± 0.0425 |
| `n_components` | 0.8228 ± 0.0425 |
| `HU_p75` | 0.7691 ± 0.1032 |
| `HU_p95` | 0.7691 ± 0.1032 |

## Interpretation

- Protocol is decoded with **AUC 1.000** from scalar lung
  features alone. Any classifier that uses these features has access to a near-perfect
  protocol cue.
- On the contrast-only subset the **same features** predict disease at AUC
  0.678 (LR 0.677) — a direct measure of the *residual*
  disease signal in the lung features after removing protocol confounding.
- The gap between full-cohort disease AUC and contrast-only disease AUC is explained
  by the protocol decoder: features that discriminate protocol mechanically contribute
  to the full-cohort disease AUC even when they have no causal disease signal.

**Reviewer-facing statement**: claims of disease-relevant lung-phenotype effects
must be supported by the contrast-only column above, not the full-cohort column.