# R16.A — Independent segmentation QC on 100 new plain-scan cases

Addresses R15 reviewer flag: Simple_AV_seg trained on CTPA → plain-scan
domain-transfer concern. No hand-label ground truth available; this is
a marginal-distribution sanity check using physiological priors.

**Total cases**: 100
**Cases with ≥1 QC flag**: 81 (81%)
**Cases with ≥3 QC flags (severe)**: 0

## Per-rule hit counts

| QC rule | n flagged |
|---|---|
| lung_vol_implausible_low | 0 |
| lung_vol_implausible_high | 79 |
| paren_HU_implausible_high | 0 |
| paren_HU_implausible_low | 4 |
| vessel_lung_ratio_too_high | 0 |
| vessel_lung_ratio_too_low | 0 |
| paren_too_few_voxels | 0 |
| av_ratio_extreme_high | 0 |
| av_ratio_extreme_low | 0 |

## Distribution summaries (physiological priors in parens)

| metric | min | p5 | median | p95 | max | plausible range |
|---|---|---|---|---|---|---|
| lung_vol_mL | 3693 | 5853 | 10839 | 16882 | 18901 | (1500-8500) |
| paren_mean_HU | -1283 | -900 | -874 | -828 | -788 | (-1000 to -500) |
| vessel/lung | 0.0051 | 0.0069 | 0.0132 | 0.0278 | 0.0341 | (0.005-0.25) |

Flagged cases listed in `seg_qc_new100_flagged.csv` (sorted by n_qc_flags).
