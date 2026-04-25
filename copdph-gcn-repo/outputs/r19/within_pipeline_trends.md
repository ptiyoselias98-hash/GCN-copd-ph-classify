# R19.F — Within-pipeline mPAP trends (legacy vs new100 stratified)

Verifies R19.E confound diagnosis: if legacy-only ρ matches R18.B
(~−0.77 for artery_len_p25), the R19.D/.E extractor is consistent.
If new100-only also has strong trend in same direction, the enlarged
result is artifact of mixing two pipeline-distinct distributions.

## Per-subset stage counts

**legacy_only**: n=153, stages = 0:47, 1:27, 2:8, 3:44, 4:27
**new100_only**: n=78, stages = 0:78
**enlarged_all**: n=231, stages = 0:125, 1:27, 2:8, 3:44, 4:27

## Spearman ρ — same feature × 3 subsets

| feature | legacy_only ρ | new100_only ρ | enlarged_all ρ |
|---|---|---|---|
| artery_tort_p10 | -0.619 | — | -0.159 |
| artery_len_p25 | -0.767 | — | -0.282 |
| artery_len_p50 | -0.753 | — | -0.320 |
| artery_len_mean | -0.731 | — | -0.535 |
| artery_total_len_mm | +0.100 | — | +0.577 |
| vein_len_p25 | -0.613 | — | -0.139 |
| vein_total_len_mm | -0.286 | — | +0.430 |

## Interpretation

If legacy-only ρ matches R18.B (~−0.77 for artery_len_p25, ~+0.63
for paren_std_HU), the R19.D extraction is consistent with R17.
If new100-only ρ has similar magnitude but different sign / scale,
the pipeline confound is real. If new100-only ρ is null/weak, the
Simple_AV_seg masks lack discriminative topology compared to legacy.

Required next step: HiPaS re-segmentation of new100 (uniform
pipeline) before claiming enlarged-cohort evolution.
