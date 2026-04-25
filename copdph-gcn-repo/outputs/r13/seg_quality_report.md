# R13.2 — Segmentation-quality audit on 345-cohort source folders

Per user 2026-04-25: even when 00000001..5 DCM counts match, some
cases have visibly broken segmentations. This script scans each
case for NIfTI masks (lung/airway/artery/vein) and flags
EMPTY/ALL_FILLED/TOO_SMALL/TOO_FRAGMENTED/LUNG_COMPONENT_ANOMALY.

**nibabel available**: True  | **scipy.ndimage.label**: True

**Total audited**: 345
**No masks found in source folder**: 345 (DCM-only cases — masks pending segmentation pipeline)
**Cases with quality issues**: 0
**Cases passing audit**: 0

## Cases flagged with quality issues

| group | case_id | bad masks | first issue |
|---|---|---|---|
| - | - | - | (none) |

## Cases without source-side masks (need pipeline ingestion)

Total: 345

(Listed in `seg_quality_report.json` under cases[].status='no_masks_in_source')