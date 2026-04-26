# R20 Locked Cohort & Provenance Manifest

_Closes R20 codex must-fix #4 (locked cohort/provenance table)._

This is the authoritative cohort manifest as of Round 20 (2026-04-26). All counts
are reproducible from artifacts in this repository.

## 1. Source data inventory

| Source | Cases | Type | Original pipeline |
|---|---|---|---|
| `H:\官方数据data\COPDPH_seg(160例增强性CT)` | 160 | PH contrast | HiPaS-style HU-sentinel masks |
| `H:\官方数据data\COPDnonPH_seg(27例增强性CT)` | 27 | nonPH contrast | HiPaS-style HU-sentinel masks |
| `H:\官方数据data\New folder-COPDNOPH 58例平扫性` | 58 | nonPH plain-scan | (legacy, partial) |
| `H:\官方数据data\4月24号-新增24个copdnoph平扫性` | 24 | nonPH plain-scan refill | new ingestion |
| `H:\官方数据data\4月24号-新增76个copdnoph平扫性` | 76 | nonPH plain-scan brand-new | new ingestion |
| **Total** | **345** | | |

## 2. Legacy 282 cohort (R1–R18 analyses)

| Stratum | n | Notes |
|---|---|---|
| PH contrast | 170 | original COPDPH_seg minus 10-overcount fix vs the 160 manifest |
| nonPH contrast | 27 | original COPDnonPH_seg |
| nonPH plain-scan | 85 | partial old ingestion of 58+24 (some not delivered cleanly) |
| **Total legacy 282** | **282** | |

`mpap_lookup_gold.json` resolves measured mPAP for **106/170 PH cases** (62%);
remaining 64 PH cases have NaN mPAP and use stage-based default in 5-stage analyses.

## 3. New100 cohort (R15 ingestion)

100 plain-scan nonPH cases (24 refill + 76 new) ingested in R15 via DCM→NIfTI;
segmented with Simple_AV_seg pipeline (lung+main_AV).

## 4. Unified-301 cohort (R20 — pipeline-unified)

Built in R20.F/G:
- 199 legacy contrast (174 PH + 27 nonPH; 1 corrupted gz `ph_zhangzongqi` excluded; 1 OOM)
- 100 new100 plain-scan
- **Sum = 299 unified-pipeline-segmented**

Cache build (R20.G via `_R19C_build_v2_patched.py`):
- **290 morphometric rows** in `outputs/r20/morph_unified301.csv`
  (10 missed: cache build had additional skips/failures)
- 84 per-structure features per case

## 5. R20.H verification subsets

| Analysis | n | Composition |
|---|---|---|
| Within-contrast cross-pipeline | 190 | 163 PH + 27 nonPH (Simple_AV_seg) |
| Severity-resolved (mPAP-known) | 102 | 75 PH (mPAP-resolved from gold lookup) + 27 nonPH (default 15) |
| Full unified-301 | 290 | 163 PH + 100 plain-scan nonPH + 27 contrast nonPH |

## 6. Pipeline naming convention (post-R20)

- **HiPaS-style legacy pipeline**: original mask delivery format (HU-sentinel
  -2048 background, structure interior carries raw HU). Public HiPaS code has no
  released checkpoint per upstream README — what we call "HiPaS-style" is the
  vendor-delivered legacy mask format.
- **Simple_AV_seg unified pipeline**: re-segmentation via the publicly-available
  Simple_AV_seg implementation (lung.pth + main_AV.pth from
  `/home/imss/cw/pulmonary_pipeline/code/Simple_AV_seg`). Binary uint8 masks.
  Used uniformly for both legacy contrast 199 and new100 plain-scan 100.

We do NOT claim the unified Simple_AV_seg pipeline is "HiPaS-equivalent" in
architecture or weights. We document only that the cross-sectional severity
DIRECTION is preserved across both pipelines (R20.H), while the legacy MAGNITUDE
ρ=-0.767 is pipeline-amplified (unified ρ=-0.211).

## 7. Exclusions / caveats locked at R20

| Exclusion | Reason | Source |
|---|---|---|
| `ph_zhangzongqi_9001787068_thursday_december_17_2020_000` | BadGzipFile CRC error in legacy CT | R20.F worker-2 log |
| 1 PH case OOM (~7-8GB allocation failure) | R20.A OOM during DDPM inference | R20 r20a_ph.log |
| 4 PH cases with 5D-array padding mismatch | R20.A unusual NIfTI shape | R20 r20a_ph.log |
| 8 R17 morphometric features | Numerical artifacts (n_terminals=0, lap_eig0~0, near-zero SD) | R20.C audit |
| Stage 0/1 mPAP defaults | plain-scan=5.0, contrast-nonPH=15.0 (not measured) | R18.B convention |
| 64 PH cases missing measured mPAP | gold lookup covers 106/170 only | mpap_lookup_gold.json |

## 8. What we do NOT have (cohort-level constraints)

- No paired same-patient longitudinal scans → cross-sectional severity ordering only
- No external/temporal validation cohort → in-sample evidence only
- No measured mPAP for nonPH cases (use protocol-based defaults 0–10 plain, 10–20 contrast)
- No HiPaS-checkpoint replication possible (proprietary; upstream has not released)
