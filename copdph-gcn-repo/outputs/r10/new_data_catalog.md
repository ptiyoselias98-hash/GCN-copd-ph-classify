# R10 new-data catalog

Refill (previously placeholder) folder: `H:\424\copdnoph平扫性` → 24 cases
New-patient folder: `H:\424新增copdnoph平扫性` → 76 cases
Total new entries: **100**

## Cross-tab source × already-in-protocol-table

| source_tag | already_in_protocol_table | count |
|---|---|---|
| new | 0 | 76 |
| refill | 0 | 24 |

**Refill matching existing cohort**: 0/24 — these replace the
placeholder-vessel cases flagged in `project_v2_cache_missing_segmentations.md`
(expected 27 such cases; matching 24 is plausible given pinyin variant spellings).

**New-patient matching existing cohort**: 0/76 — these should all
be 0 if the new folder is truly disjoint.

## Impact on cohort statistics (projected)

- Current: 170 PH + 112 nonPH = 282 cases (55% contrast, 45% plain-scan).
- After ingesting these 100 additions + ~24 refills that replace placeholders:
  ~170 PH (unchanged) + ~210 nonPH = **~380 total**, 45% contrast / 55% plain-scan.

**Pipeline required** (Round 10+):
1. DCM → NIfTI per case (dcm2niix).
2. Lung/artery/vein/airway segmentation — HiPaS-style unified model ideal.
3. Build v2 cache with kimimaro (existing `_remote_build_v2_cache.py`).
4. Rerun protocol-prediction tests — expected gain: tighter CIs on within-nonPH LR,
   potentially enabling domain-adversarial training to reach ≤0.6 target.