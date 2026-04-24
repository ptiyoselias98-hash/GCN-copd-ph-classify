# R6.2 — Cache_v2_tri_flat coverage audit

Total: 282 cases in protocol table.
In cache: **243**, missing: **39**.

| label | protocol | in cache | missing |
|---|---|---|---|
| 0 | contrast | 26 | 1 |
| 0 | plain_scan | 54 | 31 |
| 1 | contrast | 163 | 7 |

## Missing cases by reason

- **placeholder_vessel**: 32
- **other_unaccounted**: 5
- **lung_v2_error**: 2

## Reading

Per project memory `project_v2_cache_missing_segmentations.md`: 27 nonPH
have placeholder vessel masks (upstream segmentation failed on plain-scan
CT) and ~7 PH have absent vessel files. The R6.2 audit confirms which
cases the v2 cache builder dropped vs which it kept (with degraded inputs).

If the missing 39 cases are all label-correlated (mostly nonPH), then the
243-case sample is biased toward PH (163/243=67%) vs the 282-case truth
(170/282=60%). Round 6 will rebuild a small subset of placeholder cases
with degraded-graph handling to test exclusion sensitivity at the GCN level.