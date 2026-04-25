# R11 — Cohort N reconciliation (282 vs 243)

Per Round-10 reviewer: explicit accounting of full vs in-cache cohort,
missingness by (label, protocol) stratum, and what the 'n=243' results actually represent.

## (label × protocol) cross-tab

| label | protocol | full 282 | in cache 243 | missing 39 |
|---|---|---|---|---|
| 0 | contrast | 27 | 26 | 1 |
| 0 | plain_scan | 85 | 54 | 31 |
| 1 | contrast | 170 | 163 | 7 |
| **all** | | **282** | **243** | **39** |

## Per-fold val-split coverage

| fold | val cases (split) | val cases in cache |
|---|---|---|
| 1 | 57 | 49 |
| 2 | 57 | 47 |
| 3 | 56 | 48 |
| 4 | 56 | 50 |
| 5 | 56 | 49 |

## Missingness analysis

- 39/282 cases missing from cache_v2_tri_flat (13.8%).
- Of those missing, 31/39 (79%) are nonPH plain-scan, 7/39 (18%) are PH contrast, 1/39 (3%) are nonPH contrast.
- This is **strongly label/protocol-correlated**: most missing cases are plain-scan nonPH
  whose vessel segmentation produced 768-voxel placeholder files (per project memory).

## What 'n=243' results mean

All Sprint 6 / Round 5+ training and evaluation use the in-cache 243-case subset:
163 PH contrast + 26 nonPH contrast + 54 nonPH plain-scan. The 39 dropped cases are
predominantly plain-scan nonPH. This biases the evaluated cohort toward PH (67% vs 60%
in the full 282) and toward contrast (78% vs 70%).

**Implication for protocol-confound claims**: the within-nonPH protocol AUC is computed
on n=80 (26 contrast + 54 plain-scan). Sample is small and protocol-imbalanced (32%
contrast). If the 27 missing nonPH cases (24 plain + 3 PH) were retained with degraded
graphs, the within-nonPH stratum would grow to n=104 (29 contrast + 75 plain-scan),
which is the principled missingness handling Round 10 reviewer requested.

**Round 11 status**: this audit documents the gap; the rebuild + retrain on full 282
with degraded graphs requires GPU + builder rerun, queued for Round 12.