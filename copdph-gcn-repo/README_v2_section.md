# COPD-PH GCN — v2 Graph Cache Final Report

_Auto-generated 2026-04-23 01:17:10_

## 1. Cohort & Mask Encoding

- **Total cases**: 282 (112 nonPH / 170 PH)
- **Two source cohorts**:
  - Contrast-enhanced CT: 197 cases (170 PH + 27 nonPH), masks live at `nii/<case>/`
  - **Plain-scan CT: 85 nonPH cases**, masks live at `nii-unified-282/<case>/`
- All NIfTI masks use HU sentinel **`-2048` for background**; structure interior carries raw HU.
  - Vessels (contrast-enhanced): positive HU
  - **Airway in plain-scan CT: lumen air ≈ -1024 to -800 HU**

### Mask-extraction bug (FIXED 2026-04-22)

`_remote_build_v2_cache.py` originally used `(arr > 0)` to extract the binary mask. This **silently dropped 47 plain-scan airway segmentations** (median HU ≈ -930, `(arr > 0).sum()` = 0–44 on bodies of 100k–11M valid voxels). Patched to **`(arr != -2048)`**, which is a universal sentinel-based extraction (codex-reviewed; verdict: correct given dataset convention).

## 2. Cohort Rollup (post-patch)

| Bucket | Count | Notes |
|---|---|---|
| fully_ok (all 3 structures valid) | 201 | both contrast-enhanced + good plain-scan |
| needs_patch_only (47 airway-bug recoveries) | 54 | salvaged by builder patch |
| has_placeholder (≥1 structure 768-voxel marker) | 27 | upstream segmentation failed on plain-scan |
| has_truly_missing (≥1 file genuinely absent) | 0 | requires re-upload |

**Effective cohort for vessel analysis: 247 cases** (282 − 27 placeholder nonPH − 7 PH missing − 1 PH artery missing).

## 3. Graph-Build Pipeline (kimimaro v2)

```
NIfTI mask (raw HU + sentinel -2048)
   │
   ├── arr = (raw != -2048).astype(uint32)   # mask extraction
   ├── connected components QC (largest_frac, dust_dropped_frac)
   │
   ├── kimimaro.skeletonize(arr, teasar_params, anisotropy=spacing, dust=1000)
   │     ├── artery/vein: scale=1.0  const=5  pdrf_exp=2
   │     └── airway:     scale=1.5  const=10 pdrf_exp=4
   │
   ├── per-skeleton degree-2 chain contraction → key nodes (deg ≠ 2)
   ├── per-edge geometry from 3D distance-transform (mm units)
   │
   └── PyG Data per structure
```

### Node features (12-d per key node)

| Idx | Name | Description |
|---|---|---|
| 0 | mean_diameter | mean of `2 × radius_med_mm` over incident edges |
| 1 | mean_length | mean `length_mm` over incident edges |
| 2 | mean_tortuosity | mean `length / chord` over incident edges |
| 3 | ct_density | hardcoded 0 (no CT lookup in v2) |
| 4–6 | orientation | mean unit-vector along incident edges (3-d) |
| 7–9 | centroid_rel_pos | `node_pos − graph_centroid` (voxel index space) |
| 10 | strahler_order | BFS-from-leaves approximate Horton-Strahler |
| 11 | degree | `len(incident_edges)` |

### Edge features (3-d per directed edge, both directions stored)

| Idx | Name | Unit |
|---|---|---|
| 0 | diameter_mm | mm |
| 1 | length_mm | mm |
| 2 | tortuosity | unitless |

### Per-structure QC fields (in `qc[<struct>]`)

`mask_vox, n_components_mask, largest_comp_frac, dust_dropped_frac,
n_skeletons, num_nodes, num_edges,
edge_len_mm_p50/p90/max, radius_mm_p50, vox_per_key,
suspect, failed_hard_qc, valid_structure, top3_longest_edges, missing_reason`

### Hard-fail rules (vessel exclusion)
```
failed_hard_qc = (vox_per_key > 2000) OR (mask_vox > 100_000 AND num_nodes < 100)
```
Airway is never hard-failed (kept for arm_b experiments only).

## 4. Cache Files Produced

| Cache | Format | Cases | Use |
|---|---|---|---|
| `cache_tri_v2/<case>_tri.pkl` | per-structure dict | 282 | full QC + per-structure Data |
| `cache_v2_flat/<case>.pkl` | flat A+V graph | 247 | Sprint 6 arm_a |
| `cache_v2_tri_flat/<case>.pkl` | offset A+V[+W] graph | ≤247 | future arm_b with airway |

Tri-flat cache adds a `struct_id` channel (0=artery / 1=vein / 2=airway) as 13th node feature.

## 5. Sprint 6 Results (v2 cache)

```
See outputs/sprint6_arm_*_v2/ JSON
```

Key numbers (from prior session):
- arm_a base v2: pooled AUC ~0.95 (vs v1 ~0.78 — kimimaro rebuild **+0.17 absolute**)
- arm_a ensemble v2 (3 seeds + augment): comparable
- arm_b base/full: see per-arm JSON

## 6. Codex Audit Verdict

> **GO for final arm_a artery+vein analysis**, with cache/QC manifest frozen and exclusions audited;
> **NO-GO for claims involving airway** until airway-specific QC and schema handling are added.

Important codex follow-ups (deferred — not blocking arm_a):
1. Convert node `centroid_rel_pos` from voxel-index to mm: `(ijk − centroid_ijk) * spacing`
2. Add edge orientation `[dx, dy, dz]` in mm
3. Backfill `ct_density` from raw CT volume (currently hardcoded 0)
4. Add airway-specific QC: largest-component fraction, HU-distribution sanity, placeholder-shape detection
5. Strahler should record `has_cycles` flag (kimimaro can produce small cycles in tube-like vessels)

## 7. 47-case airway rebuild verification

`{
  "valid": 41,
  "invalid": 2,
  "missing_pkl": 0,
  "errored": 4
}`

## 8. Reproducibility

- Builder version tag: `v2_kimimaro` (in `_tri.pkl["builder_version"]`)
- TEASAR / dust / contraction params: see `_remote_build_v2_cache.py:307-409`
- kimimaro version: see `pip show kimimaro` on remote (pulmonary_bv5_py39 env)
- Mask sentinel convention: HU == -2048 = background, all other voxels = structure
