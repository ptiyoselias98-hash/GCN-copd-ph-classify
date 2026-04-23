# Adversarial Review Context — Round 1
_Generated 2026-04-23 10:31:46_
## 1. REPORT_v2.md (current draft)
```markdown
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


## 9. Multi-structure Phenotype Evolution (auto)

# Phenotype Evolution Report
- n_cases: 280
- n_features: 34
- n_clusters: 4
- COPD-no-PH (label=0): 110
- COPD-PH (label=1): 170

## Cluster 0 (n=18, dominant label=0, label_counts={0: 18})
- LAA_950_frac: z=+3.79 (mean=0.946)
- LAA_910_frac: z=+3.61 (mean=0.956)
- LAA_856_frac: z=+2.78 (mean=0.974)
- HU_p95: z=-2.43 (mean=-932)
- HU_p75: z=-2.26 (mean=-1.02e+03)
- mean_HU: z=-2.18 (mean=-1.01e+03)

## Cluster 1 (n=198, dominant label=1, label_counts={0: 28, 1: 170})
- HU_p5: z=+0.63 (mean=0)
- HU_p25: z=+0.59 (mean=0)
- mean_HU: z=+0.56 (mean=0.0719)
- std_HU: z=-0.54 (mean=0.25)
- HU_p50: z=+0.53 (mean=0)
- HU_p75: z=+0.52 (mean=0)

## Cluster 2 (n=45, dominant label=0, label_counts={0: 45})
- largest_comp_frac: z=-1.87 (mean=0.682)
- HU_p25: z=-1.72 (mean=-889)
- HU_p50: z=-1.72 (mean=-856)
- mean_HU: z=-1.70 (mean=-836)
- HU_p5: z=-1.68 (mean=-930)
- HU_p75: z=-1.67 (mean=-808)

## Cluster 3 (n=19, dominant label=0, label_counts={0: 19})
- n_components: z=+3.25 (mean=13.2)
- std_HU: z=+3.10 (mean=216)
- lung_vol_mL: z=-1.69 (mean=223)
- HU_p95: z=+0.77 (mean=84.4)
- HU_p5: z=-0.74 (mean=-550)
- HU_p75: z=+0.65 (mean=48.3)

## High-risk COPD-noPH cases (closest to PH centroid in UMAP)
| case_id | cluster | ph_proximity |
|---|---|---|
| nonph_lixiangqing_9002785345_thursday_august_31_2023_001 | 1 | -0.002 |
| nonph_fengjianhong_9002916926_friday_november_10_2023_000 | 1 | -0.022 |
| nonph_yangjicai_9001716005_sunday_september_27_2020_001 | 1 | -0.029 |
| nonph_wentaomei_9002233843_friday_july_15_2022_000 | 1 | -0.043 |
| nonph_shenjuhua_u13807530_thursday_november_1_2018_000 | 1 | -0.046 |
| nonph_xugaofeng_0800392509_monday_october_26_2015_001 | 1 | -0.049 |
| nonph_caochenglin_g02017953_thursday_july_9_2020_000 | 1 | -0.050 |
| nonph_chenzhanghu_5002094379_wednesday_november_20_2019_000 | 1 | -0.051 |
| nonph_huichunyi_0800212702_tuesday_july_16_2013_000 | 1 | -0.061 |
| nonph_heyude_c00242568_saturday_october_11_2014_000 | 1 | -0.083 |
| nonph_dinghuiyan_9002758757_tuesday_december_26_2023_000 | 1 | -0.087 |
| nonph_lilele_0800247111_tuesday_september_6_2022_000 | 1 | -0.103 |
| nonph_lujianlan_9002629358_monday_may_8_2023_000 | 1 | -0.140 |
| nonph_chenyunhua_9001865098_tuesday_july_27_2021_000 | 1 | -0.146 |
| nonph_huangzutian_9002735639_monday_july_24_2023_000 | 1 | -0.166 |


## 10. Training status — arm_b (tri-flat) / arm_c (+ lung feats)
- Launched on GPU0 (arm_b) + GPU1 (arm_c) via `_remote_train_arm_bc_parallel.py`.
- Logs: `/tmp/_train_arm_b_triflat.log`, `/tmp/_train_arm_c_quad.log`.

```
## 2. Builder code key sections
```python
# --- mask extraction (post-patch 2026-04-22) ---
def build_structure(mask_path, label, struct, torch_mod, Data, kimimaro_mod, nib_mod):
    """Build one Data object for one structure of one case."""
    img = nib_mod.load(str(mask_path))
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
    # Robust mask extraction: NIfTI masks in this dataset use -2048 as the
    # background sentinel and carry raw HU inside the structure. A plain
    # `(arr > 0)` filter misses plain-scan airway/lung where lumen-air HU is
    # negative (~-900). Use `!= -2048` to catch any non-background voxel.
    raw = np.asarray(img.dataobj)
    arr = (raw != -2048).astype(np.uint32)
    mask_vox = int(arr.sum())

    if mask_vox < 100:
        return _empty_data(label, torch_mod, Data), {
            "skipped": True, "reason": "tiny_mask", "mask_vox": mask_vox,
            "spacing": spacing, "valid_structure": False,
            "missing_reason": "empty_mask", "failed_hard_qc": False,
        }

    
# --- per-structure TEASAR params ---
if struct in ("artery", "vein"):
            teasar = {
                "scale": 1.0, "const": 5,
                "pdrf_scale": 100000, "pdrf_exponent": 2,
                "soma_acceptance_threshold": 3500,
                "soma_detection_threshold": 750,
                "soma_invalidation_scale": 1.0,
                "soma_invalidation_const": 300,
                "max_paths": None,
            }
        else:  # airway
            teasar = {
                "scale": 1.5, "const": 10,
                "pdrf_scale": 100000, "pdrf_exponent": 4,
                "soma_acceptance_threshold": 3500,
                "soma_detection_threshold": 750,
                "soma_invalidation_scale": 1.0,
                "soma_invalidation_const": 300,
                "max_paths": None,
            }
        
# --- node features (12-d) ---
# node features (12-dim, matching legacy schema)
    x = np.zeros((num_nodes, 12), dtype=np.float32)
    incident_edge_idx = defaultdict(list)
    for ei, e in enumerate(all_edges):
        incident_edge_idx[e["a"]].append(ei)
        incident_edge_idx[e["b"]].append(ei)

    pos = np.array(all_nodes, dtype=np.float32)
    centroid = pos.mean(axis=0) if num_nodes > 0 else np.zeros(3)

    strahler = _strahler_orders(num_nodes, all_edges)

    for n in range(num_nodes):
        eis = incident_edge_idx[n]
        if eis:
            es = [all_edges[i] for i in eis]
            diameters = [e["radius_med_mm"] * 2 for e in es]
            lengths = [e["length_mm"] for e in es]
            torts = [e["tortuosity"] for e in es]
            x[n, 0] = float(np.mean(diameters))                 # diameter
            x[n, 1] = float(np.mean(lengths))                   # length
            x[n, 2] = float(np.mean(torts))                     # tortuosity
            # x[n, 3] ct_density — left at 0 (no CT lookup in v2)
            # x[n, 4:7] orientation — set to mean unit-vector along incident edges
            ovec = np.zeros(3)
            for e in es:
                # use stored chord direction proxy: use position diff to other endpoint
                other = e["b"] if e["a"] == n else e["a"]
                d = pos[other] - pos[n]
                norm = np.linalg.norm(d)
                if norm > 0:
                    ovec += d / norm
            if np.linalg.norm(ovec) > 0:
                ovec /= np.linalg.norm(ovec)
            x[n, 4:7] = ovec
        # x[n, 7:10] centroid — relative position
        x[n, 7:10] = pos[n] - centroid
        x[n, 10] = float(strahler.get(n, 1))                    # strahler
        x[n, 11] = float(len(eis))                              # degree

    
# --- hard-fail QC ---
# hard-fail tier per codex: vox/key>2000 OR (mask_vox>100k AND num_nodes<100)
    if struct in ("artery", "vein"):
        qc["failed_hard_qc"] = bool(
            vk > 2000 or (mask_vox > 100_000 and nn < 100)
        )
    else:
        qc["failed_hard_qc"] = False
    qc["valid_structure"] = (not qc.get("skipped", False)) and (not qc["failed_hard_qc"])
    qc["missing_reason"] = None
    # top-3 longest edges with endpoint indices for spot-check overlay
    if data
```
## 3. 282-case cohort scan summary
```json
{
  "n_cases": 282,
  "summary_by_structure": {
    "artery": {
      "posHU_ok": 46,
      "mixed": 208,
      "placeholder_768": 27,
      "negHU_missed_by_gt0": 1
    },
    "vein": {
      "posHU_ok": 47,
      "mixed": 202,
      "placeholder_768": 27,
      "negHU_missed_by_gt0": 6
    },
    "airway": {
      "negHU_missed_by_gt0": 53,
      "mixed": 206,
      "posHU_ok": 20,
      "placeholder_768": 3
    }
  },
  "fully_ok_count": 201,
  "needs_patch_only_count": 54,
  "has_placeholder_count": 27,
  "has_truly_missing_count": 0
}
```
## 4. 47-case airway rebuild verification
```json
{
  "summary": {
    "valid": 41,
    "invalid": 2,
    "missing_pkl": 0,
    "errored": 4
  },
  "results": {
    "nonph_baoxiaoping_9001193765_thursday_january_2_2020_000": {
      "valid": false,
      "skipped": true,
      "mask_vox": 0,
      "num_nodes": 0,
      "num_edges": 0,
      "reason": "build_failed"
    },
    "nonph_baozhiding_k00748487_tuesday_june_18_2019_000": {
      "valid": true,
      "skipped": false,
      "mask_vox": 181527,
      "num_nodes": 235,
      "num_edges": 234,
      "reason": null
    },
    "nonph_caizhenzhang_d00133633_saturday_january_10_2015_000": {
      "valid": true,
      "skipped": false,
      "mask_vox": 207891,
      "num_nodes": 361,
      "num_edges": 360,
      "reason": null
    },
    "nonph_chaiguofeng_j00685132_monday_november_10_2014_000": {
      "valid": true,
      "skipped": false,
      "mask_vox": 183966,
      "num_nodes": 164,
      "num_edges": 163,
      "reason": null
    },
    "nonph_changhairong_e01744812_wednesday_january_19_2022_000": {
      "valid": true,
      "skipped": false,
      "mask_vox": 98897,
      "num_nodes": 225,
      "num_edges": 224,
      "reason": null
    },
    "nonph_chenjinzhi_9001551084_friday_january_3_2020_000": {
      "valid": false,
      "skipped": true,
      "mask_vox": 0,
      "num_nodes": 0,
      "num_edges": 0,
      "reason": "build_failed"
    },
    "nonph_chenqilong_j05095158_tuesday_december_3_2019_000": {
      "valid": true,
      "skipped": false,
      "mask_vox": 147359,
      "num_nodes": 331,
      "num_edges": 330,
      "reason": null
    },
    "nonph_chenyunhua_9001865098_tuesday_july_27_2021_000": {
      "valid": true,
      "skipped": false,
      "mask_vox": 177875,
      "num_nodes": 263,
      "num_edges": 262,
      "reason": null
    },
    "nonph_dangshizhong_m0411251x_friday_november_6_2020_000": {
      "valid": true,
      "skipped": false,
      "mask_vox": 163143,
      "num_nodes": 234,
      "num_edges": 233,
      "reason": null
    },
    "nonph_dingzhengde_s01699961_wednesday_december_9_2020_000": {
      "valid": true,
      "skipped": false,
      "mask_vox": 123931,
      "num_nodes": 225,
      "num_edges": 224,
      "reason": null
    },
    "nonph_fengxiaoguo_g03354804_monday_april_13_2020_000": {
      "valid": true,
      "skipped": false,
      "mask_vox": 206964,
      "num_nodes": 261,
      "num_edges": 260,
      "reason": null
    },
    "nonph_ganxiufang_l00829997_friday_august_16_2019_000": {
      "error": "[Errno 2] No such file or directory: '/home/imss/cw/GCN copdnoph copdph/tri_structure/cache_tri_v2/nonph_ganxiufang_l00829997_friday_august_16_2019_000_tri.pkl'"
    },
    "nonph_gaodaxiang_9001646216_thursday_july_2_2020_000": {
      "valid": true,
      "skipped": false,
      "mask_vox": 139613,
      "num_nodes": 154,
      "num_edges": 153,
      "reason": null
    },
    "nonph_gejianping_m03069169_monday_may_20_2019_000": {
      "valid": true,
      "skipped": false,
 
```
## 5. Sprint 6 results on v2 flat cache
