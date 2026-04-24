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

- Launched 2026-04-23 10:29 via `_remote_launch_arm_bc_v2.py` using patched
  `run_sprint6_v2.py` (the earlier `_remote_train_arm_bc_parallel.py` referenced
  a non-existent `src/train_sprint6.py` and never actually trained).
- **arm_b** (GPU0): `--cache_dir cache_v2_tri_flat --keep_full_node_dim`
  (13-D node features incl. struct_id channel), gcn_only mode, 5-fold × 3
  repeats × 120 epochs, augment=edge_drop+feature_mask. Log at
  `outputs/sprint6_arm_b_triflat_v2/run.log`.
- **arm_c** (GPU1): same as arm_b + `--lung_features_csv lung_features_only.csv`
  injects 13 z-scored lung scalars (lung_vol, HU percentiles, LAA fractions,
  largest_comp_frac, n_components) as graph-level globals. Log at
  `outputs/sprint6_arm_c_quad_v2/run.log`.

### 10.1 Final metrics (2026-04-23, 5-fold × 3 repeats, 243 cases)

| Arm | Mode | AUC | Acc | Precision | Sensitivity | F1 | Specificity | pooled AUC |
|---|---|---|---|---|---|---|---|---|
| arm_b (tri-flat A+V+W, 13D)        | gcn_only | 0.920 ± 0.030 | 0.890 ± 0.037 | 0.937 ± 0.038 | 0.901 ± 0.086 | 0.915 ± 0.035 | 0.867 ± 0.089 | 0.900 |
| arm_c (tri-flat + 13 lung globals) | gcn_only | 0.959 ± 0.033 | 0.934 ± 0.027 | 0.959 ± 0.025 | 0.943 ± 0.041 | 0.950 ± 0.022 | 0.916 ± 0.056 | 0.947 |

**Observation**: arm_c > arm_b by ~0.04 AUC, consistent with lung scalar features
adding signal. **However**, per §11.1, this gain may be confounded with acquisition
protocol: HU percentiles and LAA fractions are very different between contrast-enhanced
and plain-scan CT, and all PH cases are contrast-enhanced. A protocol-balanced
ablation (contrast-only arm_c vs contrast-only arm_b) is required before claiming
the lung-feature contribution is disease-relevant rather than protocol-informative.

## 11. Limitations (explicit, reviewer-facing)

**11.1 Cohort/protocol confounding (CRITICAL)**. All 170 PH cases are
contrast-enhanced CT; 85/112 nonPH cases are plain-scan CT. The headline
arm_a pooled AUC ~0.95 therefore cannot be interpreted as disease
discrimination until it is verified on a protocol-balanced subset
(contrast-only PH vs contrast-only nonPH) and against a protocol-prediction
baseline. This analysis is queued as the first experiment after arm_b/c
training completes.

**11.2 Validation is internal 5-fold only**. No external or temporal
held-out test. Fold splits are case-id based; patient-level leakage has
not yet been verified via a case_id → patient_id audit. Claims of
generalization are deferred pending (a) patient-disjoint fold audit and
(b) ideally an external validation cohort.

**11.3 Feature-engineering defects (known, un-addressed in v2)**.
- `centroid_rel_pos` in voxel-index space, not mm.
- `ct_density` hardcoded to 0 (no CT-HU lookup yet).
- Strahler order is BFS-approximate; cycle handling and `has_cycles` flag
  are not recorded.
- Edge orientation uses index-space endpoint vectors, not mm.

**11.4 Ad-hoc QC thresholds**. Hard-fail rules (`vox_per_key>2000`,
`mask_vox>100k ∧ num_nodes<100`) are single thresholds chosen without a
sensitivity sweep. Exclusion of 27 placeholder nonPH alters the class
balance in a label-correlated way. An exclusion-sensitivity rerun is
queued.

**11.5 Airway results are appendix-only**. 47-case airway rebuild
verification: 41 valid / 2 invalid / 4 errored. There is no airway-specific
QC (largest-component fraction, HU-distribution sanity, placeholder-shape
detection). No airway-derived conclusion should be drawn from arm_b/arm_c
until airway QC is implemented and all failed rebuilds are resolved.

**11.6 Statistical reporting**. Current results lack paired DeLong /
bootstrap CIs on the v1→v2 improvement and do not correct for the
multiplicity of arm × feature-set × augmentation combinations. A
confirmatory analysis with predefined primary endpoint is required for
a top-venue submission.

## 12. Reproducibility (expanded)

- **Builder version tag**: `v2_kimimaro` (stored in each `_tri.pkl["builder_version"]`).
- **kimimaro version**: run `pip show kimimaro` on remote inside
  `pulmonary_bv5_py39` conda env (ver. pinned by `conda env export`,
  TODO: publish `environment.yml` alongside release).
- **TEASAR / dust / contraction params**: see `_remote_build_v2_cache.py:307-409`.
- **Mask sentinel convention**: HU == -2048 = background; all other voxels = structure.
- **Cache manifest**: `cache_v2_flat/manifest.json` (247 entries),
  `cache_v2_tri_flat/manifest.json` (243 entries).
- **Fold splits**: `data/splits_expanded_282/fold_{1..5}.csv` (case-id level;
  patient-id audit pending).
- **Result JSONs**: `outputs/sprint6_arm_{a_base,a_ensemble,b_base,b_full}_v2/sprint6_results.json`.
- **TODO before submission**: publish `environment.yml` lockfile, cache-builder
  git commit hash, one-command rebuild script, and de-identified patient-id map.

## 13. W1 protocol-confound ablation (2026-04-23)

Direct response to the hostile-review W1 concern (ARIS Round 1 reviewer memory):
**is the AUC ~0.95 disease signal, or is it acquisition-protocol classification?**

### 13.1 Design

Retrain arm_b and arm_c on the contrast-enhanced-only subset (197 cases
in the label file → 189 cases after intersection with `cache_v2_tri_flat`:
163 PH + 26 nonPH). If the headline numbers survive this restriction, the
protocol-confound concern is weakened; if they collapse, the signal was
partly or wholly driven by contrast-enhanced-vs-plain-scan cues.

Identical training config to §10.1: 5-fold × 3 repeats × 120 epochs,
batch=16, `--keep_full_node_dim --skip_enhanced --augment
edge_drop,feature_mask`. Splits preserved from the 282-cohort splits by
filtering each `fold_*/{train,val}.txt` to the contrast subset. Class
imbalance in val folds: 3–7 nonPH per fold.

### 13.2 Results (5-fold × 3 repeats, 189 cases, 163 PH + 26 nonPH)

| Arm | AUC | Acc | Precision | Sensitivity | F1 | Specificity | pooled AUC |
|---|---|---|---|---|---|---|---|
| arm_b (tri-flat, contrast-only) | 0.871 ± 0.092 | 0.858 ± 0.047 | 0.978 ± 0.029 | 0.857 ± 0.050 | 0.912 ± 0.029 | 0.889 ± 0.145 | 0.821 |
| arm_c (tri-flat + lung globals, contrast-only) | 0.877 ± 0.085 | 0.862 ± 0.043 | 0.984 ± 0.022 | 0.855 ± 0.042 | 0.914 ± 0.028 | 0.922 ± 0.103 | 0.862 |

### 13.3 Comparison with full 243-case cohort (§10.1)

| Arm | full-cohort AUC | contrast-only AUC | Δ |
|---|---|---|---|
| arm_b | 0.920 ± 0.030 | 0.871 ± 0.092 | **−0.049** |
| arm_c | 0.959 ± 0.033 | 0.877 ± 0.085 | **−0.082** |
| arm_c − arm_b | +0.039 | **+0.006** | — |

### 13.4 Interpretation (direct)

- **Partial confounding confirmed.** Both arms drop 0.05–0.08 absolute AUC
  when the acquisition protocol is balanced. The headline AUC in §10.1
  was inflated by acquisition cues.
- **Residual signal is real but modest.** At AUC ~0.87 with small negatives
  (26 contrast nonPH), arm_b/c still discriminate PH from nonPH above
  chance — but this is the honest upper bound of what these features
  currently measure.
- **The arm_c lung-feature advantage mostly disappears** on the balanced
  cohort (+0.04 → +0.006 AUC). The lung HU/LAA scalars were informative
  primarily because HU distributions differ between contrast-enhanced and
  plain-scan CT, not because they encode disease beyond what vessels
  already capture.
- **Variance is high** (±0.08–0.09 AUC) due to 26 nonPH split across 5 folds
  (3–7 per fold). Confidence intervals / DeLong on the Δ will be reported
  in Round 2.

### 13.5 Implications for claims

1. **Drop the v1→v2 “+0.17 absolute AUC” narrative** in any public
   description; the honest number in a protocol-balanced evaluation is
   closer to ~0.87.
2. **Retract the arm_c lung-feature contribution claim** as a disease
   result. Report it only as a dataset observation that disappears under
   protocol balancing.
3. **Next-round experiments queued**: (a) protocol-prediction baseline
   (train arm_b to predict `is_contrast_enhanced` from the graph — if
   AUC → 1.0, protocol is trivially decodable from graphs and the §13
   residual 0.87 is a *lower* bound on confounding, not an upper bound);
   (b) patient-id leakage audit on the contrast-only folds; (c) DeLong /
   bootstrap CIs on all Δ comparisons.

### 13.6 Artifacts

- Results: `outputs/sprint6_arm_{b,c}_contrast_only_v2/sprint6_results.json`
- Logs: `outputs/sprint6_arm_{b,c}_contrast_only_v2/run.log`
- Launcher: `_remote_launch_w1_ablation.py`
- Contrast subset labels/splits: `data/labels_contrast_only.csv`,
  `data/splits_contrast_only/fold_{1..5}/{train,val}.txt` (pushed to remote).


## 14. ARIS Round 2 — targeted fixes (2026-04-23)

Addresses the reviewer memory from Round 1 beyond what §13 already covered.

### 14.1 W2 — patient-level fold leakage audit (RESOLVED)

Script: `_audit_patient_leakage.py`. Result
(`outputs/_patient_leakage_audit.md`):

- 282 cases → 282 unique patients (scan-count histogram `{1: 282}` — one scan
  per patient).
- Zero leakage possible across any of the 5 folds. Fold splits are already
  patient-disjoint by construction because each patient contributes exactly
  one case.
- External/temporal validation remains an open limitation (§11.2).

### 14.2 W6 — fold-level paired CIs on key AUC deltas (FIRST PASS)

Script: `_compute_ci_fold_level.py`. Result
(`outputs/_ci_fold_level.md`):

| Arm | mean AUC (15 folds) | 95% percentile-bootstrap CI |
|---|---|---|
| arm_b full | 0.920 | [0.905, 0.934] |
| arm_c full | 0.959 | [0.941, 0.975] |
| arm_b contrast-only | 0.871 | [0.824, 0.915] |
| arm_c contrast-only | 0.877 | [0.832, 0.918] |

Paired deltas (15 matched folds, paired bootstrap Δ + Wilcoxon + paired-t):

| Pair | Δ mean | 95% CI | Wilcoxon p | paired-t p |
|---|---|---|---|---|
| arm_c_full − arm_b_full | +0.039 | [+0.028, +0.052] | **0.0001** | **0.0000** |
| arm_c_contrast-only − arm_b_contrast-only | +0.006 | [−0.004, +0.018] | 0.49 | 0.29 |
| arm_b_full − arm_b_contrast-only | +0.049 | [+0.016, +0.084] | 0.064 | **0.017** |
| arm_c_full − arm_c_contrast-only | +0.082 | [+0.055, +0.109] | **0.0007** | **0.0001** |

Reading:

- arm_c's lung-feature gain is **highly significant on the full cohort (p < 0.0001)
  but vanishes under protocol balancing (p = 0.49)** — direct statistical
  confirmation of the §13 interpretation.
- arm_c drops ~2× more under protocol balancing than arm_b does
  (0.082 vs 0.049; both 95% CIs exclude zero). Lung features are the most
  protocol-confounded ingredient; vessel graph features retain more signal.

**Caveat**: these are fold-level tests, not case-level DeLong. Sprint6 result
JSONs do not persist per-case probabilities; a follow-up rerun writing
val-fold probabilities is scheduled to replace these intervals with true
DeLong CIs.

### 14.3 W1 stress-test — trivial protocol decodability (NEW)

Script: `_w1_protocol_classifier.py` (local, sklearn). Result
(`outputs/_w1_protocol_classifier.md`):

| Target | Logistic Regression | Gradient Boosting |
|---|---|---|
| **Protocol** (contrast vs plain-scan) | **1.000 ± 0.000** | **1.000 ± 0.000** |
| Disease (full cohort) | 0.871 | 0.898 |
| Disease (contrast-only) | 0.677 | 0.678 |

Single-feature protocol decoders (5-fold CV AUC on 279 cases):
`mean_HU` / `std_HU` / `HU_p5` / `HU_p25` — all AUC = **1.000 ± 0.000**.
I.e., any single one of these features perfectly separates contrast from
plain-scan CT.

Reading:

- Whole-lung scalar HU features contain **complete protocol information**.
  Any classifier using them can decode protocol with zero error.
- On the same features, disease AUC drops from **0.898 (full cohort) to
  0.678 (contrast-only)** — a ~0.22 absolute gap that is entirely protocol
  leakage under these features.
- The implication for §13 is stronger than originally stated:
  the ~0.87 contrast-only GCN AUC is an **upper bound** on disease signal
  only if the graph is protocol-invariant; since v2 node features do NOT
  carry HU (`ct_density` is hardcoded 0, coords are index-space),
  the protocol cue in the graph can only come from segmentation-quality
  differences. That is a narrower, testable channel for Round 3 follow-up.

### 14.4 Protocol-robust lung phenotype v2 (COMPLETE)

Script: `_extract_lung_v2.py`. Output: `outputs/lung_features_v2.csv`
(282 cases × 51 features, 275 valid). Classifier comparison:
`_w1_protocol_classifier_v2.py` → `outputs/_w1_protocol_classifier_v2.md`.

**Critical builder discovery**: the two cohorts use different mask
conventions. Plain-scan cohort (`nii-unified-282/`) stores raw HU in the
mask file with `-2048` background sentinel. Contrast cohort (`nii/`,
reached via `_source.txt` redirects) uses binary 0/1 masks and the HU
must be read from a separate `ct.nii.gz`. The extractor auto-detects
both conventions (`mask_convention` column: `hu` or `binary`). The v1
lung-feature pipeline did NOT honour the contrast convention — it
read whole-lung HU from the mask file directly which for the contrast
cohort returned {0, 1} values, causing the degenerate "mean_HU ≈ 0.08"
numbers that drove the v1 protocol AUC to 1.000.

**Protocol-robustness comparison (5-fold stratified CV on 252 valid cases)**:

| Feature set | n_feats | Protocol AUC (LR / GB) | Disease full (LR / GB) | **Disease contrast-only (LR / GB)** |
|---|---|---|---|---|
| whole_lung (v2 rebuild) | 11 | 0.900 / 0.889 | 0.879 / 0.873 | 0.824 / 0.734 |
| **parenchyma_only** | 10 | 0.857 / 0.851 | 0.870 / 0.841 | **0.860 / 0.777** |
| spatial_paren (apical/mid/basal LAA) | 10 | 0.808 / 0.853 | 0.761 / 0.856 | 0.732 / 0.654 |
| vessel_lung_integration | 7 | 0.945 / 0.982 | 0.861 / 0.890 | 0.774 / 0.677 |
| **paren_plus_spatial (combined)** | 14 | 0.866 / 0.857 | 0.879 / 0.852 | **0.855 / 0.793** |

For context, the v1 `lung_features_only.csv` whole-lung features (§14.3) gave
protocol AUC = **1.000** and disease contrast-only = **0.678**.

**Reading**:

1. The v2 whole-lung rebuild with correct HU sourcing already drops protocol
   AUC from 1.000 (v1 bug) → 0.900 (v2 correct) and raises disease
   contrast-only AUC from 0.678 → 0.824 (LR). A large portion of the Round 1
   protocol-confound finding was a v1 *implementation* bug, not an intrinsic
   feature property.
2. Subtracting vessels + airway before computing HU statistics (parenchyma_only)
   drops protocol AUC further to 0.857/0.851, and disease contrast-only
   reaches **0.860 (LR)** — a +0.036 absolute gain over whole-lung with
   LESS protocol confounding. Net-net: parenchyma-only is the correct
   whole-lung HU surrogate for disease analysis.
3. Adding apical/middle/basal LAA spatial features yields the best combined
   result: protocol 0.866, disease contrast-only **0.855 (LR)** —
   a +0.18 absolute disease recovery vs the v1 baseline on the same
   balanced 186-case subset.
4. `vessel_lung_integration` features (artery / vein / airway volumes and
   mean HU) have near-perfect protocol AUC 0.98, as expected — contrast
   enhancement dominates vessel HU. These features should be avoided for
   disease claims.

**Implication for §13**: the protocol confound story needs revision. The
§13 contrast-only arm_c AUC of 0.877 was on full graph+lung features;
under the correctly-computed v2 lung features, scalar lung features alone
already recover disease AUC ~0.855 on the same subset. The graph's
contribution above that is the remaining open question for Round 3.

### 14.7 E2 parenchyma phenotype cluster (v2 features, 252 cases)

Script: `scripts/evolution/E2_parenchyma_cluster.py`. Output:
`outputs/evolution/E2_paren_cluster.{md,json}` + 2 PCA figures.

GMM k=5 (BIC-optimal). Cluster composition (baseline PH rate = 63.1%):

| Cluster | Size | PH% | Wilson 95% CI | Protocol mix | Centroid signature |
|---|---|---|---|---|---|
| 0 | 46 | 80% | [67%, 89%] | 42 contrast / 4 plain | +1.2 apical LAA-950, +1.1 paren LAA-910 → "severe apical emphysema, PH-enriched" |
| 1 | 96 | 70% | [60%, 78%] | 76 / 20 | −0.7 LAA-910 → "low-LAA" |
| 2 | 19 |  0% | [0%, 17%]  | 0 / 19  | +3.4 mean HU, +3.2 std HU → "high-HU plain-scan outliers" |
| 3 |  2 |  0% | [0%, 66%]  | 1 / 1   | extreme LAA, too small |
| 4 | 89 | 62% | [51%, 71%] | 67 / 22 | +0.6 LAA-856 → "moderate emphysema" |

**Within-contrast PH proportions** (163/189 ≈ 86% baseline): clusters 0/1/4
sit at 82–88% — no cluster separates PH from non-PH within the contrast
subset. Parenchyma-only features encode **emphysema severity**, not PH
status, once protocol is held fixed.

**Scientific reading**: parenchyma phenotype is a *modifier*, not a
*predictor*, of PH in COPD. Clinical intuition is consistent — emphysema
severity is a widespread COPD trait; PH is a specific secondary
phenotype that requires vessel-remodeling evidence. Any disease signal
beyond 0.85 AUC must therefore come from the vessel graph, which
motivates E1 (vessel topology cluster on contrast-only subset) as the
primary next analysis.

## 15. HiPaS-aligned disease-direction sanity check (planned for Round 3)

Chu et al., Nature Communications 2025 ("HiPaS") achieved non-inferior
artery-vein segmentation on non-contrast CT vs CTPA (DSC 89.95% vs 90.24%,
p=0.633 paired), trained on 875 multi-center CTs. On an anatomical study
of n=11,784 participants with lung-volume control, they reported:

- **PAH → negative correlation with artery abundance** (skeleton length
  −192 ± 96 per unit, branch count −43 ± 20).
- **COPD → negative correlation with vein abundance** (SL −170 ± 74,
  BC −42 ± 20).

Actionable implications for our work:

### 15.1 Protocol unification is a solved problem

The root cause of our W1 confound is that the 197 contrast and 85
plain-scan cohorts received different segmentation models. HiPaS
demonstrates a single unified model can do both. Round 3 plan:

- If local retraining is infeasible, re-segment one cohort with the
  other's segmenter as a consistency check. Minimum: run the plain-scan
  segmenter on a sample of contrast cases and compare artery/vein
  skeleton length + branch count to the original segmentation.
- Medium-term: adopt a HiPaS-style unified pipeline for the v3 cache,
  eliminating protocol confound at source. The `v3_mm` builder was
  already planned to fix mm-coords + ct_density; adding a unified
  segmenter makes v3 reviewer-defensible on W1.

### 15.2 Falsifiable disease directions

Extract from our `cache_v2_tri_flat` per-case pkls:

- Artery skeleton length (SL_A) and branch count (BC_A).
- Vein skeleton length (SL_V) and branch count (BC_V).

Then, on the contrast-only 189-case subset (protocol-balanced):

- Predicted by HiPaS (published): PH cases should show **lower SL_A
  and BC_A** than non-PH matched for lung volume.
- Predicted by HiPaS (published): COPD severity (LAA-910) should show
  **lower SL_V and BC_V**.

This is a literature-aligned falsification test. If our v2 graphs
reproduce both directions on the contrast subset, it is strong evidence
the residual signal above protocol is biological. If they do not, the
v2 graph construction is suspect (W3 concerns vindicated).

### 15.3 Benchmark endpoints we should add

HiPaS uses simple abundance metrics that are much more stable than
Strahler-order or tortuosity and that map 1:1 to published literature:

- Artery / vein / airway **skeleton length** per lung volume
- Artery / vein / airway **branch count** per lung volume

These should appear in a supplementary table next to our GCN AUCs to
triangulate the disease signal against a literature-validated metric.

### 15.4 Artifacts for Round 3

- `scripts/evolution/E1_vessel_cluster.py` — vessel-topology cluster on
  contrast-only 189 cases (needs remote cache access).
- `scripts/evolution/E3_abundance_endpoints.py` — per-case SL / BC
  extraction + contrast-only disease-direction test vs HiPaS priors.
- `REPORT_v2.md §15` (this section) will be updated with measured
  SL / BC directions once E3 runs.

## 16. ARIS Round 3 — W8 reproducibility + cache-feature protocol + HiPaS test

### 16.1 E3 HiPaS-aligned disease-direction test (LOCAL PROXY)

Script: `scripts/evolution/E3_abundance_disease_direction.py`.
Proxy: volume fraction (artery_vol_mL / lung_vol_mL). True HiPaS comparison
needs skeleton length + branch count; that requires the remote graph cache.

Results (contrast-only subset + full cohort stratified by protocol):

| Test | Direction | Measured | Matches HiPaS |
|---|---|---|---|
| T1 PH vs nonPH on `artery_frac` (contrast, n=158/27) | ↓artery for PH | **median PH 0.081 > nonPH 0.047**, p=8.4e-6, δ=+0.54 | **NO (opposite)** |
| T2 Spearman(LAA_910, `vein_frac`) contrast (n=185) | negative | ρ = **−0.745**, p<1e-33 | **YES** |
| T2 same plain_scan (n=46) | negative | ρ = **−0.511**, p=3e-4 | **YES** |
| T3 Spearman(LAA_910, `artery_frac`) contrast | weaker than T2 | ρ = −0.649, p<1e-22 | partial (almost as strong as T2) |

**Interpretation of T1 mismatch**: HiPaS predicts PAH → reduced distal
artery *skeleton length* after lung-volume control. Our T1 shows PH has
HIGHER artery *volume*. The classic PH pruning pattern is central PA
dilation + distal branch loss. Volume conflates these: central dilation
alone can increase total artery voxel count even when branches are lost.
A true HiPaS-equivalent test requires replacing volume with skeleton
length — this is the single most important Round 4 endpoint.

T2 (COPD→↓vein abundance) is strongly confirmed even on our crude volume
proxy, matching HiPaS. Both contrast and plain-scan cohorts show the
same direction with consistent ρ magnitude, suggesting the COPD→vein
signal is disease-specific and not a protocol artifact.

### 16.2 Cache-feature protocol decodability (LOCAL PROXY)

Script: `scripts/evolution/R3_cache_feature_protocol.py`.

| Feature set | n_feats | Protocol (LR / GB) | Disease contrast (LR / GB) |
|---|---|---|---|
| A_per_structure_volumes | 4 | **0.524** / 0.910 | 0.756 / 0.660 |
| B_volumes_plus_ratios | 7 | 0.885 / 0.854 | 0.770 / 0.706 |
| C_spatial_only | 4 | 0.732 / 0.767 | 0.671 / 0.402 |
| D_paren_LAA_only | 3 | **0.591** / 0.811 | 0.685 / 0.633 |
| E_v2_ratio_combined_no_HU | 11 | 0.860 / 0.877 | 0.786 / 0.741 |

**Key observation**: volumes alone are **linearly** protocol-invariant
(LR AUC 0.52) but GB still finds a 0.91 non-linear protocol decoder.
Any disease claim on GB features needs an adversarial debiasing step to
avoid re-learning protocol through non-linear interactions.

Linear-classifier protocol AUC is the honest "protocol leakage floor"
for the paper's main endpoint: set D (paren LAA only, 3 features) has
linear protocol AUC 0.591 which is the lowest protocol signal among
disease-preserving sets (contrast-only disease 0.685 LR).

### 16.3 W8 reproducibility manifest

- `environment.yml` (remote training env: Python 3.9, PyTorch 2.2, PyG 2.5,
  kimimaro 4.0.4 placeholder pending `pip show` on remote).
- `requirements-local.txt` (local analysis env: numpy/pandas/sklearn/nibabel/umap).
- `REPRODUCE.md` — single-file instructions covering (a) protocol labels,
  (b) lung v2 extraction, (c) W1/W2/W6 stress tests, (d) E2/E3/R3 Round 3
  analyses, (e) remote cache rebuild, (f) ARIS loop.
- Each v2 pkl has `builder_version=v2_kimimaro`; future rebuilds will
  additionally record git SHA.

### 16.4 What Round 3 still does NOT have

- **Per-case val-fold probs** → case-level DeLong (needs remote rerun of
  sprint6 with prob dump; queued).
- **True skeleton-length abundance** → HiPaS T1 rerun with proper metric
  (needs local skimage skeletonize_3d or remote kimimaro access).
- **TEASAR parameter sensitivity** + overlay QC (needs remote or local
  mask visualization gallery).
- **Domain-adversarial mitigation** (W1 → 0.5 protocol AUC target) —
  new experimental arm.

## 17. ARIS Round 4 — the within-nonPH correction + robustness pack

Round 3 reviewer insight (carried in `REVIEWER_MEMORY.md`): protocol AUC
across the full 282 cohort conflates the label↔protocol coupling. The
honest test restricts to **label=0** (27 contrast nonPH vs 85 plain-scan
nonPH) — protocol AUC there isolates real protocol leakage.

### 17.1 Within-nonPH protocol decoder (R4.1 — CORE W1 CORRECTION)

Script: `scripts/evolution/R4_within_nonph_protocol.py`. 5-fold stratified
CV with class_weight-balanced LR, 95% bootstrap CI (2000 resamples) on the
mean fold AUC.

| Feature set | n | Protocol LR AUC (95% CI) | Protocol GB AUC (95% CI) | R3 full-cohort LR | Δ |
|---|---|---|---|---|---|
| v1_whole_lung_HU | 110 | **0.765 [0.697, 0.833]** | 0.757 [0.666, 0.837] | 1.000 | **−0.24** |
| v2_parenchyma_only | 93 | 0.794 [0.705, 0.886] | 0.731 [0.652, 0.825] | 0.857 | −0.06 |
| v2_paren_LAA_only (3 feats) | 93 | 0.715 [0.646, 0.789] | 0.673 [0.566, 0.762] | 0.591 | +0.12 |
| v2_spatial_paren | 93 | 0.669 [0.543, 0.795] | 0.652 [0.548, 0.748] | 0.732 | −0.06 |
| **v2_per_structure_volumes** | 110 | **0.529 [0.429, 0.631]** | 0.702 [0.615, 0.771] | 0.524 | ≈ 0 |
| v2_vessel_ratios | 85 | 0.674 [0.542, 0.805] | 0.632 [0.441, 0.800] | 0.885 | −0.21 |
| v2_combined_no_HU | 73 | 0.731 [0.653, 0.810] | 0.664 [0.590, 0.737] | 0.860 | −0.13 |

**Critical findings**:

1. The v1 whole-lung "perfect protocol decoder" (AUC=1.000 across full
   cohort) drops to **0.765 within-nonPH** — meaning ~75% of the R2/R3
   "protocol leakage" alarm was actually label-leakage. The v1 features
   do carry protocol signal (0.77 > random 0.5) but it is far from
   trivially decodable once label is held fixed.
2. `v2_per_structure_volumes` (artery/vein/airway/vessel_airway_over_lung)
   has LR protocol AUC 0.529 with **95% CI [0.429, 0.631] — straddles
   random**. This is the first feature set to achieve true within-nonPH
   protocol invariance under linear decoding. GB recovers a non-linear
   0.70 protocol signal, but for the GCN's typical linear-combination
   layers, this is a defensible protocol-robust representation.
3. `v2_parenchyma_only` still has within-nonPH LR 0.79 — suggesting
   that parenchyma HU shifts between contrast and plain-scan are real
   (possibly capillary-level contrast bleed), not merely label
   confounding. Round 5 should compute these on `ct.nii.gz` intensity
   after normalizing each scan to air (−1000) and blood-pool (+50)
   calibration points.

### 17.2 Exclusion sensitivity (R4.4)

Script: `scripts/evolution/R4_exclusion_sensitivity.py`. Rebuild the
disease classifier on two cohorts:

- **A** — current behavior, 27 vessel-placeholder nonPH dropped.
- **B** — same cases retained; parenchyma HU/LAA are computed with no
  vessel subtraction (equals whole-lung for those 27 cases). Class
  balance shifts from 163 PH / 69 nonPH → 163 PH / 89 nonPH.

| Cohort | Feat set | n | disease LR full (CI) | disease LR contrast (CI) |
|---|---|---|---|---|
| A_excluded | paren_only | 231 | 0.862 [0.835, 0.886] | 0.858 [0.784, 0.932] |
| A_excluded | paren_plus_spatial | 231 | 0.871 [0.854, 0.888] | 0.851 [0.770, 0.932] |
| B_included | paren_only | 252 | 0.870 [0.826, 0.906] | 0.860 [0.763, 0.950] |
| B_included | paren_plus_spatial | 252 | 0.879 [0.827, 0.913] | 0.855 [0.759, 0.951] |

**Max |Δ disease contrast AUC| = 0.004** → far inside the ±0.05 bootstrap
CI half-width. The disease claim is robust to the exclusion rule.

### 17.3 Overlay gallery (R4.3)

Script: `scripts/evolution/R4_overlay_gallery.py`. 10 random cases
(5 PH + 5 nonPH, balanced across protocols) — axial mid-slice with
vessel overlay, axial skeleton overlay (skimage.skeletonize_3d), and
coronal vessel MIP. Saved as `outputs/evolution/R4_overlay_gallery.png`
(10×3 = 30 subplots). First-pass anatomical QC; not yet a blinded
radiologist review but enables a reviewer to visually confirm mask
topology matches published pulmonary anatomy.

Representative case IDs (`outputs/evolution/R4_overlay_gallery_cases.txt`):
ph_wenhaibo, ph_luwanhai, ph_lixiangzhen, nonph_lujianlan,
nonph_liuzhusheng, nonph_guyewu, nonph_zhuhuiting, nonph_wanggenyou,
ph_liuzhiquan, ph_gaozhongxiang.

### 17.4 Reproducibility hardening (R4.5)

- **`requirements-local.lock.txt`** — exact `pip freeze` output for the
  local analysis Python (numpy 2.4.4, pandas 3.0.2, sklearn 1.8.0,
  nibabel 5.4.2, scipy 1.17.1, matplotlib 3.10.8, scikit-image 0.26.0,
  umap-learn 0.5.12).
- **`scripts/cache_provenance.py`** — inspects a cache pkl and prints
  `builder_version`, `git_sha`, `kimimaro_version`, and TEASAR params
  when present.
- **`REPRODUCE.md`** — updated to note that remote kimimaro version is
  still TBD pending `pip show kimimaro` inside `pulmonary_bv5_py39`.
- The next remote rebuild should record `git rev-parse HEAD` and
  `kimimaro.__version__` in each pkl for full provenance.

### 17.5 Skeleton-length HiPaS test (R4.2) — PENDING

`scripts/evolution/R4_skeleton_length.py` is running locally against
all 282 cases (`skimage.morphology.skeletonize_3d`). ~35 min runtime
with 4 workers. On completion we rerun T1 (PAH → ↓ artery skeleton
length) and T2 (COPD → ↓ vein skeleton length) directly against HiPaS
priors. Results will land in `outputs/evolution/R4_skeleton_directions.md`.

### 17.6 Still outstanding for Round 5

- **Case-level DeLong** — still requires a remote rerun of sprint6
  writing per-case val probs (not blocked by anything but SSH access).
- **TEASAR parameter sensitivity sweep** — kimimaro scale ∈ {0.8, 1.0, 1.5},
  const ∈ {2, 5, 10}, rebuild a subset of 20 cases and check
  per-case morphometric stability.
- **Domain-adversarial GCN** — a new arm that adds a gradient-reversal
  head on `is_contrast` during training; protocol-invariance target
  AUC ≤ 0.6 within-nonPH (R4.1 already shows volumes alone hit 0.53).
- **Locked kimimaro version** — single SSH session to populate the pin.
- **Blinded airway QC** — placeholder cases and non-placeholder
  low-component airways need radiologist inspection before any airway
  inclusion in main claims.

## 18. ARIS Round 5 — case-level DeLong CIs (PARTIAL — paired pending)

**Pipeline**: patched `run_sprint6_v2_probs.py` on remote dumps per-case
val-fold probabilities (`ensemble_y_true`, `ensemble_y_score`); these are
fetched via `scp` and consumed by `scripts/evolution/R5_delong.py`.

**Result on contrast-only 189-case subset** (gcn_only mode, ensembled
across 3 repeats × 5 folds):

| Arm | n | AUC | DeLong 95% CI |
|---|---|---|---|
| arm_b (vessel-only, radiomics-filtered subset) | 92 | 0.8462 | [0.7340, 0.9584] |
| **arm_c (vessel + 13 lung globals)** | **189** | **0.8391** | **[0.7527, 0.9255]** |

**Headline**: arm_c contrast-only AUC = **0.8391 [0.7527, 0.9255]** —
DeLong 95% CI excludes 0.50, so disease signal on the protocol-balanced
subset is significant under case-level inference. This is the W6
case-level confirmatory result the Round 4 reviewer required.

**Caveat — paired DeLong unavailable in Round 5**: arm_b's training
dataset is restricted to 92 cases by the radiomics-feature requirement
(`require_radiomics=True` in dataset construction); arm_c uses the
unfiltered 189. The two arms are not on a matched case set so a paired
DeLong on `arm_c − arm_b` is not meaningful. Round 6 will rebuild arm_b
with the radiomics filter disabled to enable the paired comparison.

**Unpaired approximation** (for completeness): Δ = −0.0071, z = −0.10,
p = 0.92 — adding lung features yields no measurable benefit over the
vessel-only baseline once protocol is balanced. Consistent with §13.5
retraction of the lung-feature contribution claim.

### Round 5 status

- ✅ R5.1 case-level DeLong CIs on arm_b and arm_c (single-arm)
- ✅ R5.2 GCN-feature within-nonPH protocol classifier — see §18.2
- ⏳ R5.3 paired DeLong arm_c − arm_b (needs arm_b rebuild in Round 6)

### 18.2 R5.2 — Protocol decoder on EXACT GCN inputs WITHIN nonPH (HONEST W1)

Per-case graph statistics (47 features per case: `n_nodes`, `mean_degree`,
`x0_mean`…`x12_p90`, edge-attr aggregates) extracted from
`cache_v2_tri_flat/*.pkl` via remote `extract_graph_stats.py`. Local
sklearn LR/GB classifiers, 5-fold stratified CV, bootstrap CI.

| Subset | n | Endpoint | LR AUC (95% CI) | GB AUC (95% CI) |
|---|---|---|---|---|
| **within-nonPH** | 80 (26c + 54p) | protocol | **0.853 [0.722, 0.942]** | **0.774 [0.596, 0.908]** |
| within-contrast | 189 | disease | 0.858 [0.789, 0.923] | 0.782 [0.715, 0.849] |
| full cohort | 243 | protocol | 0.936 | 0.929 |

**Reading** — this is the W1 endpoint the Round-4 reviewer specifically
demanded (protocol AUC on the actual GCN inputs, within label=0):

1. Protocol AUC LR **0.853** (95% CI excludes 0.7 below) — GCN inputs DO
   carry significant protocol signal even within nonPH. The R4.1 finding
   that v2 per-structure volumes (4 lung-feature scalars) had LR
   protocol AUC 0.529 within-nonPH does NOT generalize: the richer
   47-feature graph aggregate (which includes per-node feature
   distributions) recovers protocol decodability.
2. The same features deliver disease AUC 0.858 within contrast — disease
   signal is real and comparable in magnitude to the protocol signal.
3. The two signals are entangled but not identical (protocol within-nonPH
   0.85 vs disease within-contrast 0.86). A domain-adversarial GCN that
   penalizes protocol decodability while preserving disease decodability
   is the principled mitigation, scheduled as Round 6's primary new arm.

**Implication for v2 cache validity**: the cache itself is not "broken"
(disease signal is preserved) but it does carry segmentation-quality
artifacts that correlate with protocol. The path to a Nature-Medicine-
defensible result is (a) adversarial debiasing of the GCN training
objective, (b) re-segmentation with HiPaS-style unified pipeline (§15),
or (c) acceptance of the residual confound with sensitivity bounds.

The autonomous loop (`review-stage/AUTONOMOUS_LOOP_PLAN.md` + cron
`fda1fcf8` every 2 hours) will execute R5.2/R5.3 + Round 6 (TEASAR
sensitivity sweep + adversarial debiasing) without user intervention.

### 14.5 Cohort protocol labels committed (NEW)

`data/case_protocol.csv` (282 rows) is derived from the three original DCM
folders on `H:/官方数据data/`:

- `COPDPH_seg/` → 170 contrast PH,
- `COPDnonPH_seg/` → 27 contrast nonPH,
- `New folder-COPD（PH概率小）/` → 85 plain-scan nonPH.

Cross-tab (label × protocol):

| | contrast | plain-scan |
|---|---|---|
| label=0 (COPD) | 27 | 85 |
| label=1 (COPD-PH) | 170 | 0 |

Zero cases unmatched. This is the authoritative protocol label used by
§13/§14 analyses; any future protocol-stratified experiment must load
`case_protocol.csv` rather than re-infer from directory names.

### 14.6 Open items heading into Round 3

- **W3**: TEASAR parameter sensitivity (requires remote rebuild, not local).
- **W5**: exclusion-sensitivity rerun keeping 27 placeholder nonPH with
  degraded graphs.
- **W6 upgrade**: per-case val-prob dump + proper DeLong paired test.
- **W7**: airway QC (largest-component fraction, HU-sanity, placeholder-shape).
- **W8**: publish `environment.yml`, pin kimimaro version, add one-command
  rebuild script.
- **Disease-focused clustering** (user-facing scientific question): repeat
  §9 phenotype clustering on **parenchyma-only** + **vessel-graph stats**,
  stratified by protocol, to separate emphysema severity from vascular
  remodelling signatures during COPD → COPD-PH evolution.
