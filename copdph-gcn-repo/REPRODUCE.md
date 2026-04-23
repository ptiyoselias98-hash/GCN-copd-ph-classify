# Reproduce the Round 2 / Round 3 analyses

This repo reproduces the ARIS-reviewed analyses on the 282-case COPD-PH cohort.
The full training pipeline (GCN on `cache_v2_tri_flat`) runs on a remote GPU;
the confound analyses + lung-phenotype v2 + cluster experiments run locally.

## 0. Assumptions on raw data paths (Windows machine)

- `E:/桌面文件/nii格式图/nii/<case>/{airway,artery,ct,lung,vein}.nii.gz` —
  197 contrast-enhanced cases (binary 0/1 masks).
- `E:/桌面文件/nii格式图/nii-unified-282/<case>/` — all 282 cases; plain-scan
  cohort has actual files here, contrast cohort has a `_source.txt` pointer.
- `H:/官方数据data/{COPDPH_seg, COPDnonPH_seg, New folder-COPD（PH概率小）}/` —
  original DCM folders used to derive `data/case_protocol.csv`.

## 1. Local environment

```
pip install -r requirements-local.txt
```

Works on Windows Python 3.14 or Linux Python ≥ 3.10.

## 2. Derive protocol labels

```
python _derive_protocol_labels.py
```

Produces `data/case_protocol.csv` (282 rows, zero unmatched) by string-matching
on the H: drive DCM folder names. Cross-tab: 170 PH contrast / 27 nonPH contrast / 85 nonPH plain-scan.

## 3. Extract lung phenotype v2

```
python _extract_lung_v2.py
```

Reads `nii-unified-282/<case>/lung.nii.gz` (auto-detects HU-sentinel vs
binary mask conventions; follows `_source.txt` redirects for contrast cases);
produces `outputs/lung_features_v2.csv` with 51 columns including
parenchyma-only HU/LAA, apical/middle/basal bands, per-structure volumes.
4-worker CPU job, ~7 min on a 16-core Windows box with 32 GB RAM.

## 4. W1 stress-test (scalar lung features)

```
python _w1_protocol_classifier.py        # v1 whole-lung features, target AUC 1.000 for protocol
python _w1_protocol_classifier_v2.py     # v2 comparison, paren_only hits protocol ~0.86
```

## 5. W2 patient-level fold audit

```
python _audit_patient_leakage.py
```

Reports `outputs/_patient_leakage_audit.{md,json}` — 282 cases = 282 unique patients.

## 6. W6 fold-level paired statistical comparisons

```
python _compute_ci_fold_level.py
```

Bootstrap CI + paired Wilcoxon on the 15 fold AUCs from
`outputs/sprint6_arm_{b,c}_{triflat,quad,contrast_only}_v2/sprint6_results.json`.

## 7. Round 3 — E2 parenchyma cluster, E3 abundance directions, cache-feature protocol

```
python scripts/evolution/E2_parenchyma_cluster.py
python scripts/evolution/E3_abundance_disease_direction.py
python scripts/evolution/R3_cache_feature_protocol.py
```

## 8. Remote: training + cache build

The full tri-structure GCN runs on the IMSS remote box (`imss@10.60.147.117`).
Launcher scripts under `_remote_*.py` expect the remote repo to live at
`/home/imss/cw/GCN-copd-ph-classify/` with environment
`pulmonary_bv5_py39` activated. To rebuild the v2 cache:

```
ssh imss@10.60.147.117
cd /home/imss/cw/GCN-copd-ph-classify
conda activate pulmonary_bv5_py39
python _remote_build_v2_cache.py    # kimimaro TEASAR skeletonization; 282 cases
python _remote_launch_arm_bc_v2.py  # arm_b + arm_c 5-fold × 3 repeat training
```

Pinned versions (as of 2026-04-23): see `environment.yml`. kimimaro version
from `pulmonary_bv5_py39` env expected to be 4.0.4; verify with
`pip show kimimaro` inside the env and commit the exact version back here.

## 9. ARIS review loop

```
# Context payload for current round
review-stage/context_round{N}.md

# Run codex review (requires MCP codex server)
# See the Round 2 invocation in the conversation log; in summary:
#   codex --model gpt-5.2 --approval-policy never --sandbox read-only \
#         --cwd copdph-gcn-repo review-stage/context_round{N}.md

# Results persist to review-stage/REVIEW_STATE.json + AUTO_REVIEW.md
```

## 10. Cache-builder provenance

Latest v2 cache was built at commit `61c35b3` (2026-04-22) with
`_remote_build_v2_cache.py` after the `(arr != -2048)` mask-extraction
patch. Each pkl stores a `builder_version` tag and the next rebuild will
add `git_sha` + `kimimaro_version`. Inspect with:

```
python scripts/cache_provenance.py cache_v2_tri_flat/<case>.pkl
```

- `requirements-local.lock.txt` — exact local analysis-env versions
  (pip freeze on the Windows Python 3.14 analysis machine, 2026-04-23).
- `environment.yml` — remote training-env spec; kimimaro version is a
  placeholder (pinned as 4.0.4) to be **replaced with the exact output of
  `pip show kimimaro` from the `pulmonary_bv5_py39` env** on next remote
  SSH session.
- The next remote build should `git rev-parse HEAD` and include the SHA
  plus `kimimaro.__version__` in the pkl. `_remote_build_v2_cache.py`
  needs a one-liner patch to do this (tracked as a Round 4 TODO).
