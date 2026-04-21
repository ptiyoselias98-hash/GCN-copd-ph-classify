# Plan A — Unsupervised clustering on n=269 tri_structure embeddings

Source: `p_theta_269_lr2x` shared_embeddings (64-D, 269 cases, PH=164 / non-PH=105). mPAP continuous value unavailable for expanded cohort — analysis uses binary PH label only.

## 1. Cluster quality sweep (k ∈ {2..6} × {kmeans, GMM, spectral})

| method | k | silhouette | ARI vs PH | NMI vs PH | sizes |
|---|---|---|---|---|---|
| spectral | 2 | 0.141 | **0.611** | 0.542 | [189, 80] |
| kmeans | 2 | 0.136 | **0.600** | 0.512 | [186, 83] |
| gmm | 2 | 0.140 | **0.588** | 0.511 | [80, 189] |
| spectral | 3 | 0.067 | **0.446** | 0.428 | [146, 42, 81] |
| gmm | 3 | 0.084 | **0.428** | 0.392 | [37, 148, 84] |
| spectral | 4 | 0.092 | **0.348** | 0.362 | [41, 111, 81, 36] |
| kmeans | 3 | 0.086 | **0.334** | 0.340 | [72, 74, 123] |
| gmm | 4 | 0.084 | **0.287** | 0.345 | [44, 101, 90, 34] |
| spectral | 6 | 0.107 | **0.204** | 0.285 | [36, 82, 38, 34, 30, 49] |
| kmeans | 4 | 0.106 | **0.193** | 0.233 | [79, 48, 76, 66] |
| kmeans | 6 | 0.142 | **0.182** | 0.277 | [31, 39, 40, 34, 79, 46] |
| gmm | 6 | 0.102 | **0.177** | 0.265 | [48, 70, 40, 37, 38, 36] |
| spectral | 5 | 0.090 | **0.173** | 0.229 | [70, 58, 71, 31, 39] |
| kmeans | 5 | 0.120 | **0.171** | 0.227 | [38, 64, 74, 39, 54] |
| gmm | 5 | 0.082 | **0.165** | 0.204 | [49, 32, 47, 95, 46] |

Best alignment: **spectral k=2, ARI=0.611** (n=106 earlier runs: ARI ≈ 0 — cohort size crosses a separation threshold).

![2D projection](plan_a/projection_2d.png)

## 2. Per-cluster clinical profile

| cluster | n | PH rate | artery attn | vein attn | airway attn |
|---|---|---|---|---|---|
| c0 | 189 | 0.86 | 0.37 | 0.42 | 0.20 |
| c1 | 80 | 0.03 | 0.45 | 0.33 | 0.22 |

![PH-rate per cluster](plan_a/cluster_ph_profile.png)

## 3. Attention flip — artery vs vein between PH and non-PH

Non-PH (n=105): artery=0.43, vein=0.35, airway=0.22  —  artery dominant.

PH (n=164): artery=0.38, vein=0.42, airway=0.20  —  **vein dominant** (flips above artery).

![Attention flip](plan_a/attention_flip.png)

This flip is the first **mechanistic** signal the tri_structure model has produced: the cross-structure attention places more weight on vein geometry when predicting PH. It does not by itself establish causation (attention is still a classifier read-out), but it justifies investigating post-capillary / pulmonary-venous patterns as part of the PH differential in this cohort.