# P-α : cluster × clinical cross-table  (2026-04-21)

**Matched** 105/106 cluster rows to clinical via pinyin(name).
**Numeric variables tested**: 214
**Categorical variables tested**: 147

## Clustering methods ranked by their best-separated variable

| cluster_method | min_pvalue |
|---|---|
| `spectral_k4` | 2.740e-10 |
| `kmeans_k4` | 4.307e-10 |
| `kmeans_k5` | 2.341e-09 |
| `gmm_k5` | 4.346e-08 |
| `spectral_k3` | 2.636e-06 |
| `kmeans_k3` | 4.717e-05 |
| `spectral_k2` | 1.013e-04 |
| `gmm_k2` | 1.596e-04 |

## Top 30 drivers of `spectral_k4`

| rank | variable | type | test | pvalue | n |
|---|---|---|---|---|---|
| 1 | `右肺平均密度(HU)_y` | numeric | kruskal | 2.740e-10 | 95 |
| 2 | `左右肺平均密度(HU)` | numeric | kruskal | 1.367e-09 | 95 |
| 3 | `左肺平均密度(HU)_y` | numeric | kruskal | 5.609e-08 | 95 |
| 4 | `右肺LAA910(ml)_y` | numeric | kruskal | 8.150e-08 | 95 |
| 5 | `右肺LAA910(%)_y` | numeric | kruskal | 1.067e-07 | 95 |
| 6 | `左右肺LAA910(ml)` | numeric | kruskal | 2.184e-07 | 95 |
| 7 | `左右肺LAA910(%)` | numeric | kruskal | 2.444e-07 | 95 |
| 8 | `右肺LAA950(ml)_y` | numeric | kruskal | 4.299e-07 | 95 |
| 9 | `右肺LAA950(%)_y` | numeric | kruskal | 8.911e-07 | 95 |
| 10 | `静脉(右上肺叶)最小密度(HU)` | numeric | kruskal | 1.089e-06 | 95 |
| 11 | `左右肺空气体积(ml)` | numeric | kruskal | 1.304e-06 | 95 |
| 12 | `左右肺LAA950(ml)` | numeric | kruskal | 1.432e-06 | 95 |
| 13 | `右肺空气体积(ml)_y` | numeric | kruskal | 1.471e-06 | 95 |
| 14 | `肺血管BV5(ml)_y` | numeric | kruskal | 3.640e-06 | 95 |
| 15 | `左右肺LAA950(%)` | numeric | kruskal | 4.751e-06 | 95 |
| 16 | `左肺LAA910(%)_y` | numeric | kruskal | 6.629e-06 | 95 |
| 17 | `右肺容积(ml)_y` | numeric | kruskal | 7.052e-06 | 95 |
| 18 | `左右肺容积(ml)` | numeric | kruskal | 7.115e-06 | 95 |
| 19 | `静脉BV5(ml)_y` | numeric | kruskal | 7.856e-06 | 95 |
| 20 | `左肺LAA910(ml)_y` | numeric | kruskal | 8.285e-06 | 95 |
| 21 | `动脉BV5(ml)_y` | numeric | kruskal | 9.625e-06 | 95 |
| 22 | `左右肺密度标准差(HU)` | numeric | kruskal | 1.008e-05 | 95 |
| 23 | `右肺密度标准差(HU)_y` | numeric | kruskal | 1.262e-05 | 95 |
| 24 | `左肺密度标准差(HU)_y` | numeric | kruskal | 1.593e-05 | 95 |
| 25 | `动脉(左上肺叶)最小密度(HU)` | numeric | kruskal | 1.957e-05 | 95 |
| 26 | `动脉(右中肺叶)最小密度(HU)` | numeric | kruskal | 2.127e-05 | 95 |
| 27 | `左右肺支气管长度(cm).1` | numeric | kruskal | 3.484e-05 | 95 |
| 28 | `检查日期_y` | numeric | kruskal | 4.105e-05 | 105 |
| 29 | `左肺空气体积(ml)_y` | numeric | kruskal | 4.723e-05 | 95 |
| 30 | `左肺LAA950(ml)_y` | numeric | kruskal | 6.222e-05 | 95 |

## Significance summary
- Uncorrected alpha=0.05: variables passing per method
  - `spectral_k4`: 127 uncorrected, 36 Bonferroni (α=1.385e-04)
  - `kmeans_k4`: 57 uncorrected, 4 Bonferroni (α=1.385e-04)
  - `kmeans_k5`: 63 uncorrected, 2 Bonferroni (α=1.385e-04)
  - `gmm_k5`: 42 uncorrected, 1 Bonferroni (α=1.385e-04)
  - `spectral_k3`: 94 uncorrected, 6 Bonferroni (α=1.385e-04)
  - `kmeans_k3`: 61 uncorrected, 4 Bonferroni (α=1.385e-04)

## Sanity check — PH label vs best clustering

### `spectral_k4`

```
spectral_k4   0   1   2   3  total
label                             
0            10   1   9   7     27
1            15  11  41  11     78
total        25  12  50  18    105
```

### `kmeans_k4`

```
kmeans_k4   0  1  2   3  total
label                         
0          22  0  1   4     27
1          55  3  0  20     78
total      77  3  1  24    105
```

### `kmeans_k5`

```
kmeans_k5   0   1  2   3  4  total
label                             
0          12  13  0   1  1     27
1          28  37  3  10  0     78
total      40  50  3  11  1    105
```
