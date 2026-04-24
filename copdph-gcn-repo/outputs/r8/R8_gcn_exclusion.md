# R8.1 — GCN-feature exclusion sensitivity (full 282 with degraded graphs)

47-dim graph-aggregate features (n_nodes/mean_degree/node-HU percentiles/edge-attr)
from cache_v2_tri_flat. Cohort B adds 39 missing cases with all-zero (degraded) features.

| Cohort | n | n_contrast | disease LR full (CI) | disease LR contrast (CI) | protocol within-nonPH LR (CI) |
|---|---|---|---|---|---|
| `A_in_cache_243` | 243 | 189 | 0.889 [0.858, 0.917] | 0.777 [0.705, 0.851] | 0.801 [0.742, 0.869] |
| `B_full_282_degraded` | 282 | 197 | 0.879 [0.820, 0.931] | 0.790 [0.738, 0.852] | 0.856 [0.768, 0.951] |

## Delta (B − A)

- Δ disease AUC (full cohort): **-0.011**
- Δ disease AUC (contrast-only): **+0.013**
- Δ protocol AUC (within-nonPH): **+0.055**

## Reading

- If |Δ| on disease-contrast is smaller than the bootstrap CI half-width
  (typically ~0.04 at n=189), the disease claim is robust to the placeholder
  exclusion choice at the feature-classifier level.
- The GCN itself cannot be retrained this round (both GPUs busy). A full
  GCN-training exclusion sensitivity is queued for Round 9 once GPU frees.

**Caveat**: this is a classifier-on-graph-aggregates proxy for the GCN. A
true GCN-retraining sensitivity analysis may show larger or smaller deltas.