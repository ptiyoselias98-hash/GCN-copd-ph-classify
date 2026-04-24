# R5.2 — Protocol decoder on EXACT GCN-input features (cache_v2_tri_flat)

Per-case graph aggregates from `cache_v2_tri_flat/*.pkl` (47 features:
n_nodes, mean_degree, x{0..12}_mean/std/p90, e{0..2}_mean/p90).

## Within-nonPH protocol AUC (HONEST W1 ENDPOINT)

- n = 80 (cases with valid graph stats AND label=0)
- LR protocol AUC: **0.853** (95% CI [0.722, 0.942])
- GB protocol AUC: **0.774** (95% CI [0.596, 0.908])

**Reading**: this is the actual W1 endpoint per Round-4 reviewer memory.
If LR CI upper bound < 0.7 → the GCN's input features are protocol-robust under linear decoding.
If GB CI > 0.7 → non-linear protocol signal exists; an adversarial debiasing arm is needed.

## Within-contrast disease AUC (positive control)

- n = 189
- LR disease AUC: 0.858 [0.789, 0.923]
- GB disease AUC: 0.782 [0.715, 0.849]

## Full-cohort protocol AUC (label-shortcut sanity check)

- n = 243
- LR protocol AUC (full): 0.936 (expected high, as label↔protocol coupled)
- GB protocol AUC (full): 0.929
