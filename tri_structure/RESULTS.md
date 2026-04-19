# Tri-structure GCN v2 — Results (2026-04-19)

Three-encoder architecture (artery / vein / airway) with cross-structure
attention. v2 extends Phase 1 by adding **per-structure attention pooling**,
**24-D graph signatures** concatenated into the fusion head, and
**hybrid clustering** over three views (embedding / signature / hybrid).

```
artery -> skeleton -> Graph_A -> GCN_A + AttnPool -> z_A --+
                                                            +-- CrossAttn -> z_fused -> [MLP + signatures] -> logits
vein   -> skeleton -> Graph_V -> GCN_V + AttnPool -> z_V --+
airway -> skeleton -> Graph_W -> GCN_W + AttnPool -> z_W --+
```

- Cache: existing unified cache + heuristic Strahler/diameter partition (Phase 1 setup).
- n=106 gold-labelled patients.
- CV: 3 repeats x 5 folds = 15 folds, per-fold Youden threshold, mPAP auxiliary loss.
- Hardware: RTX 3090, 200 epochs, ~25 min.

## Headline metrics (15-fold CV, mean ± std)

| Metric   | **v2**           | Phase 1 (baseline) | mode_radiomics | v2 vs Phase 1 |
|----------|------------------|--------------------|----------------|---------------|
| AUC      | **0.734 ± 0.142** | 0.880 ± 0.093     | 0.885          | **−0.146**    |
| Sens     | **0.708 ± 0.144** | 0.869 ± 0.113     | 0.895          | **−0.161**    |
| Spec     | **0.824 ± 0.166** | 0.900 ± 0.119     | 0.862          | **−0.076**    |
| F1       | **0.794 ± 0.101** | 0.910 ± 0.070     | 0.919          | **−0.116**    |
| Precision| 0.926 ± 0.064    | 0.964 ± 0.041     | —              | −0.038        |
| Accuracy | 0.739 ± 0.115    | 0.877 ± 0.089     | —              | −0.138        |

**v2 is a regression across every classification metric**, with std ≈ 1.5× Phase 1.
Fold-level AUCs include 0.39, 0.56, 0.59, 0.62 — training is unstable.

## Interpretability (attention on 3 structures)

| Group           | artery | vein  | airway |
|-----------------|--------|-------|--------|
| COPD (non-PH)   | 0.563  | 0.238 | 0.200  |
| COPD-PH         | 0.656  | 0.211 | 0.133  |

| Correlation with mPAP | Phase 1 | **v2** |
|-----------------------|---------|--------|
| artery × mPAP  r      | +0.486  | **+0.074** |
| airway × mPAP  r      | −0.468  | **−0.125** |

The strong biologically-meaningful signal Phase 1 produced (artery dominance
scaling with pulmonary arterial pressure, airway dominance inversely scaling)
is **lost in v2**. Attention pooling + signature concat appears to disturb the
mechanism that made the per-structure attention interpretable.

## Unsupervised clustering — three views

All three views (embedding / signature / hybrid) produce cluster assignments
that are **not aligned with the PH label** (ARI ≈ 0 across k=2..4 for both
KMeans and GMM). This reproduces the negative finding from `project_followup_automation`.

Notable: the **signature view** yields tighter geometric clusters (Silhouette
≈ 0.30) than embedding/hybrid (≈ 0.10), but the clusters are PH-orthogonal.

## Likely causes of the regression

1. `torch_geometric.nn.glob.GlobalAttention` is **deprecated**
   (replaced by `nn.aggr.AttentionalAggregation`); numerical stability appears
   worse than the mean pooling used in Phase 1.
2. Concatenating 24-D raw graph signatures into the MLP may swamp the 64-D
   learned embedding on n=106 samples.
3. Per-structure attention-pool params add capacity that is not supported by
   the small training set (over-fitting in the attention weights themselves).

## Recommendation

**Do not adopt v2.** Phase 1 remains the best tri-structure configuration.
If signatures are worth keeping, explore them as an **auxiliary multi-task
head** (predict signatures from z_fused as a regularizer) rather than as
direct concat inputs to the classifier.

## Files

- `src/` — v2 model + pipeline source (models.py has AttnPool; tri_structure_pipeline.py emits signatures + hybrid clustering)
- `automation/auto_run_tri.py` — paramiko orchestrator (`launch_v2` subcommand)
- `automation/_run_tri_pipeline_v2.sh` — remote v2 launcher
- `outputs/phase2_v2/cv_results.json` — per-fold metrics + mean
- `outputs/phase2_v2/cluster_analysis.json` — clustering on 3 views + attention profiles

`shared_embeddings.npz` is **intentionally not included** (contains patient
identifiers).
