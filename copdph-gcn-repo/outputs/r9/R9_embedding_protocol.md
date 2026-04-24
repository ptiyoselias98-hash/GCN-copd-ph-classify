# R9 — Within-nonPH protocol decoding on GCN EMBEDDINGS

Embedding source: `outputs/r9/embeddings_full/emb_gcn_only_rep1_fold{1..5}.npz`
(penultimate z_proj from the trained arm_a attention-pooling head,
full 282-case training run with --dump_embeddings; val embeddings
span contrast + plain-scan cases across all 5 folds).

Total val embeddings: 243 (shape=(243, 64))
Protocol split in val set: 189 contrast / 54 plain-scan

## Primary test: protocol AUC on embeddings WITHIN nonPH only

- n = 80 (contrast nonPH = 26)
- LR protocol AUC on embeddings: **0.834** (95% CI [0.738, 0.901])
- GB protocol AUC on embeddings: **0.703** (95% CI [0.595, 0.822])

## Comparison with R5.2 (graph-aggregate INPUTS)

| Representation | within-nonPH protocol LR AUC | 95% CI |
|---|---|---|
| GCN INPUT aggregates (47-dim) | 0.853 | [0.722, 0.942] |
| **GCN EMBEDDINGS (z_proj)** | **0.834** | [0.738, 0.901] |

Δ (embedding − input) = **-0.019**. GCN embeddings carry about the same protocol signal as inputs (which is a good positive result).

## Sanity check: disease AUC on embeddings within contrast-only

- n = 189
- LR disease AUC: 0.741 [0.608, 0.869]

The embedding preserves disease signal (comparable to the ~0.84 AUC from
the paired DeLong primary endpoint), which is expected and serves as a
positive control that embeddings are non-degenerate.