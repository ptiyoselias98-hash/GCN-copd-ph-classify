# R11 — Fixed-GRL multi-seed sweep

Round 10 reviewer flagged: λ double-scaling bug (grads ~λ²) + adversary
trained on full cohort (PH≈contrast). Round 11 fixes: GRL coef=1.0 inside
the layer, only `λ * adv_loss` on the loss; adversary trained on **nonPH-only**
samples per batch.

Multi-seed: 3 seeds × 4 λ values = 12 runs total.

Aggregated (mean ± SD across seeds):

| λ | n_seeds | Protocol LR | Protocol MLP | Disease LR (contrast) |
|---|---|---|---|---|
| 0.0 | 3 | 0.848 ± 0.032 | 0.847 ± 0.049 | 0.731 ± 0.007 |
| 1.0 | 3 | 0.863 ± 0.029 | 0.885 ± 0.015 | 0.684 ± 0.049 |
| 5.0 | 3 | 0.823 ± 0.018 | 0.843 ± 0.043 | 0.657 ± 0.068 |
| 10.0 | 3 | 0.802 ± 0.055 | 0.881 ± 0.032 | 0.644 ± 0.023 |

**Best λ** (lowest protocol LR mean): λ=10.0 — protocol_lr=0.802, disease_lr=0.644.
Target was protocol_lr ≤ 0.60 with upper-CI ≤ 0.65: ❌ NOT MET