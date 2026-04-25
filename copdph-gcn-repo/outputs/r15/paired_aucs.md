# R15.A — Paired AUC-difference CIs

Addresses R14 reviewer flag: 'lung_only > graph_only CIs overlap; reversal needs paired AUC-diff CI'
and 'CORAL λ=1 vs GRL not yet paired on identical cases'.

## Lung vs Graph (within-contrast disease classifier)

n=184. Paired bootstrap (5000 iters) of
AUC differences on case-level OOF predictions (same case set, same y).

| comparison | AUC_a | AUC_b | Δ (a−b) | 95% CI | approx p |
|---|---|---|---|---|---|
| lung − graph | 0.844 | 0.782 | +0.062 | [-0.031, +0.160] | 0.189 |
| (lung+graph) − graph | 0.867 | 0.782 | +0.085 | [+0.029, +0.148] | 0.0008 |
| (lung+graph) − lung | 0.867 | 0.844 | +0.023 | [-0.042, +0.088] | 0.501 |

## CORAL λ=1 vs corrected-GRL λ=10 (within-nonPH protocol probe)

Paired on the intersection of common case_ids in both embedding sets,
restricted to nonPH and excluding seg-failures.

| seed | n | AUC CORAL | AUC GRL | Δ (CORAL − GRL) | 95% CI | approx p |
|---|---|---|---|---|---|---|
| 42 | 68 | 0.797 | 0.727 | +0.068 | [-0.061, +0.197] | 0.31 |
| 1042 | 68 | 0.739 | 0.754 | -0.015 | [-0.166, +0.136] | 0.85 |
| 2042 | 68 | 0.620 | 0.815 | -0.195 | [-0.342, -0.058] | 0.0072 |

## Interpretation

- For lung vs graph: if `lung − graph` 95% CI excludes 0, the reversal
  is statistically supported; otherwise the marginal CIs were misleading.
- For CORAL vs GRL: if `CORAL − GRL` is significantly negative (lower
  protocol AUC = better deconfounder), CORAL is confirmed as a real
  improvement on the same cases.
