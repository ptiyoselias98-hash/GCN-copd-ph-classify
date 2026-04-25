# R12.4 — Cross-seed protocol-LR aggregate CIs (within-nonPH)

Pools per-seed OOF predicted probabilities across 3 seeds, then
bootstraps over cases (and over seeds, hierarchically). Compare to
the per-run CIs reported in `R11_grlfix_summary.json`.

| λ | n_cases | n_seeds | seed-mean AUC ± SD | pooled (seed-avg-prob) AUC | case-bootstrap 95% CI | hierarchical (seeds × cases) 95% CI |
|---|---|---|---|---|---|---|
| 0.0 | 80 | 3 | 0.848 ± 0.032 | 0.867 | [0.783, 0.933] | [0.769, 0.937] |
| 1.0 | 80 | 3 | 0.863 ± 0.029 | 0.902 | [0.825, 0.965] | [0.796, 0.958] |
| 5.0 | 80 | 3 | 0.823 ± 0.018 | 0.886 | [0.808, 0.952] | [0.761, 0.940] |
| 10.0 | 80 | 3 | 0.802 ± 0.055 | 0.873 | [0.791, 0.942] | [0.719, 0.935] |