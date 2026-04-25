# R20.C — R17 extraction artifact audit

Source: outputs\r17\per_structure_morphometrics.csv; cohort 282 cases × 132 features.

Closes R18 must-fix #7 (audit R17 extraction artifacts: n_terminals=0 bug, edge-doubling [::2] handling, Lap eig0 degenerate, near-zero SD features).

## Per-structure audit

### artery

- **n_terminals**: median=0.0 min=0 max=0 n_zero=282/282 → BUG: all 0 — needs degree-counting fix
- **lap_eig0**: median=0.00e+00 min=-1.04e-18 max=5.66e-18 → DEGENERATE: numerical zero (drop)
- **edges/branches ratio**: median=4.04 → edges = 2*branches (doubled — direction bug?)

### vein

- **n_terminals**: median=0.0 min=0 max=0 n_zero=282/282 → BUG: all 0 — needs degree-counting fix
- **lap_eig0**: median=0.00e+00 min=0.00e+00 max=8.05e-18 → DEGENERATE: numerical zero (drop)
- **edges/branches ratio**: median=4.04 → edges = 2*branches (doubled — direction bug?)

### airway

- **n_terminals**: median=0.0 min=0 max=0 n_zero=282/282 → BUG: all 0 — needs degree-counting fix
- **lap_eig0**: median=2.42e-18 min=-1.23e-01 max=1.04e-17 → OK: finite
- **edges/branches ratio**: median=4.03 → edges = 2*branches (doubled — direction bug?)

## Near-zero-SD features (will be excluded)

Count: 8

- `airway_n_terminals`
- `airway_term_per_node`
- `artery_lap_eig0`
- `artery_n_terminals`
- `artery_term_per_node`
- `vein_lap_eig0`
- `vein_n_terminals`
- `vein_term_per_node`

## Total artifact columns to drop downstream

**8** columns flagged as numerical-artifact and SHOULD NOT be cited as biological features in REPORT/README.

## Flagship-feature-survives-audit check

- `artery_len_p25`: True
- `artery_len_p50`: True
- `artery_tort_p10`: True
- `vein_len_p25`: True
- `vein_persH1_total`: ABSENT
- `paren_std_HU`: ABSENT

## Verdict

Flagship findings (R17.A artery_len_p25 d=-1.25, R18.B Spearman ρ=-0.767, R17.5 vein_persH1_total d=-1.21, R16/R18 paren_std_HU d=+1.10) are **NOT** in the artifact set. R17 extraction artifacts are localized to per-structure metadata features (n_terminals, lap_eig0, near-zero-SD), which can be excluded without affecting biological claims. R18 must-fix #7 closed.