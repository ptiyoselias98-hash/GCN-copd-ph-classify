# Phase C1 — Signature vs Disease/mPAP Severity

_2026-04-26_

## Outputs

- `signature_group_stats.csv` — T1 within-contrast PH-vs-nonPH (n_PH=163, n_nonPH=27): MWU + Cohen's d + bootstrap-500 d CI + Holm + FDR
- `mpap_correlation_table.csv` — T2 measured-mPAP within-contrast (n=102): Spearman ρ + 500-perm null + Holm + FDR
- `borderline_deepdive.csv` — T3 borderline n=12 (mPAP 18-22) DESCRIPTIVE only (n=1 PH + 11 nonPH; no inferential)
- `pruning_curve_results.json` — T4 N(d)~d^-α slope α with 3 bin schemes (sensitivity)
- `mpap_bin_trends.png` — top-6 mPAP-correlated features × bin (<20 / 20-25 / 25-35 / ≥35)
- `top_signature_forest_plot.png` — top-20 PH-vs-nonPH Cohen's d forest plot

## Headline findings

### T1 within-contrast PH-vs-nonPH (n=190)
- **9 Holm-sig + 43 FDR-sig** out of 172 features
- Top by |d|: `vein_persH1_total` d=−1.21, `paren_HU_p95` d=+1.14, `paren_std_HU` d=+1.10, `vein_persH0_total` d=−1.03
- TDA topology loss + parenchyma densification + parenchyma heterogeneity dominate

### T2 measured-mPAP correlation (n=102, permutation null 500-iter)
- **22 Holm-sig + 58 FDR-sig** out of 172 features
- Top by |ρ| (all perm-p = 0):
  - `paren_std_HU` ρ=+0.664 (parenchyma heterogeneity ↑ with mPAP)
  - `whole_std_HU` ρ=+0.635 (lung-level HU std)
  - `vein_persH1_total` ρ=−0.594 (TDA loop-topology persistence ↓ with mPAP)
  - `vein_persH0_total` ρ=−0.561
  - `paren_HU_p95` ρ=+0.558 (denser parenchyma at high HU as mPAP rises)
  - `vein_persH1_n_loops` ρ=−0.470
  - `vessel_airway_over_lung` ρ=+0.466 (vessel volume / lung volume ↑)

### T3 borderline n=12 (descriptive, hypothesis-generating only)
- 1 PH + 11 nonPH at mPAP 18-22 — too small for inferential claims
- Top descriptive contrasts saved to CSV for future power-up

### T4 pruning curve α (sensitivity across 3 bin schemes)
- Artery α median across schemes: 1.53 / 1.57 / 1.56 (per-patient log-log fit on diameter percentiles)
- Vein α median: 1.61 / 1.62 / 1.56
- α stable across binning schemes — robust pruning law-fit
- α biological range matches prior pulmonary vasculature literature (1.5-2.0)

## Cohort discipline

- T1, T4 use **C2 within-contrast n=190** (no protocol mixing)
- T2 uses C2 ∩ measured_mpap_flag = **n=102** (no default mPAP)
- T3 uses C3 borderline **n=12** descriptive only

## Codex DUAL REVIEW history

- Pre-execution: REVISE (T4 pruning α sign — log-log slope = -α not α; fixed)
- Post-execution: T2 FDR correction returned all-NaN due to NaN propagation through `np.minimum.accumulate`; fixed via finite-mask FDR
- Final: T2 22 Holm-sig + 58 FDR-sig, T1 9 Holm-sig + 43 FDR-sig, all per-mitigation-pass-1 cohort discipline preserved
