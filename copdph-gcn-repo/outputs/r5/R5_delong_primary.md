# R5 — Single-arm DeLong CIs (contrast-only PRIMARY ENDPOINT)

**Note**: arm_b's dataset is restricted to 92 cases by the radiomics-feature
requirement; arm_c uses all 189. Paired DeLong on the same case set is
deferred to Round 6 (rebuild arm_b with --skip_radiomics_filter).

- arm_b (n=92, mode=gcn_only): AUC = **0.8462** [0.7340, 0.9584]
- arm_c (n=189, mode=gcn_only): AUC = **0.8391** [0.7527, 0.9255]
- Δ AUC (unpaired, approximate): -0.0071, z=-0.099, p=0.9214

## Reading

- arm_c CI excludes 0.5 (random)? **YES** — disease
  signal on contrast-only is significant.
- The unpaired Δ test is approximate; the paired version is the formal
  W6 endpoint. Round 6 will close that gap.