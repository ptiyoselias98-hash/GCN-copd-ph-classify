# W1 stress-test v2 — whole-lung vs parenchyma-only vs spatial

Feature-set comparison on 5-fold stratified CV (sklearn LR + GB).
Goal: find a lung-feature representation whose **protocol AUC is near 0.5**
(not trivially decodable) while **contrast-only disease AUC stays high**.

| Feature set | n_feats | Protocol AUC (LR / GB) | Disease AUC full (LR / GB) | Disease AUC contrast-only (LR / GB) |
|---|---|---|---|---|
| `whole_lung` | 11 | 0.900 / 0.889 | 0.879 / 0.873 | 0.824 / 0.734 |
| `parenchyma_only` | 10 | 0.857 / 0.851 | 0.870 / 0.841 | 0.860 / 0.777 |
| `spatial_paren` | 10 | 0.808 / 0.853 | 0.761 / 0.856 | 0.732 / 0.654 |
| `vessel_lung_integration` | 7 | 0.945 / 0.982 | 0.861 / 0.890 | 0.774 / 0.677 |
| `paren_plus_spatial` | 14 | 0.866 / 0.857 | 0.879 / 0.852 | 0.855 / 0.793 |

## Reading the matrix

- **whole_lung** (legacy v1 features) → protocol AUC 1.0, disease-contrast 0.68.
  Baseline confirming §14.3: the v1 lung-feature gain on the full cohort is almost
  entirely protocol leakage.
- **parenchyma_only** (lung − vessels − airway) → target is protocol AUC close to
  random (~0.5) with disease-contrast matching or exceeding whole_lung's 0.68.
  If protocol AUC is still high this indicates residual leakage from parenchyma
  density itself (possible: contrast in capillaries slightly changes HU).
- **spatial_paren** (apical/middle/basal LAA) → classic radiologic signature for
  upper-zone emphysema. Expected to contribute disease signal orthogonal to
  overall LAA fraction.
- **vessel_lung_integration** (vessel volumes + mean HU) → expected to have very
  high protocol AUC because artery/vein HU differ massively between contrast
  and plain-scan. Serves as a reference 'protocol-decoder' set.
- **paren_plus_spatial** → the candidate combined v2 feature set for disease
  classification that should be substantially protocol-robust.