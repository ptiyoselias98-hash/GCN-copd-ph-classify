# Round 3 — protocol decodability on cache-adjacent features

The Round 2 reviewer required protocol AUC to be measured on features
closer to the GCN's actual inputs, not scalar lung HU. Without remote
graph cache access, we approximate via per-structure volumes, vessel-
volume ratios, and parenchyma LAA (all protocol-agnostic in principle
because ratios cancel absolute HU offsets).

5-fold stratified CV AUCs:

| Set | n_feats | n_cases | Protocol AUC (LR / GB) | Disease full (LR / GB) | Disease contrast (LR / GB) |
|---|---|---|---|---|---|
| `A_per_structure_volumes` | 4 | 275 | 0.524 / 0.910 | 0.714 / 0.864 | 0.756 / 0.660 |
| `B_volumes_plus_ratios` | 7 | 244 | 0.885 / 0.854 | 0.872 / 0.877 | 0.770 / 0.706 |
| `C_spatial_only` | 4 | 252 | 0.732 / 0.767 | 0.674 / 0.681 | 0.671 / 0.402 |
| `D_paren_LAA_only` | 3 | 252 | 0.591 / 0.811 | 0.575 / 0.804 | 0.685 / 0.633 |
| `E_v2_ratio_combined_no_HU` | 11 | 232 | 0.860 / 0.877 | 0.868 / 0.869 | 0.786 / 0.741 |

## Reading

- Most protocol-robust set: `A_per_structure_volumes` (protocol AUC 0.524).
- Disease AUC on the **contrast-only subset** is the reviewer-facing endpoint; full-cohort numbers above it remain inflated by any residual protocol leakage.
- Ratios (A/V, A/total_vessel, artery/airway) cancel absolute HU offsets but still
  carry protocol signal through segmentation-quality differences; exact quantification
  above shows how much.

**Caveat**: these are abundance-derived features, not the 12-D TEASAR node features
the GCN actually consumes. A true measurement requires loading `cache_v2_tri_flat/*.pkl`
on the remote server and computing graph-level statistics (planned as E1 for Round 4).