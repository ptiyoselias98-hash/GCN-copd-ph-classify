# E3 — HiPaS-aligned disease-direction test (local vessel abundance)

HiPaS (Chu et al., Nature Comm 2025, n=11,784) reports (lung-volume controlled):

- **PAH → lower pulmonary artery abundance** (skeleton length, branch count)
- **COPD → lower pulmonary vein abundance**

We test these directions on our cohort using `artery_vol_mL / lung_vol_mL`
and `vein_vol_mL / lung_vol_mL` as abundance proxies, restricted to cases
without placeholder segmentations.

## T1_artery_frac_PH_vs_nonPH_contrast

- **hipas_prediction**: PH < nonPH (reduced artery abundance)
- **n_ph**: 158
- **n_nonph**: 27
- **median_ph**: 0.0814
- **median_nph**: 0.0473
- **U**: 3279.0000
- **p_two_sided**: 0.0000
- **p_one_sided_less**: 1.0000
- **cliffs_delta**: 0.5373
- **direction_matches_hipas**: False

## T2_vein_frac_vs_LAA910_contrast

- **hipas_prediction**: Spearman(LAA_910, vein_frac) < 0 (more emphysema → less vein)
- **n**: 185
- **spearman_rho**: -0.7446
- **spearman_p**: 0.0000
- **direction_matches_hipas**: True

## T2_vein_frac_vs_LAA910_plain_scan

- **hipas_prediction**: Spearman(LAA_910, vein_frac) < 0 (more emphysema → less vein)
- **n**: 46
- **spearman_rho**: -0.5113
- **spearman_p**: 0.0003
- **direction_matches_hipas**: True

## T3_artery_frac_vs_LAA910_contrast

- **hipas_prediction**: weaker effect than vein (COPD-specific)
- **n**: 185
- **spearman_rho**: -0.6486
- **spearman_p**: 0.0000

## T3_artery_frac_vs_LAA910_plain_scan

- **hipas_prediction**: weaker effect than vein (COPD-specific)
- **n**: 46
- **spearman_rho**: -0.4492
- **spearman_p**: 0.0017

## T4_vein_frac_PH_vs_nonPH_contrast

- **hipas_prediction**: weaker than T1 (PH is artery-specific)
- **n_ph**: 158
- **n_nonph**: 27
- **median_ph**: 0.0607
- **median_nph**: 0.0456
- **p_two_sided**: 0.0633
- **cliffs_delta**: 0.2241

## Summary

- T1 (PAH → ↓artery on contrast-only): median PH 0.0814 vs nonPH 0.0473, two-sided p=8.401e-06, Cliff's δ=+0.537. Direction matches HiPaS: **False**.
- T2 (COPD → ↓vein on contrast): Spearman ρ=-0.745, p=5.966e-34. Direction matches HiPaS: **True**.

**Reading**: we use volume fractions as a crude proxy for HiPaS's skeleton-length
and branch-count metrics. True skeleton-based abundance requires the remote kimimaro
graphs (queued). Even with this proxy, directional agreement with HiPaS serves as
a literature-aligned falsification for our cache quality.