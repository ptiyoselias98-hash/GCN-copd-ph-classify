# R4.2 Skeleton-length abundance — HiPaS T1 retry

Replaces R3 volume-fraction proxy with skimage.skeletonize_3d skeleton
length in mm per L of lung. This is the literal HiPaS metric.

## T1_SL_artery_per_L_PH_vs_nonPH_contrast
- **hipas_prediction**: PH < nonPH
- **n_ph**: 169
- **n_nonph**: 27
- **median_ph**: 5299.9897
- **median_nph**: 3901.5419
- **p_two_sided**: 0.0032
- **cliffs_delta**: 0.3539
- **direction_matches_hipas**: False

## T2_SL_vein_per_L_vs_LAA910_contrast
- **hipas_prediction**: negative
- **n**: 186
- **spearman_rho**: -0.6542
- **spearman_p**: 0.0000
- **direction_matches_hipas**: True

## T2_SL_vein_per_L_vs_LAA910_plain_scan
- **hipas_prediction**: negative
- **n**: 46
- **spearman_rho**: -0.4933
- **spearman_p**: 0.0005
- **direction_matches_hipas**: True

**T1 verdict**: direction matches HiPaS = **False**, p=0.003193, Cliff's δ=+0.354.