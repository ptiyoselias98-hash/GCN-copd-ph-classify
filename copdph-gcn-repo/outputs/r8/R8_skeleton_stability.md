# R8.2 — Skeleton stability under pre-erosion (TEASAR proxy)

True TEASAR parameter sweep requires kimimaro which is remote-only. As a
local proxy we apply skimage.skeletonize_3d (Lee94) to artery masks with
increasing binary erosion (0 / 1 / 2 voxels before skeletonization). Higher
erosion → coarser skeleton → mimics larger TEASAR `scale` parameter.

Cases: 10 (balanced across label × protocol).
Mean coefficient of variation across erosion levels: **0.584**
Max CV: **1.009**

| Case | er0 SL (mm) | er1 SL (mm) | er2 SL (mm) | CV | Δ er1 (%) | Δ er2 (%) |
|---|---|---|---|---|---|---|
| `ph_qianjufang_r14595451_saturday_ju` | 10591 | 8752 | 5976 | 0.225 | -17% | -44% |
| `ph_zhaoguoping_p02540627_thursday_f` | 20970 | 9819 | 3653 | 0.624 | -53% | -83% |
| `ph_fangmin_9001584683_thursday_apri` | 27296 | 21449 | 12364 | 0.302 | -21% | -55% |
| `ph_limaocai_l01147541_thursday_apri` | 24016 | 14059 | 5531 | 0.520 | -41% | -77% |
| `ph_chenmeizhen_9001615013_friday_ma` | 601765 | 271120 | 132859 | 0.587 | -55% | -78% |
| `nonph_heyude_c00242568_saturday_oct` | 18864 | 10984 | 4111 | 0.533 | -42% | -78% |
| `nonph_chenzhixing_n11110528_saturda` | 15581 | 10047 | 4500 | 0.450 | -36% | -71% |
| `nonph_fanhongsheng_k01700477_friday` | 52259 | 10046 | 2654 | 1.009 | -81% | -95% |
| `nonph_gejianping_m03069169_monday_m` | 17989 | 7638 | 2816 | 0.668 | -58% | -84% |
| `nonph_ganxiufang_l00829997_friday_a` | 24899 | 5822 | 1893 | 0.924 | -77% | -92% |

## Reading

Mean CV = 0.584, max = 1.009. Values < 0.2
indicate low sensitivity to mask-boundary perturbations, which is a
necessary (not sufficient) condition for TEASAR-parameter stability.

**Caveat**: this is NOT a true TEASAR `scale × const × pdrf_exp` sweep —
that requires kimimaro on the remote and is queued for a dedicated
Round 9 run once GPU contention clears (kimimaro is CPU-only but remote
is currently saturated with another user's dinomaly training).