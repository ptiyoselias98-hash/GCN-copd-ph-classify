# P-δ radar: 六指标汇总(fold-mean)

| entry | AUC | Accuracy | Precision | Sensitivity | F1 | Specificity |
|---|---|---|---|---|---|---|
| Sprint 5 v2 baseline (106) | 0.924 | 0.895 | 0.940 | 0.870 | 0.905 | 0.920 |
| arm_a_ensemble best pooled (282) | 0.945 | 0.909 | 0.951 | 0.914 | 0.930 | 0.899 |
| arm_a_base / enhanced (282) | 0.950 | 0.933 | 0.970 | 0.929 | 0.948 | 0.942 |
| arm_b_base / baseline hybrid (113) | 0.911 | 0.873 | 0.978 | 0.846 | 0.902 | 0.971 |
| arm_b_base / enhanced gcn_only (113) | 0.916 | 0.876 | 0.967 | 0.873 | 0.912 | 0.943 |

## Figures
- `radar_arm_a_vs_sprint5.png` — arm_a 四个配置叠加
- `radar_arm_b_vs_sprint5.png` — arm_b hybrid baseline/enhanced
- `radar_all_best_vs_sprint5.png` — 最佳配置 vs Sprint 5 v2 baseline