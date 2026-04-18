Sprint 5 实验结果总结
======================

实验配置:
  - Loss: focal
  - Globals: local4
  - Node Drop: 0.1
  - mPAP Aux Weight: 0.1
  - Epochs: 300
  - mPAP Stratified Splits: Yes (v2, 105 patients matched)

最佳结果 (by Pooled AUC):
  Sprint 5 enhanced/hybrid:
    Fold-mean AUC = 0.924 +/- 0.047
    Pooled AUC    = 0.889
    Sensitivity   = 0.870
    Specificity   = 0.920

vs Sprint 2 baseline/gcn_only:
    Fold-mean AUC = 0.926 +/- 0.078
    Pooled AUC    = 0.856
    Sensitivity   = 0.823
    Specificity   = 0.620

改进:
    Pooled AUC:   +0.033 (0.856 -> 0.889)
    Specificity:  +0.300 (0.620 -> 0.920)
    fold间一致性:  std 0.078 -> 0.047 (更稳定)

输出文件:
  - sprint5_radar_combined.png      : Sprint5 6配置雷达图
  - sprint5_vs_sprint2_radar.png    : Sprint5 vs Sprint2 对比雷达图
  - sprint_progression_radar.png    : 各Sprint最优配置演进雷达图
  - pooled_auc_comparison.png       : Pooled AUC 柱状图对比
  - sprint5_results.xlsx            : 详细数据表格
  - feature_ablation.png            : 特征消融实验结果 (Step 1)
