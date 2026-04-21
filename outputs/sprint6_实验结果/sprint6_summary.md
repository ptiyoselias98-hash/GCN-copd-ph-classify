# Sprint 6 六臂结果汇总

**生成时间**: from outputs/sprint6_* (fetched 2026-04-21)
**远程完成时间**: 2026-04-20 21:39
**GPU0 耗时**: ~2h (arm_a × 4)  |  **GPU1 耗时**: ~2h (arm_b × 2)

## 1. 六臂设计

| 目录 | 标签 | 数据集 | 模式 | 特殊配置 |
|---|---|---|---|---|
| `sprint6_arm_a_base` | arm_a / 282 / base | — | — | vanilla |
| `sprint6_arm_a_paao` | arm_a / 282 / paao (=base, bug) | — | — | launcher dropped --pa_ao_json; effectively a re-run of base |
| `sprint6_arm_a_full` | arm_a / 282 / full | — | — | augment=edge_drop+feature_mask, residual=True |
| `sprint6_arm_a_ensemble` | arm_a / 282 / ensemble | — | — | augment+residual, repeats=3 |
| `sprint6_arm_b_base` | arm_b / 113 / base | — | — | 3-mode (radiomics/gcn/hybrid), vanilla |
| `sprint6_arm_b_full` | arm_b / 113 / full | — | — | 3-mode, augment+residual |

> ⚠️ **`arm_a_paao` 是 launcher bug 的产物**,`launch_sprint6.sh` 漏传 `--pa_ao_json $PA_AO`,所以实际等价于 arm_a_base。数值差异纯属随机。

## 2. 全部结果(Youden-threshold, fold-mean ± std)

| arm | feat | mode | pooled_AUC | AUC_mean±std | Accuracy_mean±std | Precision_mean±std | Sensitivity_mean±std | F1_mean±std | Specificity_mean±std |
|---|---|---|---|---|---|---|---|---|---|
| arm_a / 282 / base | baseline | gcn_only | 0.9370 | 0.955±0.025 | 0.917±0.036 | 0.943±0.040 | 0.935±0.047 | 0.938±0.028 | 0.880±0.085 |
| arm_a / 282 / base | enhanced | gcn_only | 0.9350 | 0.950±0.033 | 0.933±0.036 | 0.970±0.026 | 0.929±0.048 | 0.949±0.028 | 0.943±0.051 |
| arm_a / 282 / paao (=base, bug) | baseline | gcn_only | 0.9344 | 0.957±0.026 | 0.929±0.025 | 0.949±0.033 | 0.947±0.043 | 0.947±0.021 | 0.894±0.065 |
| arm_a / 282 / paao (=base, bug) | enhanced | gcn_only | 0.9255 | 0.947±0.035 | 0.936±0.044 | 0.976±0.021 | 0.929±0.069 | 0.951±0.034 | 0.954±0.042 |
| arm_a / 282 / full | baseline | gcn_only | 0.9324 | 0.939±0.033 | 0.898±0.059 | 0.953±0.027 | 0.894±0.109 | 0.918±0.056 | 0.899±0.072 |
| arm_a / 282 / full | enhanced | gcn_only | 0.9349 | 0.945±0.034 | 0.925±0.032 | 0.965±0.028 | 0.923±0.051 | 0.942±0.025 | 0.931±0.054 |
| arm_a / 282 / ensemble | baseline | gcn_only | 0.9439 | 0.945±0.033 | 0.909±0.032 | 0.951±0.029 | 0.914±0.063 | 0.930±0.027 | 0.899±0.064 |
| arm_a / 282 / ensemble | enhanced | gcn_only | 0.9411 | 0.948±0.031 | 0.931±0.022 | 0.963±0.030 | 0.935±0.036 | 0.948±0.018 | 0.925±0.062 |
| arm_b / 113 / base | baseline | radiomics_only | 0.8000 | 0.881±0.115 | 0.846±0.085 | 0.950±0.100 | 0.841±0.099 | 0.885±0.064 | 0.914±0.171 |
| arm_b / 113 / base | baseline | gcn_only | 0.8422 | 0.863±0.059 | 0.808±0.130 | 0.970±0.038 | 0.784±0.182 | 0.851±0.104 | 0.931±0.086 |
| arm_b / 113 / base | baseline | hybrid | 0.8128 | 0.911±0.088 | 0.873±0.110 | 0.978±0.044 | 0.846±0.133 | 0.902±0.083 | 0.971±0.057 |
| arm_b / 113 / base | enhanced | radiomics_only | 0.8372 | 0.886±0.116 | 0.855±0.084 | 0.950±0.100 | 0.858±0.089 | 0.895±0.063 | 0.914±0.171 |
| arm_b / 113 / base | enhanced | gcn_only | 0.8544 | 0.916±0.069 | 0.876±0.055 | 0.967±0.067 | 0.873±0.088 | 0.912±0.037 | 0.943±0.114 |
| arm_b / 113 / base | enhanced | hybrid | 0.7850 | 0.917±0.072 | 0.867±0.080 | 0.978±0.044 | 0.837±0.097 | 0.899±0.062 | 0.971±0.057 |
| arm_b / 113 / full | baseline | radiomics_only | 0.8094 | 0.896±0.100 | 0.836±0.106 | 0.950±0.100 | 0.832±0.121 | 0.877±0.082 | 0.914±0.171 |
| arm_b / 113 / full | baseline | gcn_only | 0.8411 | 0.878±0.059 | 0.859±0.067 | 0.968±0.041 | 0.843±0.095 | 0.896±0.048 | 0.931±0.086 |
| arm_b / 113 / full | baseline | hybrid | 0.8122 | 0.908±0.096 | 0.889±0.129 | 0.943±0.089 | 0.920±0.160 | 0.918±0.097 | 0.874±0.170 |
| arm_b / 113 / full | enhanced | radiomics_only | 0.8167 | 0.903±0.090 | 0.858±0.104 | 0.945±0.078 | 0.858±0.128 | 0.893±0.082 | 0.903±0.122 |
| arm_b / 113 / full | enhanced | gcn_only | 0.8461 | 0.913±0.064 | 0.883±0.084 | 0.956±0.065 | 0.894±0.117 | 0.916±0.061 | 0.903±0.122 |
| arm_b / 113 / full | enhanced | hybrid | 0.8122 | 0.898±0.107 | 0.889±0.129 | 0.978±0.044 | 0.868±0.153 | 0.912±0.100 | 0.971±0.057 |

## 3. 最佳表现 / 对比 Sprint 5 v2 baseline

Sprint 5 v2 enhanced/hybrid: AUC **0.924** ± 0.047, Sens **0.870**, Spec **0.920**, pooled_AUC **0.889**

- **Best pooled AUC**: `arm_a / 282 / ensemble | baseline/gcn_only` → pooled_AUC=0.9439  (Sens=0.914, Spec=0.899, F1=0.930)
- **Best fold-mean AUC**: `arm_a / 282 / paao (=base, bug) | baseline/gcn_only` → mean_AUC=0.9567  (Sens=0.947, Spec=0.894, F1=0.947)
- **Best Sensitivity**: `arm_a / 282 / paao (=base, bug) | baseline/gcn_only` → mean_Sensitivity=0.9471  (Sens=0.947, Spec=0.894, F1=0.947)
- **Best Specificity**: `arm_b / 113 / base | baseline/hybrid` → mean_Specificity=0.9714  (Sens=0.846, Spec=0.971, F1=0.902)
- **Best F1**: `arm_a / 282 / paao (=base, bug) | enhanced/gcn_only` → mean_F1=0.9506  (Sens=0.929, Spec=0.954, F1=0.951)

## 4. 快速解读

- **arm_a (282样本, gcn_only)** 的 fold-mean AUC 普遍在 0.92–0.96 区间,比 arm_b 高很多,因为数据量是 arm_b 的 2.5 倍。
- **arm_b (113样本, 3-mode)** 的 `hybrid` 模式在 base 里 AUC ~0.91,和 Sprint 5 baseline (0.924) 基本持平;加 augment+residual 后 `full` 的 hybrid 几乎没变。
- **arm_a_ensemble** 跑 3 repeat 理论上降方差,可用它评估 augment+residual 的稳定性。
- **arm_a_paao 可弃用** — 是 launcher bug 的副产品。如需真 PA/Ao 实验,修 `launch_sprint6.sh` 里缺失的 `--pa_ao_json` 再跑。
