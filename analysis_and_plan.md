# 代码分析总结 + Claude Code 执行计划

## 1. 当前建图方式总结

### 图的构建流程

```
CT NIfTI → 血管 mask → 3D 骨架化 (skeletonize) → 解析拓扑 → PyG Data
```

**节点 (Nodes)**: 骨架上的分叉点 (bifurcation) 和末端点 (terminal)

**边 (Edges)**: 两个分叉/末端点之间的血管段, 双向无向边.
另外还加了空间邻近边 (距离 < 15mm 的非树连接节点对)

**节点特征 (12D → 增强后 13D)**:

| 维度 | 特征 | 来源 | 问题 |
|------|------|------|------|
| 0 | diameter | 距离变换 → 后来用商业BV5校准 | 已修复 |
| 1 | length | 骨架路径长度 | OK |
| 2 | tortuosity | 路径/欧氏距离 | OK |
| 3 | ct_density | 原来全是0 → 后用商业HU覆盖 | 已修复但是常数 |
| 4-6 | orientation | 方向向量 (3D) | OK |
| 7-9 | centroid | 空间坐标 (3D) | OK |
| 10 | strahler | Strahler 流序 | OK |
| 11 | degree | 节点度 | OK |
| 12 | curvature | 1-cos(角度), 增强后添加 | OK |

**图级别全局特征 (12D, `data.global_features`)**:
fractal_dim, artery_density, vein_density, vein_bv5, vein_branch_count,
bv5_ratio, artery_vein_vol_ratio, total_bv5, lung_density_std,
vein_bv10, total_branch_count, vessel_tortuosity

→ 这些 **不广播到节点**, 而是在 pooling 之后 concat 到图嵌入上.

### 模型架构

```
HybridGCN:
  GraphSAGE(13D→64) × 3层 → global_mean_pool → concat global_features(12D)
  → concat radiomics(45D) → MLP(121→64→32→2)
```

### 当前最佳结果

| 实验 | 配置 | AUC (fold-mean) | pooled AUC |
|------|------|-----------------|------------|
| Sprint 2 | baseline/gcn_only | **0.926** | 0.856 |
| Sprint 2 | enhanced/hybrid | 0.887 | 0.822 |
| Sprint 3 focal | baseline/gcn_only | 0.909 | 0.768 |
| Sprint 3 focal | enhanced/hybrid | 0.890 | **0.833** |
| Sprint 3 wce | enhanced/radiomics_only | 0.886 | **0.869** |
| Sprint 4b_av | baseline/hybrid | 0.907 | 0.840 |

**核心矛盾**: fold-mean AUC 高 (0.926) 但 pooled AUC 低 (0.856),
说明 fold 4/5 严重拖后腿 (AUC 0.78), 而 fold 4/5 的 borderline 病例多.

---

## 2. PA直径 vs 小血管树: 谁在起作用?

从数据中提取的关键证据:

| 指标 | PH组 | 非PH组 | 诊断敏感度 |
|------|------|--------|-----------|
| PA直径 > 2.9cm (echo) | 16/63 | 3/18 | **19%** (极低) |
| PA/Ao > 1 | 38/85 | 7/28 | **45%** (中等) |
| mPAP > 20 (金标准) | 85/85 | 0/28 | 100% |
| 小血管特征 (RF AUC) | — | — | **~0.90** |

**结论**: PA直径>29mm 和 PA/Ao>1 是非常弱的信号, 分类能力主要来自:
1. **肺小血管树拓扑** (分支数量、弯曲度、BV5/BV10、分形维度)
2. **肺叶级别差异** (右下肺叶血管变化最显著)

---

## 3. Claude Code 一键执行命令

### Step 1: 归因分析 (确认小血管树 vs PA/Ao 贡献)

```bash
cd GCN-copd-ph-classify-main

# 把脚本复制到项目目录
cp /path/to/attribution_analysis.py .
cp /path/to/improvement_experiments.py .

# 运行归因分析
python attribution_analysis.py \
  --xlsx /path/to/copd-ph患者113例0331.xlsx \
  --output_dir ./outputs/attribution
```

### Step 2: 改进实验 (radiomics baseline 提升)

```bash
python improvement_experiments.py \
  --xlsx /path/to/copd-ph患者113例0331.xlsx \
  --output_dir ./outputs/improvements
```

### Step 3: GCN 改进 — mPAP 分层 + Node-drop (需要 PyG)

在 Claude Code 中对项目执行以下修改:

```
请修改 run_sprint2.py, 实现以下 3 个改进:

1. mPAP 分层交叉验证:
   当前的 splits.json 导致 fold 4/5 的 borderline (mPAP 18-22) 
   病例过多, AUC 只有 0.78. 请重新生成 splits, 按 mPAP 四个桶
   (<18, 18-22, 22-30, >30) 做分层, 确保每个 fold 都包含足够的
   边缘病例. 保存新的 splits 为 data/splits_mpap_stratified.json

2. Node-drop augmentation:
   在 _train_fold 的训练循环中, 以 p=0.1 的概率随机删除 degree=1
   的叶节点 (及其关联边), 模拟 PH 的血管 pruning 过程. 
   只在训练时做, 验证时不做.

3. mPAP regression auxiliary:
   在 HybridGCN 中添加一个回归头:
   self.mpap_head = nn.Linear(fused_dim, 1)
   训练 loss = CE + 0.1 * MSE(mpap_pred, mpap_true)
   需要在 data.mpap 中存储每个病例的 mPAP 值.

用新的 splits 重新跑 enhanced/hybrid, 对比改进前后的 pooled AUC.
```

### Step 4: 最终对比报告

```
请汇总所有实验结果, 生成一个 comparison table:

| 方法 | fold-mean AUC | pooled AUC | Specificity |
|------|--------------|------------|-------------|
| Sprint 2 best (baseline/gcn_only) | 0.926 | 0.856 | 0.620 |
| Sprint 3 best (enhanced/hybrid focal) | 0.890 | 0.833 | 0.950 |
| + mPAP stratified splits | ? | ? | ? |
| + Node-drop augmentation | ? | ? | ? |
| + mPAP regression auxiliary | ? | ? | ? |
| Ensemble (RF+GCN) | ? | ? | ? |

保存为 outputs/final_comparison.xlsx 和 outputs/final_comparison.png
```
