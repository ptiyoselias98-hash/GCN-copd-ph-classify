# Autonomous Experiment Pipeline — 使用说明

为避免余额不足导致实验中断，所有新实验已打包成可无人值守、可断点续跑的自动化流水线。

## 一键启动

```bat
run_all.bat
```

或直接：

```bash
cd copdph-gcn-repo
python auto_pipeline.py
```

## 执行的步骤

| # | 步骤 | 位置 | 产出 |
|---|------|------|------|
| 1 | 归因分析 (PA/Ao vs 小血管树) | 本地 | `outputs/attribution/` |
| 2 | RF/GBT/Ensemble baseline 提升 | 本地 | `outputs/improvements/` |
| 3 | 生成 mPAP-分层 5-fold splits | 本地 | `copdph-gcn-repo/data/splits_mpap_stratified.json` |
| 4 | 生成 {pid: mPAP} 查找表 | 本地 | `copdph-gcn-repo/data/mpap_lookup.json` |
| 5 | 推送 + 启动 sprint5 远程训练 (3 arms, 串行 nohup) | 远程 | `outputs/sprint5_*/` |
| 6 | 每 5 分钟轮询一次，直到 sentinel 出现（最长 24 小时） | 远程 | — |
| 7 | 拉取 sprint5 所有结果 | 远程 → 本地 | `outputs/sprint5_*/sprint5_results.json` |
| 8 | 汇总最终对比表 | 本地 | `outputs/final_comparison.xlsx`, `.png` |

## Sprint 5 三个实验臂（远程 nohup 串行）

| arm | node_drop_p | mpap_aux_weight | 用途 |
|-----|-------------|-----------------|------|
| `full`       | 0.10 | 0.10 | 主设: 3 项改进全开 |
| `ndrop_only` | 0.10 | 0.00 | 消融: 去掉 mPAP 回归 |
| `mpap_only`  | 0.00 | 0.10 | 消融: 去掉 node-drop |

全部使用 mPAP-分层 splits + focal loss + local4 globals。

## 断点续跑

状态文件：`outputs/auto_pipeline_state.json`

- 重新运行 `python auto_pipeline.py` — 从第一个未完成步骤恢复
- 重置并从头开始：`python auto_pipeline.py --reset`
- 跳到指定步骤：`python auto_pipeline.py --from-step 5`
- 只跑某一步：`python auto_pipeline.py --only-step 7`

## 远程作业独立于本地

`_remote_sprint5.py launch` 用 `nohup bash -lc "..." & disown` 启动，
**即使本地 Python 退出、网络断开、余额耗尽，远程训练仍持续进行**。
恢复后运行 `python _remote_sprint5.py status` 查看进度，或 `auto_pipeline.py`
会从 step 6 继续轮询。

## 手动诊断命令

```bash
# 查看远程训练状态
python copdph-gcn-repo/_remote_sprint5.py status

# 是否已完成
python copdph-gcn-repo/_remote_sprint5.py is_done

# 拉取结果
python copdph-gcn-repo/_remote_sprint5.py fetch
```

## 先决条件检查

- 本地 Python 需要：`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `openpyxl`, `paramiko`
- 远程 `pulmonary_bv5_py39` 已包含 `torch`, `torch_geometric`, 及现有 sprint 依赖
- `copd-ph患者113例0331.xlsx` 位于项目根目录（用于提取 mPAP + PH 标签）

## 评估指标

按 `CLAUDE.md` 规定，每个结果文件均报告 6 项指标：
**AUC, Accuracy, Precision, Sensitivity, F1, Specificity**
（Youden's J 阈值校准版本 + argmax 版本）。
