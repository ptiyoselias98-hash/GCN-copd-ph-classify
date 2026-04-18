#!/usr/bin/env python3
"""
==========================================================================
 COPD-PH GCN 准确率提升实验框架 (Accuracy Improvement Sprint)

 基于当前代码分析的 5 个改进方向:

 Exp 1: Borderline-aware loss — 对 mPAP∈[18,22] 边缘病例加权
 Exp 2: mPAP regression auxiliary — 多任务: 分类 + mPAP回归
 Exp 3: Feature selection — 去除冗余特征, 只保留 top-K
 Exp 4: Ensemble — RF + GCN soft-vote
 Exp 5: Node-drop augmentation — 训练时随机删除叶节点(模拟pruning)

 用法 (Claude Code 一键执行):
   cd GCN-copd-ph-classify-main
   python improvement_experiments.py \
     --xlsx /path/to/copd-ph患者113例0331.xlsx \
     --cache_dir ./cache \
     --labels ./data/labels.csv \
     --splits ./data/splits.json \
     --radiomics ./data/copd_ph_radiomics.csv
==========================================================================
"""

import argparse, json, os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")

# ========================================
# Experiment 1: Borderline-aware weighting
# ========================================
def exp1_borderline_reweight(X, y, mpap_vals, n_splits=5):
    """
    Borderline cases (mPAP 18-22) are hardest to classify.
    Strategy: upweight borderline samples in training.
    """
    print("\n[Exp 1] Borderline-aware sample weighting")

    borderline = (mpap_vals >= 18) & (mpap_vals <= 22)
    n_bl = borderline.sum()
    print(f"  Borderline [18-22]: {n_bl} patients")

    scaler = StandardScaler()
    X_s = scaler.fit_transform(np.nan_to_num(X, nan=0.0))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {'baseline': [], 'borderline_2x': [], 'borderline_3x': []}

    for tr, te in skf.split(X_s, y):
        # Baseline
        rf = RandomForestClassifier(200, max_depth=8, class_weight='balanced', random_state=42)
        rf.fit(X_s[tr], y[tr])
        results['baseline'].append(roc_auc_score(y[te], rf.predict_proba(X_s[te])[:,1]))

        # 2x weight on borderline
        w = np.ones(len(tr))
        bl_mask = borderline[tr]
        w[bl_mask] = 2.0
        rf2 = RandomForestClassifier(200, max_depth=8, class_weight='balanced', random_state=42)
        rf2.fit(X_s[tr], y[tr], sample_weight=w)
        results['borderline_2x'].append(roc_auc_score(y[te], rf2.predict_proba(X_s[te])[:,1]))

        # 3x weight
        w3 = np.ones(len(tr))
        w3[bl_mask] = 3.0
        rf3 = RandomForestClassifier(200, max_depth=8, class_weight='balanced', random_state=42)
        rf3.fit(X_s[tr], y[tr], sample_weight=w3)
        results['borderline_3x'].append(roc_auc_score(y[te], rf3.predict_proba(X_s[te])[:,1]))

    for k, v in results.items():
        print(f"  {k:20s}: AUC = {np.mean(v):.3f} ± {np.std(v):.3f}")
    return results


# ========================================
# Experiment 3: Feature selection
# ========================================
def exp3_feature_selection(X, y, feature_names, n_splits=5):
    """Select top-K features by mutual information."""
    print("\n[Exp 3] Feature selection (mutual information)")

    X_clean = np.nan_to_num(X, nan=0.0)
    mi = mutual_info_classif(X_clean, y, random_state=42)
    ranked = np.argsort(mi)[::-1]

    results = {}
    for k in [10, 20, 30, 50, len(feature_names)]:
        k = min(k, len(feature_names))
        sel = ranked[:k]
        X_sel = X_clean[:, sel]
        X_s = StandardScaler().fit_transform(X_sel)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        aucs = []
        for tr, te in skf.split(X_s, y):
            rf = RandomForestClassifier(200, max_depth=8, class_weight='balanced', random_state=42)
            rf.fit(X_s[tr], y[tr])
            aucs.append(roc_auc_score(y[te], rf.predict_proba(X_s[te])[:,1]))

        results[f'top_{k}'] = aucs
        top_names = [feature_names[i] for i in sel[:5]]
        print(f"  Top-{k:3d}: AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}  "
              f"(top5: {', '.join(top_names[:3])}...)")

    # Print top 15 features
    print("\n  Top 15 features by mutual information:")
    for i, idx in enumerate(ranked[:15]):
        print(f"    {i+1:2d}. {feature_names[idx]:45s} MI={mi[idx]:.4f}")

    return results, mi, ranked


# ========================================
# Experiment 4: Ensemble RF + GBT
# ========================================
def exp4_ensemble(X, y, n_splits=5):
    """Soft-vote ensemble of RF + GBT + LR."""
    print("\n[Exp 4] Ensemble (RF + GBT + LR soft-vote)")

    X_s = StandardScaler().fit_transform(np.nan_to_num(X, nan=0.0))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {'RF': [], 'GBT': [], 'LR': [], 'Ensemble': []}

    for tr, te in skf.split(X_s, y):
        rf = RandomForestClassifier(200, max_depth=8, class_weight='balanced', random_state=42)
        rf.fit(X_s[tr], y[tr])
        p_rf = rf.predict_proba(X_s[te])[:, 1]

        gbt = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                          learning_rate=0.05, random_state=42)
        gbt.fit(X_s[tr], y[tr])
        p_gbt = gbt.predict_proba(X_s[te])[:, 1]

        lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
        lr.fit(X_s[tr], y[tr])
        p_lr = lr.predict_proba(X_s[te])[:, 1]

        p_ens = (p_rf + p_gbt + p_lr) / 3.0

        results['RF'].append(roc_auc_score(y[te], p_rf))
        results['GBT'].append(roc_auc_score(y[te], p_gbt))
        results['LR'].append(roc_auc_score(y[te], p_lr))
        results['Ensemble'].append(roc_auc_score(y[te], p_ens))

    for k, v in results.items():
        print(f"  {k:12s}: AUC = {np.mean(v):.3f} ± {np.std(v):.3f}")
    return results


# ========================================
# Main
# ========================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xlsx", required=True)
    p.add_argument("--output_dir", default="./outputs/improvements")
    p.add_argument("--cache_dir", default="./cache")
    p.add_argument("--labels", default="./data/labels.csv")
    p.add_argument("--splits", default="./data/splits.json")
    p.add_argument("--radiomics", default="./data/copd_ph_radiomics.csv")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_excel(args.xlsx, sheet_name='Sheet1')
    mask_ct = df['肺血管容积(ml)_y'].notna()
    sub = df[mask_ct].copy()
    y = (sub['PH'] == '是').astype(int).values

    # Feature matrix
    vasc_cols = [c for c in [
        '肺血管容积(ml)_y','肺血管平均密度(HU)','肺血管最大密度(HU)','肺血管最小密度(HU)',
        '肺血管血管分支数量_y','肺血管弯曲度','肺血管分形维度',
        '肺血管BV5(ml)_y','肺血管BV10(ml)_y','肺血管BV10+(ml)_y',
        '动脉容积(ml)','动脉平均密度(HU)','动脉弯曲度','动脉分形维度',
        '动脉BV5(ml)_y','动脉BV10(ml)_y','动脉BV10+(ml)_y',
        '静脉容积(ml)_y','静脉平均密度(HU)','静脉弯曲度','静脉分形维度',
        '静脉BV5(ml)_y','静脉BV10(ml)_y','静脉BV10+(ml)_y',
        '动脉血管分支数量_y','静脉血管分支数量_y',
    ] if c in sub.columns]

    para_cols = [c for c in [
        '左右肺容积(ml)','左右肺LAA910(%)','左右肺LAA950(%)',
        '左右肺平均密度(HU)','左右肺密度标准差(HU)','左右肺质量(g)',
        '左右肺支气管数量.1','左右肺支气管长度(cm).1','左右肺支气管体积(ml).1',
        '左右肺代','左右肺支气管数量[D<2mm]','左右肺支气管体积(ml)[D<2mm]',
        '左右肺Pi10(mm)','左右肺弯曲度','左右肺分形维度',
    ] if c in sub.columns]

    lobe_cols = [c for c in sub.columns if ('动脉(' in c or '静脉(' in c) and '肺叶' in c]
    all_cols = vasc_cols + para_cols + lobe_cols
    X = sub[all_cols].apply(pd.to_numeric, errors='coerce').values
    mpap = pd.to_numeric(sub['mPAP'], errors='coerce').values

    print("="*60)
    print(" COPD-PH GCN 准确率提升实验")
    print(f" n={len(sub)}, PH={y.sum()}, features={len(all_cols)}")
    print("="*60)

    all_results = {}

    # Exp 1
    r1 = exp1_borderline_reweight(X, y, mpap)
    all_results['exp1_borderline'] = {k: {'mean': np.mean(v), 'std': np.std(v)}
                                       for k, v in r1.items()}

    # Exp 3
    r3, mi, ranked = exp3_feature_selection(X, y, all_cols)
    all_results['exp3_feature_sel'] = {k: {'mean': np.mean(v), 'std': np.std(v)}
                                        for k, v in r3.items()}

    # Exp 4
    r4 = exp4_ensemble(X, y)
    all_results['exp4_ensemble'] = {k: {'mean': np.mean(v), 'std': np.std(v)}
                                     for k, v in r4.items()}

    # ========================================
    # Summary plot
    # ========================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Improvement Experiments — 5-fold CV AUC", fontsize=14, y=1.02)

    # Exp 1
    ax = axes[0]
    keys1 = list(r1.keys())
    means1 = [np.mean(r1[k]) for k in keys1]
    ax.bar(keys1, means1, color=['#4472C4','#5DCAA5','#D85A30'])
    ax.set_ylim(0.8, 1.0)
    ax.set_title("Exp 1: Borderline weighting")
    ax.set_ylabel("AUC")
    for i, v in enumerate(means1):
        ax.text(i, v+0.005, f"{v:.3f}", ha='center', fontsize=9)

    # Exp 3
    ax = axes[1]
    keys3 = list(r3.keys())
    means3 = [np.mean(r3[k]) for k in keys3]
    ax.bar(keys3, means3, color='#AFA9EC')
    ax.set_ylim(0.8, 1.0)
    ax.set_title("Exp 3: Feature selection")
    for i, v in enumerate(means3):
        ax.text(i, v+0.005, f"{v:.3f}", ha='center', fontsize=9)
    ax.tick_params(axis='x', rotation=30)

    # Exp 4
    ax = axes[2]
    keys4 = list(r4.keys())
    means4 = [np.mean(r4[k]) for k in keys4]
    ax.bar(keys4, means4, color=['#4472C4','#5DCAA5','#F0997B','#D85A30'])
    ax.set_ylim(0.8, 1.0)
    ax.set_title("Exp 4: Ensemble")
    for i, v in enumerate(means4):
        ax.text(i, v+0.005, f"{v:.3f}", ha='center', fontsize=9)

    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, "improvement_results.png")
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {fig_path}")

    # Save JSON
    json_path = os.path.join(args.output_dir, "improvement_results.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Saved: {json_path}")

    # ========================================
    # GCN-specific improvements (instructions for Claude Code)
    # ========================================
    print("\n" + "="*60)
    print(" GCN特定改进建议 (需在 Claude Code 中执行)")
    print("="*60)
    print("""
    以下改进需要 PyTorch + PyG 环境, 在本地 Claude Code 中执行:

    1. Node-drop augmentation (在 run_sprint2.py 的训练循环中加入):
       训练时以 p=0.1 概率随机删除叶节点, 模拟血管 pruning,
       迫使模型学习更鲁棒的拓扑特征而非过拟合节点数量.

    2. mPAP regression auxiliary (多任务学习):
       在 HybridGCN 中增加 regression head, 同时预测 mPAP 值.
       Loss = CE_loss + 0.1 * MSE_loss(mPAP_pred, mPAP_true)
       这会让 embedding 学到更细粒度的疾病严重程度信号.

    3. Fold-stratified by mPAP (改进交叉验证):
       当前 fold 4/5 差是因为 borderline 病例分布不均.
       按 mPAP 分层: <18, 18-22, 22-30, >30 四个桶, 确保每个 fold
       都有足够的 borderline cases.
    """)


if __name__ == '__main__':
    main()
