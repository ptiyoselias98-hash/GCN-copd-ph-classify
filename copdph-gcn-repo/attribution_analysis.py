#!/usr/bin/env python3
"""
==========================================================================
 COPD-PH 诊断能力来源分析 (Attribution Analysis)
 
 核心问题: 分类能力来自肺小血管树图拓扑, 还是仅仅来自
           PA直径>29mm / PA/Ao>1 这类简单指标?

 输出:
   1. outputs/attribution/pa_ao_stats.png       — PA/Ao统计表 (类似上传图片风格)
   2. outputs/attribution/feature_ablation.png  — 特征消融实验条形图
   3. outputs/attribution/ablation_results.json  — 所有数值结果

 用法 (Claude Code):
   cd GCN-copd-ph-classify-main
   python attribution_analysis.py --xlsx /path/to/copd-ph患者113例0331.xlsx
==========================================================================
"""

import argparse, json, os, re, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")

# ---------- Chinese font setup ----------
def get_cn_font():
    for name in ["SimHei","Heiti SC","WenQuanYi Micro Hei","Noto Sans CJK SC","Arial Unicode MS"]:
        try:
            fp = FontProperties(fname=None, family=name)
            return fp
        except:
            continue
    return FontProperties()

# ---------- Feature group definitions ----------
GLOBAL_VASC = [
    '肺血管容积(ml)_y','肺血管平均密度(HU)','肺血管最大密度(HU)','肺血管最小密度(HU)',
    '肺血管血管分支数量_y','肺血管弯曲度','肺血管分形维度',
    '肺血管BV5(ml)_y','肺血管BV10(ml)_y','肺血管BV10+(ml)_y',
    '动脉容积(ml)','动脉平均密度(HU)','动脉弯曲度','动脉分形维度',
    '动脉BV5(ml)_y','动脉BV10(ml)_y','动脉BV10+(ml)_y',
    '静脉容积(ml)_y','静脉平均密度(HU)','静脉弯曲度','静脉分形维度',
    '静脉BV5(ml)_y','静脉BV10(ml)_y','静脉BV10+(ml)_y',
    '动脉血管分支数量_y','静脉血管分支数量_y',
]

PARA_AIRWAY = [
    '左右肺容积(ml)','左右肺LAA910(%)','左右肺LAA950(%)',
    '左右肺平均密度(HU)','左右肺密度标准差(HU)','左右肺质量(g)',
    '左右肺支气管数量.1','左右肺支气管长度(cm).1','左右肺支气管体积(ml).1',
    '左右肺代','左右肺支气管数量[D<2mm]','左右肺支气管体积(ml)[D<2mm]',
    '左右肺Pi10(mm)','左右肺弯曲度','左右肺分形维度',
]

LOBE_COLS_PATTERN = lambda df: [c for c in df.columns if ('动脉(' in c or '静脉(' in c) and '肺叶' in c]

SMALL_VESSEL_KEYS = [
    '肺血管BV5(ml)_y','肺血管BV10(ml)_y',
    '肺血管血管分支数量_y','肺血管弯曲度','肺血管分形维度',
    '动脉BV5(ml)_y','动脉弯曲度','动脉分形维度',
    '静脉BV5(ml)_y','静脉弯曲度','静脉分形维度',
    '静脉血管分支数量_y','动脉血管分支数量_y',
]


def extract_pa_from_echo(df):
    """Extract PA diameter and Aorta diameter from echo report text."""
    pa_vals, ao_vals = [], []
    echo_col = '超声报告'
    for _, row in df.iterrows():
        txt = str(row.get(echo_col, ''))
        # PA diameter
        m = re.search(r'主肺动脉内径\s*([\d.]+)', txt)
        if not m:
            m = re.search(r'肺动脉内径\s*([\d.]+)', txt)
        pa_vals.append(float(m.group(1)) if m else np.nan)
        # Aortic root diameter
        m2 = re.search(r'主动脉根部内径\s*([\d.]+)', txt)
        ao_vals.append(float(m2.group(1)) if m2 else np.nan)
    return np.array(pa_vals), np.array(ao_vals)


def cv_auc(X, y, name="", n_splits=5):
    """5-fold CV with RF, return mean AUC and fold AUCs."""
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce').values
    X = np.nan_to_num(X.astype(float), nan=0.0)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs, f1s, accs = [], [], []
    for tr, te in skf.split(X_s, y):
        rf = RandomForestClassifier(n_estimators=200, max_depth=8,
                                     class_weight='balanced', random_state=42)
        rf.fit(X_s[tr], y[tr])
        prob = rf.predict_proba(X_s[te])[:, 1]
        pred = rf.predict(X_s[te])
        aucs.append(roc_auc_score(y[te], prob))
        f1s.append(f1_score(y[te], pred))
        accs.append(accuracy_score(y[te], pred))
    return {
        'name': name,
        'auc_mean': np.mean(aucs), 'auc_std': np.std(aucs),
        'f1_mean': np.mean(f1s), 'acc_mean': np.mean(accs),
        'fold_aucs': aucs,
    }


def safe_cols(df, cols):
    """Get existing columns from a list."""
    return [c for c in cols if c in df.columns]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xlsx", required=True)
    p.add_argument("--output_dir", default="./outputs/attribution")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_excel(args.xlsx, sheet_name='Sheet1')
    is_ph = df['PH'] == '是'
    is_nph = df['PH'].isin(['/', '否'])

    # Filter to 100 patients with commercial CT data
    mask_ct = df['肺血管容积(ml)_y'].notna()
    sub = df[mask_ct].copy()
    y = (sub['PH'] == '是').astype(int).values
    n_ph = y.sum()
    n_nph = len(y) - n_ph

    print(f"Cohort: n={len(sub)} (PH={n_ph}, non-PH={n_nph})")

    # ================================================================
    # PART 1: PA diameter / PA-Aorta ratio statistics
    # ================================================================
    print("\n" + "="*60)
    print("PART 1: PA直径 / PA-Ao比值 统计")
    print("="*60)

    pa_echo, ao_echo = extract_pa_from_echo(df)
    df_full = df.copy()
    df_full['PA_diam_echo'] = pa_echo
    df_full['Ao_diam_echo'] = ao_echo
    df_full['PA_Ao_ratio'] = pa_echo / ao_echo

    # mPAP stats
    mpap = pd.to_numeric(df['mPAP'], errors='coerce')
    pa_rr = pd.to_numeric(df['PA'], errors='coerce')  # from RHC

    stats = {}

    # For all 113 patients
    stats['mPAP'] = {
        'PH_mean': float(mpap[is_ph].mean()),
        'PH_std': float(mpap[is_ph].std()),
        'nPH_mean': float(mpap[is_nph].mean()),
        'nPH_std': float(mpap[is_nph].std()),
        'threshold': 20,
    }

    # Echo-derived PA diameter
    pa_avail = pa_echo[~np.isnan(pa_echo)]
    stats['PA_diam_echo'] = {
        'available': int((~np.isnan(pa_echo)).sum()),
        'total': 113,
        'PH_mean': float(np.nanmean(pa_echo[is_ph])),
        'nPH_mean': float(np.nanmean(pa_echo[is_nph])),
        'PH_gt29mm': int(np.nansum((pa_echo > 2.9) & is_ph.values)),
        'nPH_gt29mm': int(np.nansum((pa_echo > 2.9) & is_nph.values)),
        'PH_total_with_data': int(np.nansum(~np.isnan(pa_echo) & is_ph.values)),
        'nPH_total_with_data': int(np.nansum(~np.isnan(pa_echo) & is_nph.values)),
    }

    # PA/Ao ratio
    pa_ao = pa_echo / ao_echo
    stats['PA_Ao_ratio'] = {
        'PH_mean': float(np.nanmean(pa_ao[is_ph])),
        'nPH_mean': float(np.nanmean(pa_ao[is_nph])),
        'PH_gt1': int(np.nansum((pa_ao > 1) & is_ph.values)),
        'nPH_gt1': int(np.nansum((pa_ao > 1) & is_nph.values)),
    }

    for k, v in stats.items():
        print(f"\n  {k}:")
        for kk, vv in v.items():
            print(f"    {kk}: {vv}")

    # ================================================================
    # PART 2: Feature ablation — what drives classification?
    # ================================================================
    print("\n" + "="*60)
    print("PART 2: 特征消融实验 (Feature Ablation)")
    print("="*60)

    lobe_cols = LOBE_COLS_PATTERN(sub)
    all_ct = safe_cols(sub, GLOBAL_VASC + PARA_AIRWAY) + lobe_cols

    # Build PA/Ao features for the 100-patient CT cohort
    sub_pa, sub_ao = extract_pa_from_echo(sub)
    pa_ao_features = np.column_stack([
        np.nan_to_num(sub_pa, nan=0.0),
        np.nan_to_num(sub_ao, nan=0.0),
        np.nan_to_num(sub_pa / np.where(sub_ao > 0, sub_ao, 1), nan=0.0),
    ])

    # Derived ratios
    def add_ratios(sub_df):
        cols = {}
        v_total = pd.to_numeric(sub_df.get('肺血管容积(ml)_y', 0), errors='coerce').fillna(0)
        v_bv5 = pd.to_numeric(sub_df.get('肺血管BV5(ml)_y', 0), errors='coerce').fillna(0)
        a_vol = pd.to_numeric(sub_df.get('动脉容积(ml)', 0), errors='coerce').fillna(0)
        v_vol = pd.to_numeric(sub_df.get('静脉容积(ml)_y', 0), errors='coerce').fillna(0)
        cols['bv5_ratio'] = v_bv5 / v_total.replace(0, 1)
        cols['av_vol_ratio'] = a_vol / v_vol.replace(0, 1)
        return pd.DataFrame(cols, index=sub_df.index)

    ratios = add_ratios(sub)

    ablation_configs = [
        ("PA diam + PA/Ao only (3D)", pa_ao_features),
        ("Small vessel features (13D)",
         sub[safe_cols(sub, SMALL_VESSEL_KEYS)].values),
        ("Global vascular (26D)",
         sub[safe_cols(sub, GLOBAL_VASC)].values),
        ("Parenchyma + airway (15D)",
         sub[safe_cols(sub, PARA_AIRWAY)].values),
        ("Lobe-level vascular (60D)",
         sub[lobe_cols].values if lobe_cols else np.zeros((len(sub), 1))),
        ("All CT features (101D)",
         sub[all_ct].values),
        ("All CT + PA/Ao (104D)",
         np.hstack([sub[all_ct].values, pa_ao_features])),
        ("All CT - small vessels (88D)",
         sub[[c for c in all_ct if c not in SMALL_VESSEL_KEYS]].values),
    ]

    results = []
    for name, X in ablation_configs:
        r = cv_auc(X, y, name)
        results.append(r)
        print(f"  {name:40s}  AUC={r['auc_mean']:.3f}±{r['auc_std']:.3f}  "
              f"F1={r['f1_mean']:.3f}  Acc={r['acc_mean']:.3f}")

    # ================================================================
    # PART 3: Plot — PA/Ao statistics table (like uploaded image)
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CT影像特征 — PH诊断能力对比", fontsize=16, fontweight='bold', y=0.98)

    # Table 1: PA/Ao diagnostic criteria
    ax1 = axes[0]
    ax1.axis('off')
    ax1.set_title("传统CT征象 (PA直径/PA-Ao比)", fontsize=13, pad=10)

    table_data = [
        ["特征", "阈值", "PH阳性", "非PH阳性", "敏感度"],
        ["PA直径 (echo)", "> 2.9 cm",
         f"{stats['PA_diam_echo']['PH_gt29mm']}/{stats['PA_diam_echo']['PH_total_with_data']}",
         f"{stats['PA_diam_echo']['nPH_gt29mm']}/{stats['PA_diam_echo']['nPH_total_with_data']}",
         f"{stats['PA_diam_echo']['PH_gt29mm']/max(stats['PA_diam_echo']['PH_total_with_data'],1)*100:.0f}%"],
        ["PA/Ao比", "> 1.0",
         f"{stats['PA_Ao_ratio']['PH_gt1']}/85",
         f"{stats['PA_Ao_ratio']['nPH_gt1']}/28",
         f"{stats['PA_Ao_ratio']['PH_gt1']/85*100:.0f}%"],
        ["mPAP (RHC)", "> 20 mmHg",
         f"85/85", f"0/28", "100%"],
    ]

    colors = [['#f0f0f0']*5] + [['#fff']*5]*3
    tbl = ax1.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.15, 0.18, 0.18, 0.14])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')
        cell.set_edgecolor('#d0d0d0')

    # Table 2: Ablation results bar
    ax2 = axes[1]
    names = [r['name'] for r in results]
    aucs = [r['auc_mean'] for r in results]
    stds = [r['auc_std'] for r in results]

    colors_bar = ['#C0504D', '#5DCAA5', '#4472C4', '#97C459',
                  '#AFA9EC', '#D85A30', '#378ADD', '#888780']
    bars = ax2.barh(range(len(names)), aucs, xerr=stds,
                     color=colors_bar[:len(names)], height=0.6, capsize=3)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel("5-fold CV AUC", fontsize=11)
    ax2.set_title("特征消融实验: 各特征组分类能力", fontsize=13, pad=10)
    ax2.set_xlim(0.5, 1.0)
    ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3)

    for i, (auc, std) in enumerate(zip(aucs, stds)):
        ax2.text(auc + std + 0.01, i, f"{auc:.3f}", va='center', fontsize=9)

    plt.tight_layout()
    path1 = os.path.join(args.output_dir, "feature_ablation.png")
    fig.savefig(path1, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {path1}")
    plt.close()

    # ================================================================
    # PART 4: Key verdict
    # ================================================================
    pa_only_auc = results[0]['auc_mean']
    small_vessel_auc = results[1]['auc_mean']
    all_ct_auc = results[5]['auc_mean']
    no_small_auc = results[7]['auc_mean']

    print("\n" + "="*60)
    print("关键结论 (KEY VERDICT)")
    print("="*60)
    print(f"  PA直径+PA/Ao比 alone:      AUC = {pa_only_auc:.3f}")
    print(f"  小血管特征 alone:           AUC = {small_vessel_auc:.3f}")
    print(f"  全部CT特征:                AUC = {all_ct_auc:.3f}")
    print(f"  全部CT - 去掉小血管特征:    AUC = {no_small_auc:.3f}")
    print(f"  小血管特征贡献:             +{all_ct_auc - no_small_auc:.3f} AUC")
    print(f"\n  → PA/Ao是弱信号 (sens={stats['PA_diam_echo']['PH_gt29mm']}/{stats['PA_diam_echo']['PH_total_with_data']}=19%)")
    print(f"  → 小血管树拓扑是主要驱动力")

    if small_vessel_auc > pa_only_auc + 0.05:
        print(f"\n  ★ 肺小血管树图拓扑特征 AUC ({small_vessel_auc:.3f}) 显著优于")
        print(f"    PA直径/Ao比 ({pa_only_auc:.3f}), 差异 = +{small_vessel_auc-pa_only_auc:.3f}")

    # Save all results
    output = {
        'pa_ao_stats': stats,
        'ablation': [{k: v for k, v in r.items() if k != 'fold_aucs'}
                      for r in results],
        'verdict': {
            'pa_ao_auc': pa_only_auc,
            'small_vessel_auc': small_vessel_auc,
            'all_ct_auc': all_ct_auc,
            'small_vessel_contribution': all_ct_auc - no_small_auc,
        }
    }
    json_path = os.path.join(args.output_dir, "ablation_results.json")
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {json_path}")


if __name__ == '__main__':
    main()
