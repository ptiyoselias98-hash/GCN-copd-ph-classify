#!/usr/bin/env python3
"""Attribution analysis using CT v2 PA/Ao measurements (replaces echo).

Mirrors attribution_analysis.py but swaps the "PA diam + PA/Ao" feature matrix
from echo-derived values (2D, regex-parsed from 超声报告) to CT-derived values
(3D minor-axis on commercial artery masks, from measure_pa_aorta_v2.py).

Writes:
  outputs/attribution/ablation_results_ct.json
  outputs/attribution/feature_ablation_ct.png
"""
import argparse, json, os, sys, io, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

GLOBAL_VASC = [
    "肺血管容积(ml)_y","肺血管平均密度(HU)","肺血管最大密度(HU)","肺血管最小密度(HU)",
    "肺血管血管分支数量_y","肺血管弯曲度","肺血管分形维度",
    "肺血管BV5(ml)_y","肺血管BV10(ml)_y","肺血管BV10+(ml)_y",
    "动脉容积(ml)","动脉平均密度(HU)","动脉弯曲度","动脉分形维度",
    "动脉BV5(ml)_y","动脉BV10(ml)_y","动脉BV10+(ml)_y",
    "静脉容积(ml)_y","静脉平均密度(HU)","静脉弯曲度","静脉分形维度",
    "静脉BV5(ml)_y","静脉BV10(ml)_y","静脉BV10+(ml)_y",
    "动脉血管分支数量_y","静脉血管分支数量_y",
]
PARA_AIRWAY = [
    "左右肺容积(ml)","左右肺LAA910(%)","左右肺LAA950(%)",
    "左右肺平均密度(HU)","左右肺密度标准差(HU)","左右肺质量(g)",
    "左右肺支气管数量.1","左右肺支气管长度(cm).1","左右肺支气管体积(ml).1",
    "左右肺代","左右肺支气管数量[D<2mm]","左右肺支气管体积(ml)[D<2mm]",
    "左右肺Pi10(mm)","左右肺弯曲度","左右肺分形维度",
]
LOBE_COLS_PATTERN = lambda df: [c for c in df.columns if ("动脉(" in c or "静脉(" in c) and "肺叶" in c]
SMALL_VESSEL_KEYS = [
    "肺血管BV5(ml)_y","肺血管BV10(ml)_y",
    "肺血管血管分支数量_y","肺血管弯曲度","肺血管分形维度",
    "动脉BV5(ml)_y","动脉弯曲度","动脉分形维度",
    "静脉BV5(ml)_y","静脉弯曲度","静脉分形维度",
    "静脉血管分支数量_y","动脉血管分支数量_y",
]


def load_ct_measurements(path):
    """Return dict pinyin_lowercase -> (pa_mm, ao_mm, ratio)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for key, v in data["measurements"].items():
        parts = key.split("_")
        if len(parts) >= 2 and parts[0] in ("nonph", "ph"):
            pinyin = parts[1].lower()
        else:
            continue
        pa = v.get("pa_diameter_mm") or 0.0
        ao = v.get("ao_diameter_mm") or 0.0
        ratio = v.get("pa_ao_ratio")
        if ratio is None:
            ratio = pa / ao if ao > 0 else np.nan
        if pinyin not in out:
            out[pinyin] = (float(pa), float(ao), float(ratio) if ratio else np.nan)
    return out


def ct_features_for_rows(df, ct_map):
    """Return (Nx3) matrix of [pa_mm, ao_mm, ratio], NaN-filled to 0."""
    pa_vals, ao_vals, ratio_vals = [], [], []
    matched = 0
    for _, row in df.iterrows():
        fname = str(row.get("ct文件名", "")).strip()
        pinyin = fname.split("_")[0].lower() if fname else ""
        rec = ct_map.get(pinyin)
        if rec:
            pa_vals.append(rec[0])
            ao_vals.append(rec[1])
            ratio_vals.append(rec[2] if not np.isnan(rec[2]) else 0.0)
            matched += 1
        else:
            pa_vals.append(0.0)
            ao_vals.append(0.0)
            ratio_vals.append(0.0)
    return np.column_stack([pa_vals, ao_vals, ratio_vals]), matched


def cv_auc(X, y, name, n_splits=5, seed=42):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce").values
    X = np.nan_to_num(X.astype(float), nan=0.0)
    X_s = StandardScaler().fit_transform(X)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs, f1s, accs = [], [], []
    for tr, te in skf.split(X_s, y):
        rf = RandomForestClassifier(n_estimators=200, max_depth=8,
                                    class_weight="balanced", random_state=seed)
        rf.fit(X_s[tr], y[tr])
        prob = rf.predict_proba(X_s[te])[:, 1]
        pred = rf.predict(X_s[te])
        aucs.append(roc_auc_score(y[te], prob))
        f1s.append(f1_score(y[te], pred))
        accs.append(accuracy_score(y[te], pred))
    return {
        "name": name,
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "f1_mean": float(np.mean(f1s)),
        "acc_mean": float(np.mean(accs)),
    }


def safe_cols(df, cols):
    return [c for c in cols if c in df.columns]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xlsx", required=True)
    p.add_argument("--ct_json", required=True,
                   help="ct_pa_ao_measurements_v2.json from measure_pa_aorta_v2.py")
    p.add_argument("--output_dir", default="./outputs/attribution")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_excel(args.xlsx, sheet_name="Sheet1")
    mask_ct = df["肺血管容积(ml)_y"].notna()
    sub = df[mask_ct].copy()
    y = (sub["PH"] == "是").astype(int).values
    n_ph = int(y.sum()); n_nph = len(y) - n_ph

    print(f"Cohort: n={len(sub)} (PH={n_ph}, non-PH={n_nph})")

    # CT v2 measurements
    ct_map = load_ct_measurements(args.ct_json)
    print(f"CT v2 measurements: {len(ct_map)} unique pinyin keys")

    pa_ao_ct, matched = ct_features_for_rows(sub, ct_map)
    print(f"Cohort matched to CT v2: {matched}/{len(sub)}")

    # Per-group CT stats within the cohort
    is_ph = y == 1
    pa_ph = pa_ao_ct[is_ph, 0]; pa_nph = pa_ao_ct[~is_ph, 0]
    ratio_ph = pa_ao_ct[is_ph, 2]; ratio_nph = pa_ao_ct[~is_ph, 2]
    # only count rows with non-zero PA (matched)
    pa_ph_v = pa_ph[pa_ph > 0]; pa_nph_v = pa_nph[pa_nph > 0]
    ratio_ph_v = ratio_ph[ratio_ph > 0]; ratio_nph_v = ratio_nph[ratio_nph > 0]

    stats = {
        "method": "CT v2 minor-axis (measure_pa_aorta_v2.py)",
        "cohort_n": int(len(sub)),
        "matched_to_ct": int(matched),
        "PA_diam_ct": {
            "threshold_mm": 29.0,
            "PH_mean_mm": float(pa_ph_v.mean()) if len(pa_ph_v) else None,
            "PH_std_mm": float(pa_ph_v.std()) if len(pa_ph_v) else None,
            "nPH_mean_mm": float(pa_nph_v.mean()) if len(pa_nph_v) else None,
            "nPH_std_mm": float(pa_nph_v.std()) if len(pa_nph_v) else None,
            "PH_gt29mm": int((pa_ph_v > 29).sum()),
            "nPH_gt29mm": int((pa_nph_v > 29).sum()),
            "PH_total_with_data": int(len(pa_ph_v)),
            "nPH_total_with_data": int(len(pa_nph_v)),
        },
        "PA_Ao_ratio_ct": {
            "threshold": 1.0,
            "PH_mean": float(ratio_ph_v.mean()) if len(ratio_ph_v) else None,
            "PH_std": float(ratio_ph_v.std()) if len(ratio_ph_v) else None,
            "nPH_mean": float(ratio_nph_v.mean()) if len(ratio_nph_v) else None,
            "nPH_std": float(ratio_nph_v.std()) if len(ratio_nph_v) else None,
            "PH_gt1": int((ratio_ph_v > 1).sum()),
            "nPH_gt1": int((ratio_nph_v > 1).sum()),
            "PH_total_with_data": int(len(ratio_ph_v)),
            "nPH_total_with_data": int(len(ratio_nph_v)),
        },
    }
    print("\nCT v2 stats (cohort):")
    print(f"  PH  : PA={stats['PA_diam_ct']['PH_mean_mm']:.2f}±"
          f"{stats['PA_diam_ct']['PH_std_mm']:.2f} mm, "
          f"ratio={stats['PA_Ao_ratio_ct']['PH_mean']:.3f}±"
          f"{stats['PA_Ao_ratio_ct']['PH_std']:.3f}")
    print(f"  nPH : PA={stats['PA_diam_ct']['nPH_mean_mm']:.2f}±"
          f"{stats['PA_diam_ct']['nPH_std_mm']:.2f} mm, "
          f"ratio={stats['PA_Ao_ratio_ct']['nPH_mean']:.3f}±"
          f"{stats['PA_Ao_ratio_ct']['nPH_std']:.3f}")

    lobe_cols = LOBE_COLS_PATTERN(sub)
    all_ct = safe_cols(sub, GLOBAL_VASC + PARA_AIRWAY) + lobe_cols

    ablation_configs = [
        ("PA diam + PA/Ao only (CT, 3D)", pa_ao_ct),
        ("Small vessel features (13D)", sub[safe_cols(sub, SMALL_VESSEL_KEYS)].values),
        ("Global vascular (26D)", sub[safe_cols(sub, GLOBAL_VASC)].values),
        ("Parenchyma + airway (15D)", sub[safe_cols(sub, PARA_AIRWAY)].values),
        ("Lobe-level vascular (60D)",
         sub[lobe_cols].values if lobe_cols else np.zeros((len(sub), 1))),
        ("All CT features (101D)", sub[all_ct].values),
        ("All CT + PA/Ao (CT, 104D)", np.hstack([sub[all_ct].values, pa_ao_ct])),
        ("All CT - small vessels (88D)",
         sub[[c for c in all_ct if c not in SMALL_VESSEL_KEYS]].values),
    ]

    results = []
    for name, X in ablation_configs:
        r = cv_auc(X, y, name)
        results.append(r)
        print(f"  {name:38s}  AUC={r['auc_mean']:.3f}±{r['auc_std']:.3f}  "
              f"F1={r['f1_mean']:.3f}  Acc={r['acc_mean']:.3f}")

    # ---- Figure ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CT影像特征 — PH诊断能力对比 (PA/Ao 来源: CT v2 minor-axis)",
                 fontsize=15, fontweight="bold", y=0.99)

    ax1 = axes[0]
    ax1.axis("off")
    ax1.set_title("CT实测 PA/Ao 指标 (vs 直接CT测量)", fontsize=12, pad=10)
    s_pa = stats["PA_diam_ct"]; s_r = stats["PA_Ao_ratio_ct"]
    table_data = [
        ["特征", "阈值", "PH阳性", "非PH阳性", "敏感度"],
        ["PA直径 (CT v2)", "> 29 mm",
         f"{s_pa['PH_gt29mm']}/{s_pa['PH_total_with_data']}",
         f"{s_pa['nPH_gt29mm']}/{s_pa['nPH_total_with_data']}",
         f"{s_pa['PH_gt29mm']/max(s_pa['PH_total_with_data'],1)*100:.0f}%"],
        ["PA/Ao比 (CT)", "> 1.0",
         f"{s_r['PH_gt1']}/{s_r['PH_total_with_data']}",
         f"{s_r['nPH_gt1']}/{s_r['nPH_total_with_data']}",
         f"{s_r['PH_gt1']/max(s_r['PH_total_with_data'],1)*100:.0f}%"],
        ["mPAP (RHC)", "> 20 mmHg", f"{n_ph}/{n_ph}", f"0/{n_nph}", "100%"],
    ]
    tbl = ax1.table(cellText=table_data, cellLoc="center", loc="center",
                    colWidths=[0.28, 0.15, 0.18, 0.18, 0.14])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.8)
    for (row, _col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#d0d0d0")

    ax2 = axes[1]
    names = [r["name"] for r in results]
    aucs = [r["auc_mean"] for r in results]
    stds = [r["auc_std"] for r in results]
    colors_bar = ["#C0504D","#5DCAA5","#4472C4","#97C459",
                  "#AFA9EC","#D85A30","#378ADD","#888780"]
    ax2.barh(range(len(names)), aucs, xerr=stds,
             color=colors_bar[:len(names)], height=0.6, capsize=3)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel("5-fold CV AUC", fontsize=11)
    ax2.set_title("特征消融实验 (PA/Ao 来自 CT v2)", fontsize=12, pad=10)
    ax2.set_xlim(0.4, 1.0)
    ax2.axvline(x=0.5, color="gray", linestyle="--", alpha=0.3)
    for i, (auc, std) in enumerate(zip(aucs, stds)):
        ax2.text(auc + std + 0.01, i, f"{auc:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    png_path = os.path.join(args.output_dir, "feature_ablation_ct.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {png_path}")

    pa_only_auc = results[0]["auc_mean"]
    small_vessel_auc = results[1]["auc_mean"]
    all_ct_auc = results[5]["auc_mean"]
    no_small_auc = results[7]["auc_mean"]

    print("\n" + "=" * 60)
    print("关键结论 (CT v2 版本)")
    print("=" * 60)
    print(f"  PA直径+PA/Ao比 (CT) alone: AUC = {pa_only_auc:.3f}")
    print(f"  小血管特征 alone:           AUC = {small_vessel_auc:.3f}")
    print(f"  全部CT特征:                AUC = {all_ct_auc:.3f}")
    print(f"  全部CT - 去掉小血管:        AUC = {no_small_auc:.3f}")

    output = {
        "pa_ao_stats_ct": stats,
        "ablation": results,
        "verdict": {
            "pa_ao_ct_auc": pa_only_auc,
            "small_vessel_auc": small_vessel_auc,
            "all_ct_auc": all_ct_auc,
            "small_vessel_contribution": all_ct_auc - no_small_auc,
        },
    }
    json_path = os.path.join(args.output_dir, "ablation_results_ct.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
