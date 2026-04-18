"""Compare PA/Ao (from masks) vs small-vessel-tree topology as PH predictors.

Uses the per-case measurements in data/pa_ao_measurements.csv (produced by
compute_pa_ao_from_masks.py) plus radiomics + xlsx labels.

Outputs:
  outputs/pa_ao_vs_tree/cohort_stats.json      — PA/Ao distributions by PH
  outputs/pa_ao_vs_tree/ablation.json           — 5-fold RF AUC for 4 configs
  outputs/pa_ao_vs_tree/comparison.png          — bar chart
  outputs/pa_ao_vs_tree/per_case.csv            — merged table

Verdict (printed): which group contributes more to classification —
the mask-derived PA/Ao ratio, or the small-vessel tree features.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    from run_hybrid import case_to_pinyin
except Exception:
    def case_to_pinyin(n: str) -> str:
        return "".join(ch for ch in str(n) if ord(ch) < 128)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("pa_ao_vs_tree")

SMALL_VESSEL_KEYS = [
    "肺血管BV5(ml)_y", "肺血管BV10(ml)_y",
    "肺血管血管分支数量_y", "肺血管弯曲度", "肺血管分形维度",
    "动脉BV5(ml)_y", "动脉弯曲度", "动脉分形维度",
    "静脉BV5(ml)_y", "静脉弯曲度", "静脉分形维度",
    "静脉血管分支数量_y", "动脉血管分支数量_y",
]
METRICS = ["AUC", "Accuracy", "F1", "Sensitivity", "Specificity"]


def cv_metrics(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
    X = np.nan_to_num(X.astype(float), nan=0.0)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X_s = StandardScaler().fit_transform(X)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs, f1s, accs, sens, specs = [], [], [], [], []
    for tr, te in skf.split(X_s, y):
        rf = RandomForestClassifier(200, max_depth=8, class_weight="balanced",
                                     random_state=42)
        rf.fit(X_s[tr], y[tr])
        p = rf.predict_proba(X_s[te])[:, 1]
        pred = (p >= 0.5).astype(int)
        aucs.append(roc_auc_score(y[te], p) if len(set(y[te])) > 1 else np.nan)
        f1s.append(f1_score(y[te], pred, zero_division=0))
        accs.append(accuracy_score(y[te], pred))
        tp = ((pred == 1) & (y[te] == 1)).sum()
        fn = ((pred == 0) & (y[te] == 1)).sum()
        tn = ((pred == 0) & (y[te] == 0)).sum()
        fp = ((pred == 1) & (y[te] == 0)).sum()
        sens.append(tp / max(tp + fn, 1))
        specs.append(tn / max(tn + fp, 1))
    return {
        "AUC": float(np.nanmean(aucs)), "AUC_std": float(np.nanstd(aucs)),
        "Accuracy": float(np.mean(accs)), "F1": float(np.mean(f1s)),
        "Sensitivity": float(np.mean(sens)), "Specificity": float(np.mean(specs)),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--xlsx", required=True)
    p.add_argument("--pa_ao_csv", required=True)
    p.add_argument("--radiomics", default="./data/copd_ph_radiomics.csv")
    p.add_argument("--output_dir", default="./outputs/pa_ao_vs_tree")
    p.add_argument("--name_col", default="姓名")
    args = p.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    df_x = pd.read_excel(args.xlsx, sheet_name="Sheet1")
    name_col = args.name_col if args.name_col in df_x.columns else df_x.columns[0]
    df_x["patient_id"] = df_x[name_col].astype(str).map(case_to_pinyin)
    df_x["PH"] = (df_x["PH"] == "是").astype(int)

    df_pa = pd.read_csv(args.pa_ao_csv)
    # Try to align keys: if case_id already pinyin, direct merge works;
    # otherwise attempt case_to_pinyin on case_id
    if not df_pa["case_id"].isin(df_x["patient_id"]).any():
        df_pa["case_id"] = df_pa["case_id"].astype(str).map(case_to_pinyin)
    merged = df_x.merge(df_pa, left_on="patient_id", right_on="case_id",
                        how="inner")
    logger.info("merged %d cases (xlsx=%d, pa_ao=%d)",
                len(merged), len(df_x), len(df_pa))

    # Optionally enrich with radiomics for small-vessel features
    try:
        df_r = pd.read_csv(args.radiomics)
        merged = merged.merge(df_r, on="patient_id", how="left",
                              suffixes=("", "_rad"))
    except Exception as e:
        logger.warning("radiomics merge failed: %s", e)

    merged.to_csv(out / "per_case.csv", index=False, encoding="utf-8-sig")

    y = merged["PH"].values
    pa = pd.to_numeric(merged.get("pa_diam_mm"), errors="coerce").values
    ao = pd.to_numeric(merged.get("ao_diam_mm"), errors="coerce").values
    ratio = pd.to_numeric(merged.get("pa_ao_ratio"), errors="coerce").values

    # --- Cohort stats
    stats = {
        "n_total": int(len(merged)),
        "n_PH": int((y == 1).sum()),
        "n_nonPH": int((y == 0).sum()),
        "pa_diam_mm": {
            "PH_mean": float(np.nanmean(pa[y == 1])),
            "PH_std":  float(np.nanstd(pa[y == 1])),
            "nPH_mean": float(np.nanmean(pa[y == 0])),
            "nPH_std":  float(np.nanstd(pa[y == 0])),
            "PH_gt29mm": int(np.nansum((pa > 29) & (y == 1))),
            "nPH_gt29mm": int(np.nansum((pa > 29) & (y == 0))),
        },
        "pa_ao_ratio": {
            "PH_mean": float(np.nanmean(ratio[y == 1])),
            "PH_std":  float(np.nanstd(ratio[y == 1])),
            "nPH_mean": float(np.nanmean(ratio[y == 0])),
            "nPH_std":  float(np.nanstd(ratio[y == 0])),
            "PH_gt1": int(np.nansum((ratio > 1) & (y == 1))),
            "nPH_gt1": int(np.nansum((ratio > 1) & (y == 0))),
        },
    }
    (out / "cohort_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("cohort stats: %s", json.dumps(stats, indent=2, ensure_ascii=False))

    # --- Ablation: 5-fold AUC under 4 feature sets
    small = [c for c in SMALL_VESSEL_KEYS if c in merged.columns]
    X_pa_alone = np.column_stack([
        np.nan_to_num(pa), np.nan_to_num(ao), np.nan_to_num(ratio)
    ])
    X_tree_alone = merged[small].apply(pd.to_numeric, errors="coerce").values \
        if small else np.zeros((len(merged), 1))
    X_both = np.hstack([X_tree_alone, X_pa_alone])
    X_tree_no_pa = X_tree_alone

    ablation = {}
    for name, X in [
        ("PA_Ao_only", X_pa_alone),
        ("SmallVesselTree_only", X_tree_alone),
        ("Combined", X_both),
        ("Tree_no_PA_Ao", X_tree_no_pa),
    ]:
        ablation[name] = cv_metrics(X, y)
        logger.info("%-24s AUC=%.3f Acc=%.3f F1=%.3f Sens=%.3f Spec=%.3f",
                    name, ablation[name]["AUC"], ablation[name]["Accuracy"],
                    ablation[name]["F1"], ablation[name]["Sensitivity"],
                    ablation[name]["Specificity"])
    (out / "ablation.json").write_text(
        json.dumps(ablation, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(ablation.keys())
    x = np.arange(len(names)); w = 0.18
    for i, k in enumerate(METRICS):
        vals = [ablation[n][k] for n in names]
        ax.bar(x + (i - 2) * w, vals, w, label=k)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=10, fontsize=9)
    ax.set_ylim(0, 1.0); ax.legend(loc="lower right", fontsize=8)
    ax.set_title("PA/Ao (mask-derived) vs Small-Vessel Tree — 5-fold RF")
    ax.axhline(0.8, color="gray", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(out / "comparison.png", dpi=150, bbox_inches="tight")

    # --- Verdict
    pa_auc = ablation["PA_Ao_only"]["AUC"]
    tree_auc = ablation["SmallVesselTree_only"]["AUC"]
    both_auc = ablation["Combined"]["AUC"]
    print("\n=== VERDICT ===")
    print(f"  PA/Ao alone:           AUC = {pa_auc:.3f}")
    print(f"  Small-vessel tree:     AUC = {tree_auc:.3f}")
    print(f"  Combined:              AUC = {both_auc:.3f}")
    print(f"  Tree advantage:        +{tree_auc - pa_auc:+.3f}")
    if tree_auc > pa_auc + 0.05:
        print("  → Small-vessel tree topology is the dominant driver.")
    elif pa_auc > tree_auc + 0.05:
        print("  → PA/Ao dominates; tree adds little.")
    else:
        print("  → Both signals are comparable; combined is best.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
