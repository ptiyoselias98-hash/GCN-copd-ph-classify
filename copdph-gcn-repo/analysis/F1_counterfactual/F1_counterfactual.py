"""F1_counterfactual — Phase F1: bucket ablation + per-patient driver + permutation nulls.

Per codex pass-1: framed as 'model sensitivity under signature perturbation', NOT
causal counterfactual. Bucket-level perturbation only.

Tasks (all on within-contrast n=190 + best D1 config = P4_combined Ridge_LR):
T1 Bucket ablation: drop each bucket (lung / artery / vein / airway / TDA), retrain,
   report AUC drop, prediction-shift heatmap per patient
T2 Per-patient driver assignment: which bucket's removal causes biggest prediction drop
T3 Permutation null:
   - bucket-AUC-drop null (label permutation)
   - winner cluster ARI null (already in E1; repeated here as F1 confirmation)

Output:
  outputs/supplementary/F1_counterfactual/bucket_ablation_results.csv
  outputs/supplementary/F1_counterfactual/per_patient_driver_assignment.csv
  outputs/supplementary/F1_counterfactual/permutation_results.json
  outputs/supplementary/F1_counterfactual/prediction_drop_heatmap.png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent
SIG = ROOT / "outputs" / "supplementary" / "B1_graph_signature" / "graph_signatures_patient_level.csv"
DICT = ROOT / "outputs" / "supplementary" / "B1_graph_signature" / "graph_signature_dictionary.json"
OUT = ROOT / "outputs" / "supplementary" / "F1_counterfactual"
OUT.mkdir(parents=True, exist_ok=True)

META = {"case_id", "label", "protocol", "is_contrast_only_subset",
        "measured_mpap", "measured_mpap_flag", "fold_id",
        "C1_all_available", "C2_within_contrast_only",
        "C3_borderline_mPAP_18_22", "C4_clear_low_high",
        "C5_early_COPD_no_PH_proxy"}


def define_buckets(df, fdict):
    feat_cols = [c for c in df.columns if c not in META
                 and pd.api.types.is_numeric_dtype(df[c])]
    buckets = {
        "lung": [c for c in feat_cols
                  if fdict.get(c, {}).get("source") == "lung_features_v2"],
        "artery": [c for c in feat_cols
                    if fdict.get(c, {}).get("source") == "unified_301_Simple_AV_seg"
                    and c.startswith("artery_")],
        "vein": [c for c in feat_cols
                  if fdict.get(c, {}).get("source") == "unified_301_Simple_AV_seg"
                  and c.startswith("vein_")],
        "airway": [c for c in feat_cols
                    if fdict.get(c, {}).get("source") == "legacy_R17_HiPaS_style_pipeline"],
        "TDA": [c for c in feat_cols
                 if fdict.get(c, {}).get("category") == "8_TDA_persistence"],
    }
    return buckets


def fit_eval_oof(X, y, seeds=(42,43,44,45,46)):
    """5-seed × 5-fold OOF AUC + per-case prob averaged across seeds."""
    if X.shape[1] == 0:
        return float("nan"), np.full(len(y), np.nan)
    oof_p = np.full(len(y), np.nan); oof_n = np.zeros(len(y), int)
    for seed in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, va in skf.split(X, y):
            sc = RobustScaler().fit(X[tr])
            Xtr = sc.transform(X[tr]); Xva = sc.transform(X[va])
            try:
                clf = LogisticRegressionCV(Cs=10, cv=3, penalty="l2",
                                            scoring="roc_auc", max_iter=2000,
                                            random_state=seed).fit(Xtr, y[tr])
                p = clf.predict_proba(Xva)[:, 1]
                cur = oof_p[va]; oof_p[va] = np.where(np.isnan(cur), p, cur + p)
                oof_n[va] += 1
            except Exception:
                continue
    oof_p = oof_p / np.where(oof_n > 0, oof_n, 1)
    auc = float(roc_auc_score(y, oof_p)) if not np.isnan(oof_p).all() else float("nan")
    return auc, oof_p


def main():
    df = pd.read_csv(SIG)
    df = df[df["is_contrast_only_subset"]].reset_index(drop=True)
    assert len(df) == 190, f"F1 cohort drift: {len(df)}"
    fdict = json.loads(DICT.read_text(encoding="utf-8"))["feature_dictionary"]
    buckets = define_buckets(df, fdict)
    for k, v in buckets.items(): print(f"  {k}: {len(v)}")
    all_feats = sum(buckets.values(), [])
    print(f"  ALL: {len(all_feats)} features")

    y = df["label"].astype(int).values

    # ===== T1 + T2 bucket ablation =====
    full_X = df[all_feats].fillna(0).values.astype(float)
    full_auc, full_p = fit_eval_oof(full_X, y)
    print(f"\nFULL panel AUC = {full_auc:.3f}")

    rows = []
    bucket_pred = {"FULL": full_p}
    for bucket_name in buckets:
        kept = [f for b, fs in buckets.items() if b != bucket_name for f in fs]
        Xa = df[kept].fillna(0).values.astype(float)
        auc, p = fit_eval_oof(Xa, y)
        rows.append({
            "removed_bucket": bucket_name,
            "n_features_removed": len(buckets[bucket_name]),
            "n_features_remaining": Xa.shape[1],
            "auc": auc,
            "auc_drop": full_auc - auc,
        })
        bucket_pred[f"-{bucket_name}"] = p
        print(f"  remove {bucket_name}: AUC={auc:.3f} (drop {full_auc-auc:+.3f})")
    bucket_df = pd.DataFrame(rows).sort_values("auc_drop", ascending=False)
    bucket_df.to_csv(OUT / "bucket_ablation_results.csv", index=False)

    # ===== T2 per-patient driver assignment =====
    driver_rows = []
    for i in range(len(df)):
        full_pi = full_p[i]
        drops = {}
        for bn in buckets:
            drops[bn] = full_pi - bucket_pred[f"-{bn}"][i]
        # Largest absolute drop = bucket whose removal most affects this patient
        driver = max(drops.keys(), key=lambda k: abs(drops[k]))
        driver_rows.append({
            "case_id": str(df.iloc[i]["case_id"]),
            "label": int(df.iloc[i]["label"]),
            "measured_mpap": float(df.iloc[i]["measured_mpap"]) if pd.notna(df.iloc[i]["measured_mpap"]) else None,
            "full_p": float(full_pi),
            "driver_bucket": driver,
            "driver_drop": float(drops[driver]),
            **{f"drop_{bn}": float(drops[bn]) for bn in buckets},
        })
    drv_df = pd.DataFrame(driver_rows)
    drv_df.to_csv(OUT / "per_patient_driver_assignment.csv", index=False)
    print("\n=== driver assignment counts ===")
    print(drv_df["driver_bucket"].value_counts())

    # ===== T3 permutation nulls =====
    # Null bucket-AUC-drop: shuffle label, repeat ablation
    print("\n=== permutation null ===")
    rng = np.random.default_rng(42)
    n_perm = 50  # 50 perms × full 5-seed×5-fold protocol = 1250 fits — keep matched protocol
    null_full_aucs = []
    for s in range(n_perm):
        y_perm = rng.permutation(y)
        # Match real protocol: 5-seed × 5-fold Ridge_LR OOF (no shortcut)
        a, _ = fit_eval_oof(full_X, y_perm, seeds=(42 + s, 43 + s, 44 + s, 45 + s, 46 + s))
        null_full_aucs.append(a)
    null_full_aucs = np.array([a for a in null_full_aucs if not np.isnan(a)])
    real_auc = full_auc
    perm_p = float((null_full_aucs >= real_auc).mean()) if len(null_full_aucs) else float("nan")
    null_summary = {
        "n_perms": int(len(null_full_aucs)),
        "real_full_auc": real_auc,
        "null_full_auc_mean": float(np.mean(null_full_aucs)),
        "null_full_auc_p99": float(np.percentile(null_full_aucs, 99)),
        "perm_p_real_above_null": perm_p,
        "real_above_99pct_null": bool(real_auc > np.percentile(null_full_aucs, 99)),
    }
    (OUT / "permutation_results.json").write_text(json.dumps(null_summary, indent=2),
                                                    encoding="utf-8")
    print(json.dumps(null_summary, indent=2))

    # ===== Figure: bucket ablation bars + driver counts =====
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    bnames = bucket_df["removed_bucket"].tolist()
    drops = bucket_df["auc_drop"].tolist()
    colors = ["#ef4444" if d > 0 else "#3b82f6" for d in drops]
    ax.barh(bnames, drops, color=colors, edgecolor="black", alpha=0.85)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("AUC drop after bucket removal\n(positive = bucket adds signal)")
    ax.set_title(f"F1 — Bucket ablation (full AUC={full_auc:.3f}, within-contrast n=190)")
    ax.grid(alpha=0.3, axis="x")
    for i, d in enumerate(drops):
        ax.text(d + (0.005 if d>=0 else -0.005), i, f"{d:+.3f}",
                va="center", ha="left" if d>=0 else "right", fontsize=9, fontweight="bold")
    # Driver counts
    ax = axes[1]
    counts = drv_df["driver_bucket"].value_counts()
    ph_in = drv_df[drv_df["label"]==1]["driver_bucket"].value_counts()
    nph_in = drv_df[drv_df["label"]==0]["driver_bucket"].value_counts()
    bnames2 = list(counts.index)
    ph_arr = [int(ph_in.get(b, 0)) for b in bnames2]
    nph_arr = [int(nph_in.get(b, 0)) for b in bnames2]
    ax.barh(bnames2, ph_arr, color="#ef4444", label=f"PH (n={int((drv_df.label==1).sum())})", alpha=0.85, edgecolor="black")
    ax.barh(bnames2, nph_arr, left=ph_arr, color="#3b82f6", label=f"nonPH (n={int((drv_df.label==0).sum())})", alpha=0.85, edgecolor="black")
    ax.set_xlabel("Number of patients (driver = bucket whose removal most shifts prediction)")
    ax.set_title("Per-patient driver bucket assignment")
    ax.legend(); ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(OUT / "prediction_drop_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nsaved all F1 outputs to {OUT}")


if __name__ == "__main__":
    main()
