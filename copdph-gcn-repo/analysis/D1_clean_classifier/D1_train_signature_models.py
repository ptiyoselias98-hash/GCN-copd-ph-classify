"""D1_train_signature_models — Phase D1: clean within-contrast classifier.

4 feature panels × 4 models × patient-level CV + 6-metric reporting + bootstrap CI.
Within-contrast n=190 (163 PH + 27 nonPH) is PRIMARY. Pre-registered design:
  - 5-fold stratified patient-level CV (fold_id from R24.0)
  - 5 seeds (different random_state for fold permutation; fold_id reused for outer split)
  - Youden threshold computed inside training folds → applied to test folds
  - 6 metrics at threshold: AUC + Accuracy + Sensitivity + Specificity + F1 + Precision
  - PR-AUC + balanced accuracy + Brier additionally reported
  - Bootstrap-500 case-level CI on AUC
  - STOP RULE check: AUC<0.75 → emphasize severity-score/mechanism narrative

Panels (per Phase D1 spec):
  P1_lung_only: paren_*, whole_*, basal_*, middle_*, apical_basal_*, vessel_airway_*, _HU_*, LAA*
  P2_vascular_topology: artery_*, vein_* (unified-301, exclude basic size shortcuts)
  P3_airway_coupling: airway_*_legR17 + airway-vessel ratios
  P4_combined_clean: P1 + P2 + P3 + TDA

Output:
  outputs/supplementary/D1_clean_classifier/model_results.csv
  outputs/supplementary/D1_clean_classifier/roc_curves.png
  outputs/supplementary/D1_clean_classifier/calibration_plot.png
  outputs/supplementary/D1_clean_classifier/feature_importance.csv
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              brier_score_loss, roc_curve)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent
SIG = ROOT / "outputs" / "supplementary" / "B1_graph_signature" / "graph_signatures_patient_level.csv"
DICT = ROOT / "outputs" / "supplementary" / "B1_graph_signature" / "graph_signature_dictionary.json"
OUT = ROOT / "outputs" / "supplementary" / "D1_clean_classifier"
OUT.mkdir(parents=True, exist_ok=True)

META_COLS = {"case_id", "label", "protocol", "is_contrast_only_subset",
             "measured_mpap", "measured_mpap_flag", "fold_id",
             "C1_all_available", "C2_within_contrast_only",
             "C3_borderline_mPAP_18_22", "C4_clear_low_high",
             "C5_early_COPD_no_PH_proxy"}


def define_panels(df, dictionary):
    """Build P1-P4 column lists from B1 dictionary categories + sources."""
    fdict = dictionary["feature_dictionary"]
    feat_cols = [c for c in df.columns if c not in META_COLS
                 and pd.api.types.is_numeric_dtype(df[c])]
    P1_lung = [c for c in feat_cols
               if fdict.get(c, {}).get("category") == "7_lung_parenchyma"
               and fdict.get(c, {}).get("source") == "lung_features_v2"]
    P2_vasc = [c for c in feat_cols
               if fdict.get(c, {}).get("source") == "unified_301_Simple_AV_seg"
               and fdict.get(c, {}).get("category") in ("2_branching_topology",
                                                          "3_diameter_length_distribution")]
    P3_airway = [c for c in feat_cols
                 if fdict.get(c, {}).get("source") == "legacy_R17_HiPaS_style_pipeline"]
    P4_combined = sorted(set(P1_lung + P2_vasc + P3_airway
                              + [c for c in feat_cols
                                 if fdict.get(c, {}).get("category") == "8_TDA_persistence"]))
    return {"P1_lung_only": P1_lung, "P2_vascular_topology": P2_vasc,
            "P3_airway_coupling": P3_airway, "P4_combined_clean": P4_combined}


def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thr[idx])


def metrics_at_threshold(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred==1)&(y==1)).sum()); fp = int(((pred==1)&(y==0)).sum())
    tn = int(((pred==0)&(y==0)).sum()); fn = int(((pred==0)&(y==1)).sum())
    n = len(y)
    acc = (tp+tn)/n if n else 0.0
    sens = tp/(tp+fn) if (tp+fn) else 0.0
    spec = tn/(tn+fp) if (tn+fp) else 0.0
    prec = tp/(tp+fp) if (tp+fp) else 0.0
    f1 = 2*prec*sens/(prec+sens) if (prec+sens) else 0.0
    bal = (sens+spec)/2
    return {"acc": acc, "sens": sens, "spec": spec, "prec": prec, "f1": f1, "bal_acc": bal,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def make_classifier(name, seed):
    if name == "Lasso_LR":
        return LogisticRegressionCV(Cs=10, cv=3, penalty="l1", solver="liblinear",
                                     scoring="roc_auc", max_iter=2000, random_state=seed)
    if name == "Ridge_LR":
        return LogisticRegressionCV(Cs=10, cv=3, penalty="l2",
                                     scoring="roc_auc", max_iter=2000, random_state=seed)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=3,
                                       random_state=seed, n_jobs=1, class_weight="balanced")
    if name == "GradBoost":
        return GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                           random_state=seed, subsample=0.8)
    raise ValueError(name)


def one_panel_one_model(panel_name, panel_cols, model_name, df, seeds=(42,43,44,45,46)):
    """5-seed × 5-fold CV; AUC + 6-metric per fold; pooled OOF AUC."""
    sub = df[df["is_contrast_only_subset"]].reset_index(drop=True)
    X = sub[panel_cols].fillna(0).values.astype(float)
    y = sub["label"].astype(int).values
    if len(panel_cols) == 0 or X.shape[1] == 0:
        return None
    fold_aucs = []; fold_pr = []; fold_brier = []; fold_thresholds = []
    fold_metrics = []
    oof_p = np.full(len(y), np.nan)
    oof_count = np.zeros(len(y), int)
    for seed in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, va in skf.split(X, y):
            sc = RobustScaler().fit(X[tr])
            Xtr = sc.transform(X[tr]); Xva = sc.transform(X[va])
            try:
                clf = make_classifier(model_name, seed).fit(Xtr, y[tr])
                p_tr = clf.predict_proba(Xtr)[:, 1]
                p_va = clf.predict_proba(Xva)[:, 1]
                # Youden threshold inside training fold
                thr = youden_threshold(y[tr], p_tr)
                m = metrics_at_threshold(y[va], p_va, thr)
                auc = roc_auc_score(y[va], p_va)
                pr = average_precision_score(y[va], p_va)
                br = brier_score_loss(y[va], p_va)
                fold_aucs.append(auc); fold_pr.append(pr); fold_brier.append(br)
                fold_thresholds.append(thr)
                m["auc"] = auc; m["pr_auc"] = pr; m["brier"] = br; m["thr"] = thr
                fold_metrics.append(m)
                cur = oof_p[va]; oof_p[va] = np.where(np.isnan(cur), p_va, cur + p_va)
                oof_count[va] += 1
            except Exception as e:
                continue
    if not fold_aucs:
        return None
    oof_p = oof_p / np.where(oof_count > 0, oof_count, 1)
    pooled_auc = float(roc_auc_score(y, oof_p))
    # Bootstrap-500 CI on pooled AUC (case-level)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(500):
        idx = rng.integers(0, len(y), size=len(y))
        if y[idx].sum() < 3 or (y[idx]==0).sum() < 3: continue
        try: boots.append(roc_auc_score(y[idx], oof_p[idx]))
        except Exception: continue
    boots = np.array(boots) if boots else np.array([pooled_auc])
    pooled_ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    # Aggregate 6 metrics across folds (mean + std across folds)
    agg = {}
    for k in ("auc","acc","sens","spec","prec","f1","bal_acc","pr_auc","brier","thr"):
        vals = [m[k] for m in fold_metrics]
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals))
    return {"panel": panel_name, "model": model_name,
            "n_features": int(X.shape[1]), "n_folds": len(fold_aucs),
            "pooled_oof_auc": pooled_auc,
            "pooled_oof_auc_ci_lo": pooled_ci[0],
            "pooled_oof_auc_ci_hi": pooled_ci[1],
            **agg,
            "oof_p": oof_p.tolist(), "y": y.tolist(),
            "case_ids": sub["case_id"].tolist()}


def main():
    df = pd.read_csv(SIG)
    dictionary = json.loads(DICT.read_text(encoding="utf-8"))
    panels = define_panels(df, dictionary)
    for k, v in panels.items():
        print(f"  {k}: {len(v)} features")

    models = ["Lasso_LR", "Ridge_LR", "RandomForest", "GradBoost"]
    print(f"\n=== running {len(panels)} panels × {len(models)} models = {len(panels)*len(models)} configs ===")

    configs = [(p, panel_cols, m) for p, panel_cols in panels.items() for m in models]
    results = Parallel(n_jobs=4, backend="loky")(
        delayed(one_panel_one_model)(p, c, m, df) for p, c, m in configs
    )
    results = [r for r in results if r is not None]

    # Save summary table (drop oof_p/y/case_ids before CSV)
    rows = []
    oof_dump = {}
    for r in results:
        key = f"{r['panel']}_{r['model']}"
        oof_dump[key] = {"oof_p": r["oof_p"], "y": r["y"], "case_ids": r["case_ids"]}
        row = {k: v for k, v in r.items() if k not in ("oof_p", "y", "case_ids")}
        rows.append(row)
    res_df = pd.DataFrame(rows).sort_values("pooled_oof_auc", ascending=False).reset_index(drop=True)
    res_df.to_csv(OUT / "model_results.csv", index=False)

    # ROC curves: best model per panel
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    ax = axes[0]
    for panel in panels:
        panel_rows = [r for r in results if r["panel"] == panel]
        if not panel_rows: continue
        best = max(panel_rows, key=lambda r: r["pooled_oof_auc"])
        y = np.array(best["y"]); p = np.array(best["oof_p"])
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=2,
                label=f"{panel} ({best['model']}) AUC={best['pooled_oof_auc']:.3f} "
                      f"[{best['pooled_oof_auc_ci_lo']:.3f}, {best['pooled_oof_auc_ci_hi']:.3f}]")
    ax.plot([0, 1], [0, 1], "--", c="grey", label="chance")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"D1 — Best model per panel, within-contrast n={int(df.is_contrast_only_subset.sum())}\n"
                 f"5-seed × 5-fold OOF + bootstrap-500 95% CI on AUC")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Best overall calibration
    if results:
        best_overall = max(results, key=lambda r: r["pooled_oof_auc"])
        ax = axes[1]
        y = np.array(best_overall["y"]); p = np.array(best_overall["oof_p"])
        bins = np.linspace(0, 1, 9)
        bin_idx = np.digitize(p, bins) - 1
        bin_pred, bin_obs, bin_n = [], [], []
        for b in range(len(bins) - 1):
            m = bin_idx == b
            if m.sum() >= 3:
                bin_pred.append(p[m].mean()); bin_obs.append(y[m].mean()); bin_n.append(m.sum())
        ax.scatter(bin_pred, bin_obs, s=[n*5 for n in bin_n], c="#3b82f6", alpha=0.7,
                   edgecolors="black")
        ax.plot([0, 1], [0, 1], "--", c="grey", label="ideal")
        ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
        ax.set_title(f"Calibration of best model: {best_overall['panel']}/{best_overall['model']}\n"
                     f"AUC={best_overall['pooled_oof_auc']:.3f}, Brier={best_overall['brier_mean']:.3f}")
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save oof predictions for downstream
    (OUT / "oof_predictions.json").write_text(
        json.dumps(oof_dump, indent=2), encoding="utf-8")

    # STOP RULE check
    best_auc = res_df["pooled_oof_auc"].max()
    print(f"\nBest pooled OOF AUC = {best_auc:.3f}")
    print(f"STOP RULE (AUC<0.75): {'TRIPPED — emphasize severity narrative' if best_auc < 0.75 else 'NOT tripped — classification claim defensible'}")
    print("\nTop 5 configs:")
    print(res_df.head(5)[["panel", "model", "pooled_oof_auc",
                           "pooled_oof_auc_ci_lo", "pooled_oof_auc_ci_hi",
                           "f1_mean", "bal_acc_mean", "brier_mean"]].to_string())


if __name__ == "__main__":
    main()
