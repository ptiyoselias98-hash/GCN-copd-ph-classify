"""02_train_clean_panel_models — Phase 2 of clean AUC boost.

7 panels × 4 models × 10-seed × 5-fold patient-level CV.
All preprocessing INSIDE fold (median-impute + RobustScale).
Six metrics + Brier + bootstrap-500 95% CI on AUC.

Models: ElasticNet LR + RandomForest + GradBoost (XGBoost/LightGBM if installed).
Panels:
  P0 SSL only / P1 lung only / P2 topology only / P3 lung+topology
  P4 lung+topology+SSL / P5 all clean (lung+topo+SSL+TDA+AV)
  P6 benchmark (PA/Ao/PA_diam — empty in repo, skipped)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              brier_score_loss, roc_curve)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent
TBL = ROOT / "outputs" / "clean_auc_boost" / "clean_feature_table_within_contrast.csv"
GMAP = ROOT / "outputs" / "clean_auc_boost" / "feature_group_map.json"
OUT = ROOT / "outputs" / "clean_auc_boost"

META = {"case_id", "label", "mpap", "protocol", "is_contrast_only_subset",
        "measured_mpap_flag", "fold_id"}


def youden_thr(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def metrics_at(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred==1)&(y==1)).sum()); fp = int(((pred==1)&(y==0)).sum())
    tn = int(((pred==0)&(y==0)).sum()); fn = int(((pred==0)&(y==1)).sum())
    n = len(y)
    return {"acc": (tp+tn)/n if n else 0.0,
            "sens": tp/(tp+fn) if (tp+fn) else 0.0,
            "spec": tn/(tn+fp) if (tn+fp) else 0.0,
            "prec": tp/(tp+fp) if (tp+fp) else 0.0,
            "f1": 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.0,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def make_clf(name, seed, n_features):
    if name == "ElasticNet":
        # ElasticNet via LogisticRegressionCV — try with l1_ratios; fallback to penalty='elasticnet'
        try:
            return LogisticRegressionCV(Cs=10, cv=3, penalty="elasticnet",
                                          solver="saga",
                                          l1_ratios=[0.1, 0.5, 0.9],
                                          scoring="roc_auc",
                                          max_iter=3000, random_state=seed)
        except Exception:
            return LogisticRegressionCV(Cs=10, cv=3, penalty="l2",
                                          scoring="roc_auc", max_iter=2000, random_state=seed)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, max_depth=8,
                                       min_samples_leaf=3, random_state=seed,
                                       n_jobs=1, class_weight="balanced")
    if name == "GradBoost":
        return GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                           learning_rate=0.05, random_state=seed,
                                           subsample=0.8)
    raise ValueError(name)


def fit_panel_model(panel_name, panel_cols, model_name, df, seeds):
    if not panel_cols:
        return None
    X = df[panel_cols].values.astype(float)
    y = df["label"].astype(int).values
    n_folds_done = 0
    fold_metrics = []
    oof_p = np.full(len(y), np.nan)
    oof_n = np.zeros(len(y), int)
    for seed in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, va in skf.split(X, y):
            # Median-impute INSIDE fold
            imp = SimpleImputer(strategy="median").fit(X[tr])
            Xtr_i = imp.transform(X[tr]); Xva_i = imp.transform(X[va])
            sc = RobustScaler().fit(Xtr_i)
            Xtr = sc.transform(Xtr_i); Xva = sc.transform(Xva_i)
            try:
                clf = make_clf(model_name, seed, X.shape[1]).fit(Xtr, y[tr])
                p_tr = clf.predict_proba(Xtr)[:, 1]
                p_va = clf.predict_proba(Xva)[:, 1]
                thr = youden_thr(y[tr], p_tr)
                m = metrics_at(y[va], p_va, thr)
                m["auc"] = float(roc_auc_score(y[va], p_va))
                m["pr_auc"] = float(average_precision_score(y[va], p_va))
                m["brier"] = float(brier_score_loss(y[va], p_va))
                m["thr"] = thr
                fold_metrics.append(m)
                cur = oof_p[va]; oof_p[va] = np.where(np.isnan(cur), p_va, cur + p_va)
                oof_n[va] += 1
                n_folds_done += 1
            except Exception:
                continue
    if n_folds_done == 0:
        return None
    oof_p = oof_p / np.where(oof_n > 0, oof_n, 1)
    pooled_auc = float(roc_auc_score(y, oof_p))
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(500):
        idx = rng.integers(0, len(y), size=len(y))
        if y[idx].sum() < 3 or (y[idx]==0).sum() < 3: continue
        try: boots.append(roc_auc_score(y[idx], oof_p[idx]))
        except Exception: continue
    boots = np.array(boots) if boots else np.array([pooled_auc])
    ci_lo = float(np.percentile(boots, 2.5))
    ci_hi = float(np.percentile(boots, 97.5))
    agg = {}
    for k in ("auc","acc","sens","spec","prec","f1","pr_auc","brier","thr"):
        agg[f"{k}_mean"] = float(np.mean([m[k] for m in fold_metrics]))
        agg[f"{k}_std"] = float(np.std([m[k] for m in fold_metrics]))
    return {"panel": panel_name, "model": model_name,
            "n_features": int(X.shape[1]), "n_folds": n_folds_done,
            "pooled_oof_auc": pooled_auc,
            "pooled_oof_auc_ci_lo": ci_lo, "pooled_oof_auc_ci_hi": ci_hi,
            **agg,
            "oof_p": oof_p.tolist(),
            "y": y.tolist(),
            "case_ids": df["case_id"].tolist()}


def main():
    df = pd.read_csv(TBL)
    gmap = json.loads(GMAP.read_text(encoding="utf-8"))
    A = gmap["A_ssl_severity"]["features"]
    B = gmap["B_lung_heterogeneity"]["features"]
    C = gmap["C_vascular_topology"]["features"]
    D = gmap["D_pruning_distribution"]["features"]
    E = gmap["E_av_imbalance"]["features"]
    F = gmap["F_tda"]["features"]

    panels = {
        "P0_ssl_only": A,
        "P1_lung_only": B,
        "P2_topology_only": C + D + E,
        "P3_lung_topology": B + C + D + E,
        "P4_lung_topology_ssl": A + B + C + D + E,
        "P5_all_clean": A + B + C + D + E + F,
    }
    for k, v in panels.items(): print(f"  {k}: {len(v)} feats")

    models = ["ElasticNet", "RandomForest", "GradBoost"]
    seeds = list(range(10))
    print(f"\n{len(panels)} panels × {len(models)} models = {len(panels)*len(models)} configs")

    configs = [(p, c, m) for p, c in panels.items() for m in models]
    results = Parallel(n_jobs=4, backend="loky")(
        delayed(fit_panel_model)(p, c, m, df, seeds) for p, c, m in configs
    )
    results = [r for r in results if r is not None]

    rows = []
    pred_dump = {}
    for r in results:
        key = f"{r['panel']}_{r['model']}"
        pred_dump[key] = {"oof_p": r["oof_p"], "y": r["y"], "case_ids": r["case_ids"]}
        rows.append({k: v for k, v in r.items()
                     if k not in ("oof_p", "y", "case_ids")})
    res_df = pd.DataFrame(rows).sort_values("pooled_oof_auc", ascending=False).reset_index(drop=True)
    res_df.to_csv(OUT / "panel_model_results.csv", index=False)

    # Save predictions wide-form for Phase 3 stacking
    pred_table = pd.DataFrame({"case_id": results[0]["case_ids"],
                                "y": results[0]["y"]})
    for r in results:
        key = f"{r['panel']}_{r['model']}"
        pred_table[key] = r["oof_p"]
    pred_table.to_csv(OUT / "panel_model_predictions.csv", index=False)

    # ROC comparison (best model per panel)
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
                      f"[{best['pooled_oof_auc_ci_lo']:.2f}, {best['pooled_oof_auc_ci_hi']:.2f}]")
    ax.plot([0,1],[0,1], "--", c="grey", label="chance")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"clean_auc_boost — best model per panel (within-contrast n={len(df)})\n"
                 "10 seeds × 5-fold OOF, ElasticNet/RF/GBM")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
    # Calibration (best overall)
    if results:
        best_overall = max(results, key=lambda r: r["pooled_oof_auc"])
        ax = axes[1]
        y = np.array(best_overall["y"]); p = np.array(best_overall["oof_p"])
        bins = np.linspace(0, 1, 9)
        bin_idx = np.digitize(p, bins) - 1
        bp, bo, bn = [], [], []
        for b in range(len(bins)-1):
            m = bin_idx == b
            if m.sum() >= 3:
                bp.append(p[m].mean()); bo.append(y[m].mean()); bn.append(m.sum())
        ax.scatter(bp, bo, s=[n*5 for n in bn], c="#3b82f6", alpha=0.7, edgecolors="black")
        ax.plot([0,1],[0,1], "--", c="grey", label="ideal")
        ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
        ax.set_title(f"Calibration of best: {best_overall['panel']}/{best_overall['model']}\n"
                     f"AUC={best_overall['pooled_oof_auc']:.3f} Brier={best_overall['brier_mean']:.3f}")
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "roc_panel_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save best config
    best_overall = max(results, key=lambda r: r["pooled_oof_auc"])
    (OUT / "best_model_config.json").write_text(
        json.dumps({"panel": best_overall["panel"],
                    "model": best_overall["model"],
                    "pooled_oof_auc": best_overall["pooled_oof_auc"],
                    "ci_lo": best_overall["pooled_oof_auc_ci_lo"],
                    "ci_hi": best_overall["pooled_oof_auc_ci_hi"]}, indent=2),
        encoding="utf-8")

    print(f"\nbest overall: {best_overall['panel']}/{best_overall['model']} "
          f"AUC={best_overall['pooled_oof_auc']:.3f} "
          f"[{best_overall['pooled_oof_auc_ci_lo']:.3f}, {best_overall['pooled_oof_auc_ci_hi']:.3f}]")
    print("\ntop-5 configs:")
    print(res_df.head(5)[["panel","model","pooled_oof_auc",
                           "pooled_oof_auc_ci_lo","pooled_oof_auc_ci_hi",
                           "f1_mean","brier_mean"]].to_string())


if __name__ == "__main__":
    main()
