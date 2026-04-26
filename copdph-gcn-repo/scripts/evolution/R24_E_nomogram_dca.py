"""R24.E — Risk nomogram + calibration + DCA on within-contrast n=190.

10× repeated nested 5-fold CV with Lasso; calibration slope/intercept + Brier;
DCA at prevalence anchors {10, 25, 50, 86%}; explicit EXPLORATORY stamp.

Output:
  outputs/r24/r24e_nomogram_dca.csv
  outputs/r24/r24e_validation.json
  outputs/figures/fig_r24e_nomogram_dca.png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent
COHORT = ROOT / "outputs" / "r24" / "cohort_locked_table.csv"
MORPH = ROOT / "outputs" / "r20" / "morph_unified301.csv"
OUT = ROOT / "outputs" / "r24"
FIG = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

ARTIFACTS = {"airway_n_terminals", "airway_term_per_node",
             "artery_lap_eig0", "artery_n_terminals", "artery_term_per_node",
             "vein_lap_eig0", "vein_n_terminals", "vein_term_per_node"}


def calibration_slope_intercept(y, p):
    """Fit logistic regression: logit(y) = a + b * logit(p). Slope=b."""
    p = np.clip(p, 1e-3, 1 - 1e-3)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, fit_intercept=True).fit(z, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def dca_at_threshold(y, p, prevalence):
    """Net Benefit at decision threshold matching given prevalence as 'high-risk'."""
    threshold = np.quantile(p, 1 - prevalence)
    pred = p >= threshold
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    n = len(y)
    if n == 0: return 0.0, 0.0
    pt = threshold
    nb = (tp / n) - (fp / n) * (pt / (1 - pt + 1e-9))
    nb_treat_all = (y.sum() / n) - ((n - y.sum()) / n) * (pt / (1 - pt + 1e-9))
    return float(nb), float(nb_treat_all)


def main():
    cohort = pd.read_csv(COHORT)
    morph = pd.read_csv(MORPH).drop(columns=["label"], errors="ignore")
    df = cohort.merge(morph, on="case_id", how="inner")
    df = df[df["is_contrast_only_subset"]].reset_index(drop=True)
    print(f"R24.E within-contrast n={len(df)}")

    feat_cols = [c for c in morph.columns
                 if c not in ("case_id", "source_cache")
                 and c not in ARTIFACTS]
    X_all = df[feat_cols].fillna(0).values.astype(float)
    y_all = df["label"].astype(int).values

    # 10× repeated 5-fold CV (50 fold-models)
    seeds = list(range(10))
    fold_aucs = []
    oof_p = np.full(len(y_all), np.nan)
    oof_count = np.zeros(len(y_all), int)
    for seed in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, va in skf.split(X_all, y_all):
            sc = RobustScaler().fit(X_all[tr])
            Xtr = sc.transform(X_all[tr]); Xva = sc.transform(X_all[va])
            try:
                clf = LogisticRegressionCV(Cs=10, cv=3, penalty="l1",
                                            solver="liblinear", scoring="roc_auc",
                                            max_iter=2000, random_state=seed).fit(Xtr, y_all[tr])
                p = clf.predict_proba(Xva)[:, 1]
                fold_aucs.append(roc_auc_score(y_all[va], p))
                # Average predictions across seeds
                cur = oof_p[va]; oof_p[va] = np.where(np.isnan(cur), p, cur + p)
                oof_count[va] += 1
            except Exception:
                continue
    oof_p = oof_p / np.where(oof_count > 0, oof_count, 1)

    # Calibration + Brier on averaged OOF
    cal_slope, cal_intercept = calibration_slope_intercept(y_all, oof_p)
    brier = float(brier_score_loss(y_all, oof_p))
    auc_full = float(roc_auc_score(y_all, oof_p))

    # DCA at multiple prevalence anchors
    dca_rows = []
    for prev in [0.10, 0.25, 0.50, 0.86]:
        nb, nb_treat_all = dca_at_threshold(y_all, oof_p, prev)
        dca_rows.append({"prevalence": prev, "net_benefit_model": nb,
                         "net_benefit_treat_all": nb_treat_all,
                         "delta_nb": nb - nb_treat_all})
    dca_df = pd.DataFrame(dca_rows)
    dca_df.to_csv(OUT / "r24e_nomogram_dca.csv", index=False)

    out = {
        "n": int(len(df)),
        "n_PH": int(y_all.sum()),
        "n_nonPH": int((y_all == 0).sum()),
        "fold_count": len(fold_aucs),
        "fold_auc_mean": float(np.mean(fold_aucs)),
        "fold_auc_std": float(np.std(fold_aucs)),
        "fold_auc_min": float(np.min(fold_aucs)),
        "fold_auc_max": float(np.max(fold_aucs)),
        "oof_auc_pooled": auc_full,
        "calibration_slope": cal_slope,
        "calibration_intercept": cal_intercept,
        "brier_score": brier,
        "dca_anchors": dca_rows,
        "exploratory_stamp": "EXPLORATORY — external validation required",
    }
    (OUT / "r24e_validation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Figure: 2-panel calibration + DCA
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    # Calibration
    ax = axes[0]
    bins = np.linspace(0, 1, 9)
    bin_idx = np.digitize(oof_p, bins) - 1
    bin_pred = []; bin_obs = []; bin_n = []
    for b in range(len(bins) - 1):
        m = bin_idx == b
        if m.sum() >= 3:
            bin_pred.append(oof_p[m].mean())
            bin_obs.append(y_all[m].mean())
            bin_n.append(m.sum())
    ax.scatter(bin_pred, bin_obs, s=[n * 5 for n in bin_n], c="#3b82f6", alpha=0.7,
               edgecolors="black")
    ax.plot([0, 1], [0, 1], "--", c="grey", label="ideal")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title(f"Calibration (slope={cal_slope:+.2f}, int={cal_intercept:+.2f}, "
                 f"Brier={brier:.3f})\nfold-AUC={out['fold_auc_mean']:.3f} ± {out['fold_auc_std']:.3f}")
    ax.legend(); ax.grid(alpha=0.3)
    # DCA at 4 anchors
    ax = axes[1]
    x = [r["prevalence"] for r in dca_rows]
    nb = [r["net_benefit_model"] for r in dca_rows]
    nb_all = [r["net_benefit_treat_all"] for r in dca_rows]
    width = 0.025
    ax.bar([xi - width/2 for xi in x], nb, width=width, label="model", color="#ef4444", alpha=0.8)
    ax.bar([xi + width/2 for xi in x], nb_all, width=width, label="treat-all", color="#94a3b8", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"{p*100:.0f}%" for p in x])
    ax.set_xlabel("Assumed prevalence anchor")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision Curve Analysis at multiple prevalence anchors")
    ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle(f"R24.E — Exploratory risk nomogram (within-contrast n={out['n']})\n"
                 f"⚠ EXPLORATORY — external validation required (n_nonPH={out['n_nonPH']} small)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out_fig = FIG / "fig_r24e_nomogram_dca.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close()

    print(json.dumps(out, indent=2))
    print(f"saved {out_fig}")


if __name__ == "__main__":
    main()
