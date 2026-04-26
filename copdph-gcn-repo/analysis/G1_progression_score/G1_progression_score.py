"""G1_progression_score — Phase G1: PH-like structural severity score + thresholds.

Per spec (cross-sectional language, NOT longitudinal):
- "cross-sectional severity ordering"
- "PH-like structural progression score"
- "early high-risk phenotype"

Tasks:
T1 Build score: ElasticNet on within-contrast features (D1 P4 best panel) → continuous score
T2 Project: C5_nonPH_proxy → score; identify early high-risk phenotype within nonPH
T3 Thresholds: 90% sensitivity / 90% specificity / Youden thresholds + NPV / PPV / LR+ / LR-
T4 Borderline analysis on C3 mPAP 18-22

Per codex pass-1: calibrate score using C2 contrast-only; project C5 contrast vs plain separately.

Output:
  outputs/supplementary/G1_progression_score/severity_score_patient_level.csv
  outputs/supplementary/G1_progression_score/early_copd_projection.csv
  outputs/supplementary/G1_progression_score/threshold_metrics.csv
  outputs/supplementary/G1_progression_score/severity_axis_plot.png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegressionCV, ElasticNetCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent
SIG = ROOT / "outputs" / "supplementary" / "B1_graph_signature" / "graph_signatures_patient_level.csv"
DICT = ROOT / "outputs" / "supplementary" / "B1_graph_signature" / "graph_signature_dictionary.json"
OUT = ROOT / "outputs" / "supplementary" / "G1_progression_score"
OUT.mkdir(parents=True, exist_ok=True)

META = {"case_id", "label", "protocol", "is_contrast_only_subset",
        "measured_mpap", "measured_mpap_flag", "fold_id",
        "C1_all_available", "C2_within_contrast_only",
        "C3_borderline_mPAP_18_22", "C4_clear_low_high",
        "C5_early_COPD_no_PH_proxy"}


def define_panel(df, fdict):
    """P4 combined-clean panel from D1 (lung + artery + vein + airway-legR17 + TDA)."""
    feat_cols = [c for c in df.columns if c not in META
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
    P_tda = [c for c in feat_cols
              if fdict.get(c, {}).get("category") == "8_TDA_persistence"]
    return sorted(set(P1_lung + P2_vasc + P3_airway + P_tda))


def metrics_at(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred==1)&(y==1)).sum()); fp = int(((pred==1)&(y==0)).sum())
    tn = int(((pred==0)&(y==0)).sum()); fn = int(((pred==0)&(y==1)).sum())
    sens = tp/(tp+fn) if (tp+fn) else 0.0
    spec = tn/(tn+fp) if (tn+fp) else 0.0
    ppv = tp/(tp+fp) if (tp+fp) else 0.0
    npv = tn/(tn+fn) if (tn+fn) else 0.0
    lr_plus = sens/(1-spec) if (1-spec) > 1e-9 else float("inf")
    lr_minus = (1-sens)/spec if spec > 1e-9 else float("inf")
    return {"thr": float(thr), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sens": sens, "spec": spec, "ppv": ppv, "npv": npv,
            "lr_plus": lr_plus, "lr_minus": lr_minus}


def main():
    df = pd.read_csv(SIG)
    fdict = json.loads(DICT.read_text(encoding="utf-8"))["feature_dictionary"]
    panel = define_panel(df, fdict)
    print(f"G1 panel = {len(panel)} features")

    # ===== T1 build severity score on C2 within-contrast =====
    sub = df[df["is_contrast_only_subset"]].reset_index(drop=True)
    assert len(sub) == 190, f"G1 cohort drift: {len(sub)}"
    X = sub[panel].fillna(0).values.astype(float)
    y = sub["label"].astype(int).values

    # 5-seed × 5-fold OOF severity score (LR L2; same as D1 best)
    seeds = [42, 43, 44, 45, 46]
    oof_p = np.full(len(y), np.nan); oof_n = np.zeros(len(y), int)
    feat_imp = np.zeros(len(panel))
    for seed in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, va in skf.split(X, y):
            sc = RobustScaler().fit(X[tr])
            Xtr = sc.transform(X[tr]); Xva = sc.transform(X[va])
            clf = LogisticRegressionCV(Cs=10, cv=3, penalty="l2",
                                        scoring="roc_auc", max_iter=2000,
                                        random_state=seed).fit(Xtr, y[tr])
            p = clf.predict_proba(Xva)[:, 1]
            cur = oof_p[va]; oof_p[va] = np.where(np.isnan(cur), p, cur + p)
            oof_n[va] += 1
            feat_imp += np.abs(clf.coef_[0])
    oof_p = oof_p / np.where(oof_n > 0, oof_n, 1)
    feat_imp = feat_imp / (len(seeds) * 5)
    auc = float(roc_auc_score(y, oof_p))
    print(f"T1 within-contrast OOF severity score AUC={auc:.3f}")

    # Save score per case (within-contrast)
    score_df = sub[["case_id", "label", "measured_mpap",
                     "measured_mpap_flag", "C3_borderline_mPAP_18_22",
                     "C4_clear_low_high", "fold_id"]].copy()
    score_df["severity_score"] = oof_p
    score_df.to_csv(OUT / "severity_score_patient_level.csv", index=False)

    # Spearman with mPAP (n=102 measured)
    valid = sub["measured_mpap_flag"].values
    if valid.sum() >= 5:
        rho, p_rho = spearmanr(oof_p[valid], sub.loc[valid, "measured_mpap"].astype(float))
    else:
        rho, p_rho = float("nan"), float("nan")

    # ===== T2 project C5 (nonPH) cases; train final model on full within-contrast =====
    sc_full = RobustScaler().fit(X)
    X_full = sc_full.transform(X)
    final_clf = LogisticRegressionCV(Cs=10, cv=3, penalty="l2",
                                      scoring="roc_auc", max_iter=2000,
                                      random_state=42).fit(X_full, y)
    # Project ALL cases (include plain-scan + contrast-nonPH C5) with the same scaler
    X_all = df[panel].fillna(0).values.astype(float)
    X_all_sc = sc_full.transform(X_all)
    p_all = final_clf.predict_proba(X_all_sc)[:, 1]
    proj_df = df[["case_id", "label", "protocol", "measured_mpap", "measured_mpap_flag",
                   "C5_early_COPD_no_PH_proxy"]].copy()
    proj_df["progression_score"] = p_all
    # Identify "early high-risk phenotype" within nonPH (label=0)
    nonph = proj_df[proj_df["label"] == 0].copy()
    nonph["high_risk_top_quartile"] = nonph["progression_score"] >= nonph["progression_score"].quantile(0.75)
    proj_df.to_csv(OUT / "early_copd_projection.csv", index=False)

    # Stratify nonPH by protocol per codex pass-1
    print("\nnonPH C5 projection (separate by protocol):")
    for prot in nonph["protocol"].unique():
        sub_p = nonph[nonph["protocol"] == prot]
        print(f"  {prot}: n={len(sub_p)}, mean score={sub_p.progression_score.mean():.3f}, "
              f"high-risk (top 25%) = {int(sub_p.high_risk_top_quartile.sum())}")

    # ===== T3 thresholds: 90% sens / 90% spec / Youden =====
    thr_metrics = []
    fpr, tpr, thr = roc_curve(y, oof_p)
    # Youden
    j = tpr - fpr; idx = int(np.argmax(j))
    youden_thr = float(thr[idx])
    thr_metrics.append({"name": "Youden_J", **metrics_at(y, oof_p, youden_thr)})
    # 90% sensitivity: roc_curve returns thr DESCENDING, tpr ASCENDING.
    # First index with tpr>=0.90 = HIGHEST threshold meeting sens constraint (= max-spec given sens≥0.9).
    high_sens = np.where(tpr >= 0.90)[0]
    if len(high_sens):
        sens_thr = float(thr[high_sens[0]])
        thr_metrics.append({"name": "Sens_90pct", **metrics_at(y, oof_p, sens_thr)})
    # 90% specificity: fpr ASCENDING. Last index with fpr<=0.10 = HIGHEST threshold meeting spec constraint
    # (= max-sens given spec≥0.9).
    high_spec = np.where(fpr <= 0.10)[0]
    if len(high_spec):
        spec_thr = float(thr[high_spec[-1]])
        thr_metrics.append({"name": "Spec_90pct", **metrics_at(y, oof_p, spec_thr)})
    pd.DataFrame(thr_metrics).to_csv(OUT / "threshold_metrics.csv", index=False)
    print("\n=== threshold metrics ===")
    print(pd.DataFrame(thr_metrics).to_string())

    # ===== T4 borderline subgroup analysis =====
    bd = score_df[score_df["C3_borderline_mPAP_18_22"]].copy()
    print(f"\nT4 borderline n={len(bd)} mPAP 18-22 (descriptive only):")
    if len(bd):
        print(bd[["case_id", "label", "measured_mpap", "severity_score"]].to_string(index=False))

    summary = {
        "n_within_contrast": int(len(sub)),
        "panel_size": len(panel),
        "oof_severity_auc": auc,
        "spearman_rho_score_vs_mpap": float(rho),
        "p_rho": float(p_rho),
        "n_mpap_measured": int(valid.sum()),
        "thresholds": thr_metrics,
        "framing": "cross-sectional severity ordering + PH-like structural progression score (NOT longitudinal)",
    }
    (OUT / "g1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ===== Figure: severity score vs mPAP / by group =====
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    # Panel 1: score vs measured mPAP
    ax = axes[0]
    sc = sub[sub["measured_mpap_flag"]].copy()
    sc["score"] = oof_p[sub["measured_mpap_flag"].values]
    ax.scatter(sc["measured_mpap"], sc["score"], c=sc["label"], cmap="RdBu_r",
               s=30, alpha=0.7, edgecolors="black", linewidths=0.4)
    ax.set_xlabel("measured mPAP (mmHg)")
    ax.set_ylabel("OOF severity score")
    ax.set_title(f"Severity score vs mPAP (within-contrast n={int(sub.measured_mpap_flag.sum())})\n"
                 f"Spearman ρ = {rho:+.3f}, p = {p_rho:.2g}")
    ax.grid(alpha=0.3)
    # Panel 2: distribution by label
    ax = axes[1]
    ax.hist([oof_p[y==0], oof_p[y==1]], bins=15, label=["nonPH", "PH"],
            color=["#3b82f6", "#ef4444"], alpha=0.7, edgecolor="black")
    youden = thr_metrics[0]["thr"]
    ax.axvline(youden, color="red", linestyle="--", lw=2, label=f"Youden thr={youden:.2f}")
    ax.set_xlabel("OOF severity score"); ax.set_ylabel("Count")
    ax.set_title("Score distribution by PH label (within-contrast)")
    ax.legend(); ax.grid(alpha=0.3)
    # Panel 3: nonPH C5 projection by protocol (early high-risk)
    ax = axes[2]
    nonph_proj = proj_df[proj_df["label"] == 0]
    for prot, color in [("contrast", "#10b981"), ("plain_scan", "#f59e0b")]:
        sub_p = nonph_proj[nonph_proj["protocol"] == prot]
        if len(sub_p):
            ax.hist(sub_p["progression_score"], bins=12,
                    label=f"{prot} (n={len(sub_p)})",
                    color=color, alpha=0.65, edgecolor="black")
    ax.set_xlabel("PH-like structural progression score")
    ax.set_ylabel("nonPH cases")
    ax.set_title("nonPH C5 projection (cross-sectional)\nseparate by protocol per codex pass-1")
    ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle(f"G1 — PH-like structural progression score (within-contrast n=190)\n"
                 f"AUC={auc:.3f}, ρ vs mPAP={rho:+.3f}, framed as cross-sectional severity NOT longitudinal",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "severity_axis_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved all G1 outputs to {OUT}")


if __name__ == "__main__":
    main()
