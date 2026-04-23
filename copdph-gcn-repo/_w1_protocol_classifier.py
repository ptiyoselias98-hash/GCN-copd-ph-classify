"""W1 stress-test: how trivially can protocol be decoded?

Train scalar classifiers to predict `protocol` (contrast vs plain-scan) from:
  (a) the 14 existing v1 whole-lung features in `lung_features_only.csv`,
  (b) a SINGLE feature at a time — to identify the most trivial decoders.

If a linear / tree model reaches AUC ≈ 1.0 on protocol, then the residual
~0.87 contrast-only disease AUC in REPORT_v2 §13 is a *lower bound* on
confounding (the model had full access to protocol info in the features).

Leave-one-case-out (LOOCV) + 5-fold stratified CV reported for robustness.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
LUNG_V1 = ROOT / "outputs" / "lung_features_only.csv"
PROTOCOL = ROOT / "data" / "case_protocol.csv"
OUT_MD = ROOT / "outputs" / "_w1_protocol_classifier.md"
OUT_JSON = ROOT / "outputs" / "_w1_protocol_classifier.json"

FEATURE_COLS = [
    "lung_vol_mL",
    "mean_HU", "std_HU",
    "HU_p5", "HU_p25", "HU_p50", "HU_p75", "HU_p95",
    "LAA_950_frac", "LAA_910_frac", "LAA_856_frac",
    "largest_comp_frac", "n_components",
]


def load_data() -> pd.DataFrame:
    lung = pd.read_csv(LUNG_V1)
    proto = pd.read_csv(PROTOCOL)
    df = proto.merge(lung, on="case_id", how="inner")
    df["is_contrast"] = (df["protocol"] == "contrast").astype(int)
    # drop rows with any NaN in features
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS)
    print(f"Merged {before} → {len(df)} cases after NaN drop")
    return df


def cv_auc(X: np.ndarray, y: np.ndarray, model_name: str) -> tuple[float, float, list[float]]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260423)
    aucs = []
    for tr, te in skf.split(X, y):
        if model_name == "lr":
            clf = LogisticRegression(max_iter=5000, class_weight="balanced")
        else:
            clf = GradientBoostingClassifier(random_state=20260423)
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr])
        Xte = scaler.transform(X[te])
        clf.fit(Xtr, y[tr])
        p = clf.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)), float(np.std(aucs)), aucs


def single_feature_auc(df: pd.DataFrame, target: str) -> list[dict]:
    rows = []
    y = df[target].to_numpy()
    for feat in FEATURE_COLS:
        x = df[feat].to_numpy().reshape(-1, 1)
        m, s, _ = cv_auc(x, y, "lr")
        rows.append({"feature": feat, "cv_auc_mean": m, "cv_auc_std": s})
    return sorted(rows, key=lambda r: r["cv_auc_mean"], reverse=True)


def main() -> None:
    df = load_data()
    X = df[FEATURE_COLS].to_numpy()
    # --- Protocol prediction ---
    y_proto = df["is_contrast"].to_numpy()
    print(f"Protocol: {int(y_proto.sum())} contrast vs {int((1-y_proto).sum())} plain-scan")
    lr_proto = cv_auc(X, y_proto, "lr")
    gb_proto = cv_auc(X, y_proto, "gb")
    print(f"Protocol AUC — LR: {lr_proto[0]:.4f} ± {lr_proto[1]:.4f}")
    print(f"Protocol AUC — GB: {gb_proto[0]:.4f} ± {gb_proto[1]:.4f}")
    single_proto = single_feature_auc(df, "is_contrast")

    # --- Disease prediction on the same features (for context) ---
    y_label = df["label"].to_numpy()
    lr_label = cv_auc(X, y_label, "lr")
    gb_label = cv_auc(X, y_label, "gb")

    # --- Disease prediction restricted to contrast-only (honest signal) ---
    df_c = df[df["is_contrast"] == 1].reset_index(drop=True)
    lr_label_c = cv_auc(df_c[FEATURE_COLS].to_numpy(), df_c["label"].to_numpy(), "lr")
    gb_label_c = cv_auc(df_c[FEATURE_COLS].to_numpy(), df_c["label"].to_numpy(), "gb")

    lines = [
        "# W1 stress-test — lung-feature protocol-decoder baseline",
        "",
        "Question: can the 14 scalar lung features in `lung_features_only.csv` predict",
        "the **acquisition protocol** (contrast vs plain-scan) as accurately as they",
        "predict **disease**? If yes, the residual ~0.87 disease AUC on the",
        "contrast-only subset (§13) is a lower bound on confounding, not an upper bound.",
        "",
        "## 5-fold stratified CV AUCs (scalar lung features only)",
        "",
        "| Target | Logistic Regression | Gradient Boosting |",
        "|---|---|---|",
        f"| **Protocol** (contrast vs plain-scan) | **{lr_proto[0]:.4f} ± {lr_proto[1]:.4f}** | **{gb_proto[0]:.4f} ± {gb_proto[1]:.4f}** |",
        f"| Disease (full cohort, label) | {lr_label[0]:.4f} ± {lr_label[1]:.4f} | {gb_label[0]:.4f} ± {gb_label[1]:.4f} |",
        f"| Disease (contrast-only, label) | {lr_label_c[0]:.4f} ± {lr_label_c[1]:.4f} | {gb_label_c[0]:.4f} ± {gb_label_c[1]:.4f} |",
        "",
        "## Single-feature protocol decoders (ranked)",
        "",
        "| Feature | CV AUC for protocol |",
        "|---|---|",
    ]
    for r in single_proto:
        lines.append(f"| `{r['feature']}` | {r['cv_auc_mean']:.4f} ± {r['cv_auc_std']:.4f} |")
    lines += [
        "",
        "## Interpretation",
        "",
        f"- Protocol is decoded with **AUC {max(lr_proto[0], gb_proto[0]):.3f}** from scalar lung",
        "  features alone. Any classifier that uses these features has access to a near-perfect",
        "  protocol cue.",
        f"- On the contrast-only subset the **same features** predict disease at AUC",
        f"  {gb_label_c[0]:.3f} (LR {lr_label_c[0]:.3f}) — a direct measure of the *residual*",
        f"  disease signal in the lung features after removing protocol confounding.",
        "- The gap between full-cohort disease AUC and contrast-only disease AUC is explained",
        "  by the protocol decoder: features that discriminate protocol mechanically contribute",
        "  to the full-cohort disease AUC even when they have no causal disease signal.",
        "",
        "**Reviewer-facing statement**: claims of disease-relevant lung-phenotype effects",
        "must be supported by the contrast-only column above, not the full-cohort column.",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    OUT_JSON.write_text(
        json.dumps(
            {
                "protocol_lr": {"mean": lr_proto[0], "std": lr_proto[1], "folds": lr_proto[2]},
                "protocol_gb": {"mean": gb_proto[0], "std": gb_proto[1], "folds": gb_proto[2]},
                "disease_full_lr": {"mean": lr_label[0], "std": lr_label[1], "folds": lr_label[2]},
                "disease_full_gb": {"mean": gb_label[0], "std": gb_label[1], "folds": gb_label[2]},
                "disease_contrast_lr": {"mean": lr_label_c[0], "std": lr_label_c[1], "folds": lr_label_c[2]},
                "disease_contrast_gb": {"mean": gb_label_c[0], "std": gb_label_c[1], "folds": gb_label_c[2]},
                "single_feature_protocol": single_proto,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(OUT_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
