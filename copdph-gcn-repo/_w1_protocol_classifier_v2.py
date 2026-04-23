"""W1 stress-test v2 — compare whole-lung (v1) vs parenchyma-only (v2) features.

Key question: does subtracting vessels + airway from the lung mask before
computing HU / LAA statistics break the perfect protocol decoder?

Three feature sets tested on 5-fold stratified CV (GB + LR):
  1. whole_* : mimic v1 lung_features_only (whole-lung HU/LAA)
  2. paren_* : protocol-robust (parenchyma only)
  3. spatial_*: apical/middle/basal LAA_950 + gradient (parenchyma)

For each set we report protocol-AUC, disease-AUC (full), disease-AUC (contrast-only).
Target: paren_* and spatial_* should have protocol AUC ≪ 1.0 while retaining
or improving contrast-only disease AUC.
"""

from __future__ import annotations

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
LUNG_V2 = ROOT / "outputs" / "lung_features_v2.csv"
PROTOCOL = ROOT / "data" / "case_protocol.csv"
OUT_MD = ROOT / "outputs" / "_w1_protocol_classifier_v2.md"
OUT_JSON = ROOT / "outputs" / "_w1_protocol_classifier_v2.json"

FEATURE_SETS = {
    "whole_lung": [
        "lung_vol_mL",
        "whole_mean_HU", "whole_std_HU",
        "whole_HU_p5", "whole_HU_p25", "whole_HU_p50", "whole_HU_p75", "whole_HU_p95",
        "whole_LAA_950_frac", "whole_LAA_910_frac", "whole_LAA_856_frac",
    ],
    "parenchyma_only": [
        "paren_mean_HU", "paren_std_HU",
        "paren_HU_p5", "paren_HU_p25", "paren_HU_p50", "paren_HU_p75", "paren_HU_p95",
        "paren_LAA_950_frac", "paren_LAA_910_frac", "paren_LAA_856_frac",
    ],
    "spatial_paren": [
        "apical_LAA_950_frac", "apical_LAA_910_frac", "apical_mean_HU",
        "middle_LAA_950_frac", "middle_LAA_910_frac", "middle_mean_HU",
        "basal_LAA_950_frac", "basal_LAA_910_frac", "basal_mean_HU",
        "apical_basal_LAA950_gradient",
    ],
    "vessel_lung_integration": [
        "artery_vol_mL", "vein_vol_mL", "airway_vol_mL",
        "vessel_airway_over_lung",
        "artery_mean_HU", "vein_mean_HU", "airway_mean_HU",
    ],
    "paren_plus_spatial": [  # combined: the candidate v2 disease feature set
        "paren_mean_HU", "paren_std_HU",
        "paren_HU_p5", "paren_HU_p25", "paren_HU_p50", "paren_HU_p75", "paren_HU_p95",
        "paren_LAA_950_frac", "paren_LAA_910_frac", "paren_LAA_856_frac",
        "apical_LAA_950_frac", "middle_LAA_950_frac", "basal_LAA_950_frac",
        "apical_basal_LAA950_gradient",
    ],
}


def cv_auc(X: np.ndarray, y: np.ndarray, model_name: str, seed: int = 20260423):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        if model_name == "lr":
            clf = LogisticRegression(max_iter=5000, class_weight="balanced")
        else:
            clf = GradientBoostingClassifier(random_state=seed)
        scaler = StandardScaler().fit(X[tr])
        clf.fit(scaler.transform(X[tr]), y[tr])
        p = clf.predict_proba(scaler.transform(X[te]))[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)), float(np.std(aucs)), aucs


def eval_feature_set(df: pd.DataFrame, feats: list[str]) -> dict:
    sub = df.dropna(subset=feats)
    X = sub[feats].to_numpy()
    y_p = sub["is_contrast"].to_numpy()
    y_d = sub["label"].to_numpy()
    sub_c = sub[sub["is_contrast"] == 1]
    Xc = sub_c[feats].to_numpy()
    yc = sub_c["label"].to_numpy()
    return {
        "n_cases": len(sub),
        "n_contrast_only": len(sub_c),
        "protocol_lr": cv_auc(X, y_p, "lr"),
        "protocol_gb": cv_auc(X, y_p, "gb"),
        "disease_full_lr": cv_auc(X, y_d, "lr"),
        "disease_full_gb": cv_auc(X, y_d, "gb"),
        "disease_contrast_lr": cv_auc(Xc, yc, "lr"),
        "disease_contrast_gb": cv_auc(Xc, yc, "gb"),
    }


def main() -> None:
    lung = pd.read_csv(LUNG_V2)
    proto = pd.read_csv(PROTOCOL)
    df = proto.merge(lung, on="case_id", how="inner")
    df["is_contrast"] = (df["protocol"] == "contrast").astype(int)
    print(f"Merged {len(df)} cases")

    results: dict[str, dict] = {}
    for name, feats in FEATURE_SETS.items():
        print(f"\n=== {name} (n_feats={len(feats)}) ===")
        r = eval_feature_set(df, feats)
        results[name] = r
        print(f"  n_cases={r['n_cases']} contrast-only={r['n_contrast_only']}")
        print(f"  Protocol  LR {r['protocol_lr'][0]:.3f}±{r['protocol_lr'][1]:.3f}  "
              f"GB {r['protocol_gb'][0]:.3f}±{r['protocol_gb'][1]:.3f}")
        print(f"  Dis full  LR {r['disease_full_lr'][0]:.3f}  "
              f"GB {r['disease_full_gb'][0]:.3f}")
        print(f"  Dis contr LR {r['disease_contrast_lr'][0]:.3f}  "
              f"GB {r['disease_contrast_gb'][0]:.3f}")

    # Markdown summary
    lines = [
        "# W1 stress-test v2 — whole-lung vs parenchyma-only vs spatial",
        "",
        "Feature-set comparison on 5-fold stratified CV (sklearn LR + GB).",
        "Goal: find a lung-feature representation whose **protocol AUC is near 0.5**",
        "(not trivially decodable) while **contrast-only disease AUC stays high**.",
        "",
        "| Feature set | n_feats | Protocol AUC (LR / GB) | Disease AUC full (LR / GB) | Disease AUC contrast-only (LR / GB) |",
        "|---|---|---|---|---|",
    ]
    for name, feats in FEATURE_SETS.items():
        r = results[name]
        lines.append(
            f"| `{name}` | {len(feats)} | "
            f"{r['protocol_lr'][0]:.3f} / {r['protocol_gb'][0]:.3f} | "
            f"{r['disease_full_lr'][0]:.3f} / {r['disease_full_gb'][0]:.3f} | "
            f"{r['disease_contrast_lr'][0]:.3f} / {r['disease_contrast_gb'][0]:.3f} |"
        )
    lines += [
        "",
        "## Reading the matrix",
        "",
        "- **whole_lung** (legacy v1 features) → protocol AUC 1.0, disease-contrast 0.68.",
        "  Baseline confirming §14.3: the v1 lung-feature gain on the full cohort is almost",
        "  entirely protocol leakage.",
        "- **parenchyma_only** (lung − vessels − airway) → target is protocol AUC close to",
        "  random (~0.5) with disease-contrast matching or exceeding whole_lung's 0.68.",
        "  If protocol AUC is still high this indicates residual leakage from parenchyma",
        "  density itself (possible: contrast in capillaries slightly changes HU).",
        "- **spatial_paren** (apical/middle/basal LAA) → classic radiologic signature for",
        "  upper-zone emphysema. Expected to contribute disease signal orthogonal to",
        "  overall LAA fraction.",
        "- **vessel_lung_integration** (vessel volumes + mean HU) → expected to have very",
        "  high protocol AUC because artery/vein HU differ massively between contrast",
        "  and plain-scan. Serves as a reference 'protocol-decoder' set.",
        "- **paren_plus_spatial** → the candidate combined v2 feature set for disease",
        "  classification that should be substantially protocol-robust.",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    OUT_JSON.write_text(
        json.dumps(
            {k: {k2: (v2 if isinstance(v2, (int, float)) else list(v2)) for k2, v2 in r.items()}
             for k, r in results.items()},
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n" + OUT_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
