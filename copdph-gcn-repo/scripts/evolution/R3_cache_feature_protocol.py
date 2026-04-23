"""Round 3 — protocol decodability on features closer to what the GCN sees.

The Round 2 reviewer insisted that protocol AUC must be measured on features
that the GCN actually consumes, not on raw scalar lung HU (which we already
showed hits protocol AUC 1.000). Without direct access to the remote graph
cache, we approximate "what the GCN sees" with the per-structure volumes and
diameter/tortuosity proxies that kimimaro TEASAR produces.

Feature sets tested (5-fold stratified CV on 282 cases):

  A. per_structure_volumes : artery/vein/airway vol_mL + vessel_airway_over_lung
  B. per_structure_plus_ratios : A + A/V ratio, A/(A+V), artery/airway, vein/airway
  C. spatial_only : apical/middle/basal LAA + gradient (no HU)
  D. paren_only_no_HU_abs : paren LAA fractions only (no absolute HU)
  E. v2_combined_no_HU : C + D (exclude all absolute HU)

Interpretation:
  - If protocol AUC drops below 0.7 with a disease-preserving feature set,
    we can claim a protocol-robust representation for the paper.
  - If even D/E hit protocol AUC > 0.9, the cache structure itself carries
    protocol and we need domain-adversarial mitigation before any claim.
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

ROOT = Path(__file__).parent.parent.parent
V2 = ROOT / "outputs" / "lung_features_v2.csv"
PROTO = ROOT / "data" / "case_protocol.csv"
OUT_MD = ROOT / "outputs" / "_r3_cache_feature_protocol.md"
OUT_JSON = ROOT / "outputs" / "_r3_cache_feature_protocol.json"


def cv_auc(X: np.ndarray, y: np.ndarray, model_name: str, seed: int = 20260423):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        if model_name == "lr":
            clf = LogisticRegression(max_iter=5000, class_weight="balanced")
        else:
            clf = GradientBoostingClassifier(random_state=seed)
        s = StandardScaler().fit(X[tr])
        clf.fit(s.transform(X[tr]), y[tr])
        p = clf.predict_proba(s.transform(X[te]))[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)), float(np.std(aucs))


def main() -> None:
    v2 = pd.read_csv(V2)
    proto = pd.read_csv(PROTO)
    df = proto.merge(v2, on="case_id", how="inner")
    df["is_contrast"] = (df["protocol"] == "contrast").astype(int)

    # Build derived ratios
    df["artery_over_vein"] = df["artery_vol_mL"] / df["vein_vol_mL"].replace(0, np.nan)
    df["artery_over_total_vessel"] = df["artery_vol_mL"] / (
        df["artery_vol_mL"] + df["vein_vol_mL"]
    ).replace(0, np.nan)
    df["artery_over_airway"] = df["artery_vol_mL"] / df["airway_vol_mL"].replace(0, np.nan)
    df["vein_over_airway"] = df["vein_vol_mL"] / df["airway_vol_mL"].replace(0, np.nan)

    sets = {
        "A_per_structure_volumes": [
            "artery_vol_mL", "vein_vol_mL", "airway_vol_mL",
            "vessel_airway_over_lung",
        ],
        "B_volumes_plus_ratios": [
            "artery_vol_mL", "vein_vol_mL", "airway_vol_mL",
            "artery_over_vein", "artery_over_total_vessel",
            "artery_over_airway", "vein_over_airway",
        ],
        "C_spatial_only": [
            "apical_LAA_950_frac", "middle_LAA_950_frac", "basal_LAA_950_frac",
            "apical_basal_LAA950_gradient",
        ],
        "D_paren_LAA_only": [
            "paren_LAA_950_frac", "paren_LAA_910_frac", "paren_LAA_856_frac",
        ],
        "E_v2_ratio_combined_no_HU": [
            "paren_LAA_950_frac", "paren_LAA_910_frac", "paren_LAA_856_frac",
            "apical_LAA_950_frac", "middle_LAA_950_frac", "basal_LAA_950_frac",
            "apical_basal_LAA950_gradient",
            "artery_over_vein", "artery_over_total_vessel",
            "artery_over_airway", "vein_over_airway",
        ],
    }

    out = {"sets": {}, "interpretation": ""}
    lines = [
        "# Round 3 — protocol decodability on cache-adjacent features",
        "",
        "The Round 2 reviewer required protocol AUC to be measured on features",
        "closer to the GCN's actual inputs, not scalar lung HU. Without remote",
        "graph cache access, we approximate via per-structure volumes, vessel-",
        "volume ratios, and parenchyma LAA (all protocol-agnostic in principle",
        "because ratios cancel absolute HU offsets).",
        "",
        "5-fold stratified CV AUCs:",
        "",
        "| Set | n_feats | n_cases | Protocol AUC (LR / GB) | Disease full (LR / GB) | Disease contrast (LR / GB) |",
        "|---|---|---|---|---|---|",
    ]
    for name, feats in sets.items():
        sub = df.dropna(subset=feats)
        X = sub[feats].to_numpy()
        y_p = sub["is_contrast"].to_numpy()
        y_d = sub["label"].to_numpy()
        sub_c = sub[sub["is_contrast"] == 1]
        Xc = sub_c[feats].to_numpy()
        yc = sub_c["label"].to_numpy()
        p_lr = cv_auc(X, y_p, "lr")
        p_gb = cv_auc(X, y_p, "gb")
        d_lr = cv_auc(X, y_d, "lr")
        d_gb = cv_auc(X, y_d, "gb")
        dc_lr = cv_auc(Xc, yc, "lr")
        dc_gb = cv_auc(Xc, yc, "gb")
        out["sets"][name] = {
            "n_feats": len(feats),
            "n_cases": int(len(sub)),
            "n_contrast_only": int(len(sub_c)),
            "protocol_lr": p_lr, "protocol_gb": p_gb,
            "disease_full_lr": d_lr, "disease_full_gb": d_gb,
            "disease_contrast_lr": dc_lr, "disease_contrast_gb": dc_gb,
        }
        lines.append(
            f"| `{name}` | {len(feats)} | {len(sub)} | "
            f"{p_lr[0]:.3f} / {p_gb[0]:.3f} | "
            f"{d_lr[0]:.3f} / {d_gb[0]:.3f} | "
            f"{dc_lr[0]:.3f} / {dc_gb[0]:.3f} |"
        )

    # Heuristic interpretation
    best_protocol_robust = min(
        out["sets"].items(),
        key=lambda kv: min(kv[1]["protocol_lr"][0], kv[1]["protocol_gb"][0]),
    )
    lines += [
        "",
        "## Reading",
        "",
        f"- Most protocol-robust set: `{best_protocol_robust[0]}` "
        f"(protocol AUC {min(best_protocol_robust[1]['protocol_lr'][0], best_protocol_robust[1]['protocol_gb'][0]):.3f}).",
        "- Disease AUC on the **contrast-only subset** is the reviewer-facing endpoint; full-cohort numbers above it remain inflated by any residual protocol leakage.",
        "- Ratios (A/V, A/total_vessel, artery/airway) cancel absolute HU offsets but still",
        "  carry protocol signal through segmentation-quality differences; exact quantification",
        "  above shows how much.",
        "",
        "**Caveat**: these are abundance-derived features, not the 12-D TEASAR node features",
        "the GCN actually consumes. A true measurement requires loading `cache_v2_tri_flat/*.pkl`",
        "on the remote server and computing graph-level statistics (planned as E1 for Round 4).",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(OUT_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
