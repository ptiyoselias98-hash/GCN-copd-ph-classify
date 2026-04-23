"""R4.1 — Protocol decoder WITHIN nonPH only (critical W1 fix).

The Round 3 reviewer flagged that our protocol AUC measurements across the
full 282-case cohort let the model shortcut via `label → contrast` (since
all 170 PH are contrast). The honest test restricts to label=0 cases:
27 contrast nonPH vs 85 plain-scan nonPH. Protocol AUC here isolates real
protocol signal from label signal.

Feature sets repeated from R3 with the same model pipeline (LR + GB,
5-fold stratified CV). Bootstrap 95% CIs on the mean CV AUC.

Output:
  outputs/_r4_within_nonph_protocol.md
  outputs/_r4_within_nonph_protocol.json
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
OUT_MD = ROOT / "outputs" / "_r4_within_nonph_protocol.md"
OUT_JSON = ROOT / "outputs" / "_r4_within_nonph_protocol.json"

SEED = 20260423
N_BOOT = 2000


def cv_auc(X: np.ndarray, y: np.ndarray, model_name: str, seed: int = SEED):
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
    return float(np.mean(aucs)), float(np.std(aucs)), aucs


def bootstrap_ci(aucs: list[float], rng: np.random.Generator, n_boot: int = N_BOOT):
    a = np.asarray(aucs, dtype=float)
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, len(a), size=len(a))
        boot[b] = a[idx].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    v2 = pd.read_csv(V2)
    proto = pd.read_csv(PROTO)
    df = proto.merge(v2, on="case_id", how="inner")

    # Restrict to nonPH
    non_ph = df[df["label"] == 0].copy()
    non_ph["is_contrast"] = (non_ph["protocol"] == "contrast").astype(int)
    print(f"nonPH cases: {len(non_ph)} "
          f"(contrast={int(non_ph['is_contrast'].sum())}, "
          f"plain={int((1 - non_ph['is_contrast']).sum())})")

    # Derived ratios
    non_ph["artery_over_vein"] = non_ph["artery_vol_mL"] / non_ph["vein_vol_mL"].replace(0, np.nan)
    non_ph["artery_over_total_vessel"] = non_ph["artery_vol_mL"] / (
        non_ph["artery_vol_mL"] + non_ph["vein_vol_mL"]
    ).replace(0, np.nan)

    sets = {
        "v1_whole_lung_HU": [
            "lung_vol_mL",
            "whole_mean_HU", "whole_std_HU",
            "whole_HU_p5", "whole_HU_p25", "whole_HU_p50", "whole_HU_p75", "whole_HU_p95",
            "whole_LAA_950_frac", "whole_LAA_910_frac", "whole_LAA_856_frac",
        ],
        "v2_parenchyma_only": [
            "paren_mean_HU", "paren_std_HU",
            "paren_HU_p5", "paren_HU_p25", "paren_HU_p50", "paren_HU_p75", "paren_HU_p95",
            "paren_LAA_950_frac", "paren_LAA_910_frac", "paren_LAA_856_frac",
        ],
        "v2_paren_LAA_only": [
            "paren_LAA_950_frac", "paren_LAA_910_frac", "paren_LAA_856_frac",
        ],
        "v2_spatial_paren": [
            "apical_LAA_950_frac", "middle_LAA_950_frac", "basal_LAA_950_frac",
            "apical_basal_LAA950_gradient",
        ],
        "v2_per_structure_volumes": [
            "artery_vol_mL", "vein_vol_mL", "airway_vol_mL",
            "vessel_airway_over_lung",
        ],
        "v2_vessel_ratios": [
            "artery_over_vein", "artery_over_total_vessel",
        ],
        "v2_combined_no_HU": [
            "paren_LAA_950_frac", "paren_LAA_910_frac", "paren_LAA_856_frac",
            "apical_LAA_950_frac", "middle_LAA_950_frac", "basal_LAA_950_frac",
            "apical_basal_LAA950_gradient",
            "artery_over_vein", "artery_over_total_vessel",
        ],
    }

    rng = np.random.default_rng(SEED)
    results = {"n_cases": int(len(non_ph)),
               "n_contrast": int(non_ph["is_contrast"].sum()),
               "n_plain": int((1 - non_ph["is_contrast"]).sum()),
               "sets": {}}

    lines = [
        "# R4.1 Protocol decoder WITHIN nonPH only",
        "",
        "Critical methodological fix (Round 3 reviewer): protocol AUC across the full",
        "cohort conflates label↔protocol coupling (all 170 PH are contrast). This",
        "test restricts to **label=0** cases (27 contrast + 85 plain-scan) to isolate",
        "protocol leakage from label signal.",
        "",
        f"Cases: {results['n_cases']} "
        f"(contrast={results['n_contrast']}, plain-scan={results['n_plain']})",
        "",
        "5-fold stratified CV, 95% bootstrap CIs on mean CV AUC (2000 resamples):",
        "",
        "| Feature set | n_feats | n_cases | Protocol AUC LR (95% CI) | Protocol AUC GB (95% CI) |",
        "|---|---|---|---|---|",
    ]

    for name, feats in sets.items():
        sub = non_ph.dropna(subset=feats)
        if len(sub) < 20 or sub["is_contrast"].nunique() < 2:
            continue
        X = sub[feats].to_numpy()
        y = sub["is_contrast"].to_numpy()
        lr = cv_auc(X, y, "lr")
        gb = cv_auc(X, y, "gb")
        lo_lr, hi_lr = bootstrap_ci(lr[2], rng)
        lo_gb, hi_gb = bootstrap_ci(gb[2], rng)
        results["sets"][name] = {
            "n_feats": len(feats),
            "n_cases": int(len(sub)),
            "n_contrast": int(sub["is_contrast"].sum()),
            "n_plain": int((1 - sub["is_contrast"]).sum()),
            "protocol_lr": {"mean": lr[0], "std": lr[1], "ci95": [lo_lr, hi_lr]},
            "protocol_gb": {"mean": gb[0], "std": gb[1], "ci95": [lo_gb, hi_gb]},
        }
        lines.append(
            f"| `{name}` | {len(feats)} | {len(sub)} | "
            f"{lr[0]:.3f} [{lo_lr:.3f}, {hi_lr:.3f}] | "
            f"{gb[0]:.3f} [{lo_gb:.3f}, {hi_gb:.3f}] |"
        )

    # Find lowest protocol LR AUC set
    sets_ok = results["sets"]
    best = min(sets_ok.items(), key=lambda kv: kv[1]["protocol_lr"]["mean"])
    lines += [
        "",
        "## Interpretation",
        "",
        f"- Lowest LR protocol AUC (within nonPH): `{best[0]}` at "
        f"{best[1]['protocol_lr']['mean']:.3f} (95% CI "
        f"[{best[1]['protocol_lr']['ci95'][0]:.3f}, "
        f"{best[1]['protocol_lr']['ci95'][1]:.3f}]).",
        "- Compare to R3 numbers (across full 282) — if a feature set has similar",
        "  within-nonPH AUC, the R3 signal was real protocol; if within-nonPH drops",
        "  to ~0.5, the R3 signal was mostly label-shortcut.",
        "- Sample size is small (n=112); CIs are wider. A set with upper-CI below",
        "  0.7 is defensible as 'protocol-robust' for the contrast/plain comparison.",
        "",
        "**Warning**: 27 vs 85 is imbalanced (class balance 0.24). LR uses",
        "`class_weight=balanced` to mitigate; GB does not. Prefer LR numbers for",
        "the 'protocol-decodability' endpoint.",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(OUT_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
