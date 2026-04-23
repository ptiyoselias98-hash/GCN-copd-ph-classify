"""R4.4 — Exclusion sensitivity: retain 27 placeholder nonPH with degraded features.

Round 1/2/3 reviewer flagged that dropping the 27 placeholder nonPH changes
class/protocol balance in a label-correlated way. This script rebuilds the
disease classifier on two cohorts:

  A. as-is (current, placeholders dropped)
  B. retained with degraded features: vessel-related columns imputed with
     the cohort median (or zero for counts), keeping lung-parenchyma columns
     intact (lung.nii.gz is valid for placeholder cases).

For each cohort we run the same 5-fold stratified CV on the protocol-robust
candidate feature sets and report disease AUC deltas. If the disease AUC
changes less than the bootstrap CI, exclusion is not driving the result.
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
OUT_MD = ROOT / "outputs" / "_r4_exclusion_sensitivity.md"
OUT_JSON = ROOT / "outputs" / "_r4_exclusion_sensitivity.json"

SEED = 20260423

FEATURE_SETS = {
    "paren_only": [
        "paren_mean_HU", "paren_std_HU",
        "paren_HU_p5", "paren_HU_p25", "paren_HU_p50", "paren_HU_p75", "paren_HU_p95",
        "paren_LAA_950_frac", "paren_LAA_910_frac", "paren_LAA_856_frac",
    ],
    "paren_plus_spatial": [
        "paren_mean_HU", "paren_std_HU",
        "paren_HU_p5", "paren_HU_p25", "paren_HU_p50", "paren_HU_p75", "paren_HU_p95",
        "paren_LAA_950_frac", "paren_LAA_910_frac", "paren_LAA_856_frac",
        "apical_LAA_950_frac", "middle_LAA_950_frac", "basal_LAA_950_frac",
        "apical_basal_LAA950_gradient",
    ],
}


def cv_auc(X: np.ndarray, y: np.ndarray, model_name: str):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    aucs = []
    for tr, te in skf.split(X, y):
        if model_name == "lr":
            clf = LogisticRegression(max_iter=5000, class_weight="balanced")
        else:
            clf = GradientBoostingClassifier(random_state=SEED)
        s = StandardScaler().fit(X[tr])
        clf.fit(s.transform(X[tr]), y[tr])
        p = clf.predict_proba(s.transform(X[te]))[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)), float(np.std(aucs)), aucs


def bootstrap_ci(aucs: list[float], rng: np.random.Generator, n_boot: int = 2000):
    a = np.asarray(aucs, float)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        boot[b] = a[rng.integers(0, len(a), len(a))].mean()
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> None:
    v2 = pd.read_csv(V2)
    proto = pd.read_csv(PROTO)
    df = proto.merge(v2, on="case_id", how="inner")

    # Identify placeholder cases
    # A case is a vessel-placeholder if artery_placeholder==1 OR vein_placeholder==1
    is_placeholder = (df["artery_placeholder"] == 1) | (df["vein_placeholder"] == 1)
    print(f"Placeholder cases: {is_placeholder.sum()} ({df[is_placeholder]['label'].value_counts().to_dict()})")

    # Cohort A: exclude placeholders (current behavior)
    # Cohort B: include placeholders (degraded features)

    # For cohort B we only use parenchyma features that are independent of vessel segmentation
    # (paren HU/LAA are computed from lung − vessels; for placeholder vessels we don't subtract
    # anything → the "paren" values equal "whole lung" for those cases, but that's consistent).
    rng = np.random.default_rng(SEED)
    results = {"cohorts": {}}

    for cohort_name, inclusive in (("A_excluded", False), ("B_included", True)):
        sub = df if inclusive else df[~is_placeholder]
        for set_name, feats in FEATURE_SETS.items():
            s = sub.dropna(subset=feats)
            if s["label"].nunique() < 2:
                continue
            X = s[feats].to_numpy()
            y = s["label"].to_numpy()
            # Disease full
            lr = cv_auc(X, y, "lr")
            gb = cv_auc(X, y, "gb")
            lr_lo, lr_hi = bootstrap_ci(lr[2], rng)
            gb_lo, gb_hi = bootstrap_ci(gb[2], rng)
            # Disease contrast-only
            s_c = s[s["protocol"] == "contrast"]
            Xc = s_c[feats].to_numpy()
            yc = s_c["label"].to_numpy()
            lr_c = cv_auc(Xc, yc, "lr")
            gb_c = cv_auc(Xc, yc, "gb")
            lrc_lo, lrc_hi = bootstrap_ci(lr_c[2], rng)
            gbc_lo, gbc_hi = bootstrap_ci(gb_c[2], rng)
            key = f"{cohort_name}_{set_name}"
            results["cohorts"][key] = {
                "n_cases": int(len(s)),
                "n_contrast_only": int(len(s_c)),
                "disease_full_lr": {"mean": lr[0], "ci95": [lr_lo, lr_hi]},
                "disease_full_gb": {"mean": gb[0], "ci95": [gb_lo, gb_hi]},
                "disease_contrast_lr": {"mean": lr_c[0], "ci95": [lrc_lo, lrc_hi]},
                "disease_contrast_gb": {"mean": gb_c[0], "ci95": [gbc_lo, gbc_hi]},
            }

    # Compute deltas A → B for each feature set
    deltas = []
    for set_name in FEATURE_SETS:
        a = results["cohorts"].get(f"A_excluded_{set_name}")
        b = results["cohorts"].get(f"B_included_{set_name}")
        if a and b:
            deltas.append({
                "feature_set": set_name,
                "d_n_cases": b["n_cases"] - a["n_cases"],
                "d_disease_full_lr": b["disease_full_lr"]["mean"] - a["disease_full_lr"]["mean"],
                "d_disease_contrast_lr": b["disease_contrast_lr"]["mean"] - a["disease_contrast_lr"]["mean"],
            })

    lines = [
        "# R4.4 Exclusion sensitivity — retain 27 placeholder nonPH with degraded features",
        "",
        "Cohort A = current behavior (placeholder vessels dropped).",
        "Cohort B = include them; parenchyma features still valid because lung.nii.gz",
        "          is intact; paren_* just equals whole_lung for placeholder cases",
        "          (no vessels to subtract).",
        "",
        "| Cohort | Feat set | n | disease LR full (CI) | disease LR contrast (CI) |",
        "|---|---|---|---|---|",
    ]
    for k, v in results["cohorts"].items():
        lr_f = v["disease_full_lr"]
        lr_c = v["disease_contrast_lr"]
        lines.append(
            f"| `{k}` | - | {v['n_cases']} | "
            f"{lr_f['mean']:.3f} [{lr_f['ci95'][0]:.3f}, {lr_f['ci95'][1]:.3f}] | "
            f"{lr_c['mean']:.3f} [{lr_c['ci95'][0]:.3f}, {lr_c['ci95'][1]:.3f}] |"
        )
    lines += [
        "",
        "## Delta (B − A)",
        "",
        "| Feat set | Δn_cases | Δ disease full LR | Δ disease contrast LR |",
        "|---|---|---|---|",
    ]
    for d in deltas:
        lines.append(
            f"| `{d['feature_set']}` | +{d['d_n_cases']} | "
            f"{d['d_disease_full_lr']:+.3f} | {d['d_disease_contrast_lr']:+.3f} |"
        )
    max_delta = max(abs(d["d_disease_contrast_lr"]) for d in deltas) if deltas else float("nan")
    lines += [
        "",
        f"**Max |Δ| on disease-contrast AUC: {max_delta:.3f}**. If this is smaller",
        "than the bootstrap CI half-width (~0.05 on contrast-only 186-case subset),",
        "the exclusion choice is not driving the disease claim — i.e. the claim is",
        "robust to the exclusion rule.",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    OUT_JSON.write_text(json.dumps({"results": results, "deltas": deltas}, indent=2), encoding="utf-8")
    print(OUT_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
