"""R5.2 — Protocol decoder on EXACT GCN-input features WITHIN nonPH only.

Round 4 reviewer's #1 must-fix: protocol AUC must be measured on the actual
features the GCN consumes, not on lung-feature scalar proxies, and within
label=0 only.

Input: `outputs/r5/graph_stats_v2.json` (per-case node-feature aggregates +
edge-attr aggregates from cache_v2_tri_flat/*.pkl, computed by remote
`/tmp/extract_graph_stats.py`).

Test:
  - 5-fold stratified CV with class_weight=balanced LR + GB
  - Subset: 112 nonPH cases (27 contrast + 85 plain-scan)
  - Protocol AUC + 95% bootstrap CI

If LR AUC CI excludes 0.7, the actual GCN inputs do carry decodable protocol
signal even within nonPH — domain-adversarial training is then warranted.
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
GS = ROOT / "outputs" / "r5" / "graph_stats_v2.json"
PROTO = ROOT / "data" / "case_protocol.csv"
OUT_MD = ROOT / "outputs" / "r5" / "R5_gcn_feature_within_nonph.md"
OUT_JSON = ROOT / "outputs" / "r5" / "R5_gcn_feature_within_nonph.json"

SEED = 20260423


def cv_auc(X, y, model, seed=SEED):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=5000, class_weight="balanced") if model == "lr" \
            else GradientBoostingClassifier(random_state=seed)
        s = StandardScaler().fit(X[tr])
        clf.fit(s.transform(X[tr]), y[tr])
        p = clf.predict_proba(s.transform(X[te]))[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)), float(np.std(aucs)), aucs


def boot_ci(aucs, rng, n=2000):
    a = np.asarray(aucs, float)
    boot = np.array([a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)])
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> None:
    gs = json.loads(GS.read_text())
    proto = pd.read_csv(PROTO)
    rows = []
    for _, r in proto.iterrows():
        cid = r["case_id"]
        if cid not in gs or "error" in gs[cid]:
            continue
        d = {"case_id": cid, "label": int(r["label"]), "protocol": r["protocol"]}
        d.update(gs[cid])
        rows.append(d)
    df = pd.DataFrame(rows)
    feats = [c for c in df.columns if c not in ("case_id", "label", "protocol", "n_feats", "max_degree", "n_edges")]
    df = df.dropna(subset=feats)
    print(f"All cases with graph stats: {len(df)}; features: {len(feats)}")

    rng = np.random.default_rng(SEED)
    out = {"feature_count": len(feats), "subsets": {}}

    # Subset 1: within-nonPH only (the key Round-4 reviewer endpoint)
    sub = df[df["label"] == 0].copy()
    sub["is_contrast"] = (sub["protocol"] == "contrast").astype(int)
    print(f"within-nonPH: n={len(sub)}, contrast={int(sub['is_contrast'].sum())}, plain={int((1-sub['is_contrast']).sum())}")
    if sub["is_contrast"].nunique() == 2:
        X = sub[feats].to_numpy(); y = sub["is_contrast"].to_numpy()
        lr = cv_auc(X, y, "lr"); gb = cv_auc(X, y, "gb")
        lr_lo, lr_hi = boot_ci(lr[2], rng); gb_lo, gb_hi = boot_ci(gb[2], rng)
        out["subsets"]["within_nonph_protocol"] = {
            "n": int(len(sub)),
            "n_feats": len(feats),
            "protocol_lr": {"mean": lr[0], "ci95": [lr_lo, lr_hi]},
            "protocol_gb": {"mean": gb[0], "ci95": [gb_lo, gb_hi]},
        }

    # Subset 2: within-contrast disease (positive control — disease should be detectable)
    sub2 = df[df["protocol"] == "contrast"].copy()
    if sub2["label"].nunique() == 2:
        X = sub2[feats].to_numpy(); y = sub2["label"].to_numpy()
        lr = cv_auc(X, y, "lr"); gb = cv_auc(X, y, "gb")
        lr_lo, lr_hi = boot_ci(lr[2], rng); gb_lo, gb_hi = boot_ci(gb[2], rng)
        out["subsets"]["contrast_only_disease"] = {
            "n": int(len(sub2)),
            "disease_lr": {"mean": lr[0], "ci95": [lr_lo, lr_hi]},
            "disease_gb": {"mean": gb[0], "ci95": [gb_lo, gb_hi]},
        }

    # Subset 3: full-cohort protocol (to confirm label-shortcut effect)
    sub3 = df.copy()
    sub3["is_contrast"] = (sub3["protocol"] == "contrast").astype(int)
    if sub3["is_contrast"].nunique() == 2:
        X = sub3[feats].to_numpy(); y = sub3["is_contrast"].to_numpy()
        lr = cv_auc(X, y, "lr"); gb = cv_auc(X, y, "gb")
        out["subsets"]["full_cohort_protocol"] = {
            "n": int(len(sub3)),
            "protocol_lr_mean": lr[0],
            "protocol_gb_mean": gb[0],
        }

    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        "# R5.2 — Protocol decoder on EXACT GCN-input features (cache_v2_tri_flat)",
        "",
        f"Per-case graph aggregates from `cache_v2_tri_flat/*.pkl` ({len(feats)} features:",
        "n_nodes, mean_degree, x{0..12}_mean/std/p90, e{0..2}_mean/p90).",
        "",
        "## Within-nonPH protocol AUC (HONEST W1 ENDPOINT)",
        "",
    ]
    if "within_nonph_protocol" in out["subsets"]:
        s = out["subsets"]["within_nonph_protocol"]
        lines += [
            f"- n = {s['n']} (cases with valid graph stats AND label=0)",
            f"- LR protocol AUC: **{s['protocol_lr']['mean']:.3f}** (95% CI [{s['protocol_lr']['ci95'][0]:.3f}, {s['protocol_lr']['ci95'][1]:.3f}])",
            f"- GB protocol AUC: **{s['protocol_gb']['mean']:.3f}** (95% CI [{s['protocol_gb']['ci95'][0]:.3f}, {s['protocol_gb']['ci95'][1]:.3f}])",
            "",
            "**Reading**: this is the actual W1 endpoint per Round-4 reviewer memory.",
            f"If LR CI upper bound < 0.7 → the GCN's input features are protocol-robust under linear decoding.",
            f"If GB CI > 0.7 → non-linear protocol signal exists; an adversarial debiasing arm is needed.",
            "",
        ]
    if "contrast_only_disease" in out["subsets"]:
        s = out["subsets"]["contrast_only_disease"]
        lines += [
            "## Within-contrast disease AUC (positive control)",
            "",
            f"- n = {s['n']}",
            f"- LR disease AUC: {s['disease_lr']['mean']:.3f} [{s['disease_lr']['ci95'][0]:.3f}, {s['disease_lr']['ci95'][1]:.3f}]",
            f"- GB disease AUC: {s['disease_gb']['mean']:.3f} [{s['disease_gb']['ci95'][0]:.3f}, {s['disease_gb']['ci95'][1]:.3f}]",
            "",
        ]
    if "full_cohort_protocol" in out["subsets"]:
        s = out["subsets"]["full_cohort_protocol"]
        lines += [
            "## Full-cohort protocol AUC (label-shortcut sanity check)",
            "",
            f"- n = {s['n']}",
            f"- LR protocol AUC (full): {s['protocol_lr_mean']:.3f} (expected high, as label↔protocol coupled)",
            f"- GB protocol AUC (full): {s['protocol_gb_mean']:.3f}",
            "",
        ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
