"""R8.1 — GCN-feature-level exclusion sensitivity on the full 282-case cohort.

Round 7 reviewer: "GCN-level placeholder/exclusion sensitivity on the full 282
with paired inference on identical case sets."

Without GPU access we cannot retrain the GCN. Instead we use the 47-dim
graph-aggregate features from `outputs/r5/graph_stats_v2.json` (per-case
node-feature moments + edge-attr aggregates, computed from cache_v2_tri_flat
pkls — the same features the GCN consumes, reduced to fixed-length).

Two cohorts:
  A — current: 243 cases that have valid pkls
  B — full-282: 39 missing cases imputed with all-zero features
      (degraded-graph representation: n_nodes=1, no edges, HU=0, etc.)

Report disease AUC + protocol AUC on each, both on the full cohort and
restricted to contrast-only. If the A vs B delta is within the bootstrap
CI half-width, exclusion is not driving the claim.
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

ROOT = Path(__file__).parent.parent.parent
GS = ROOT / "outputs" / "r5" / "graph_stats_v2.json"
PROTO = ROOT / "data" / "case_protocol.csv"
OUT_MD = ROOT / "outputs" / "r8" / "R8_gcn_exclusion.md"
OUT_JSON = ROOT / "outputs" / "r8" / "R8_gcn_exclusion.json"
OUT_MD.parent.mkdir(parents=True, exist_ok=True)

SEED = 20260424


def cv_auc(X, y, model="lr", seed=SEED):
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
    return (float(np.percentile([a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)], 2.5)),
            float(np.percentile([a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)], 97.5)))


def main() -> None:
    gs = json.loads(GS.read_text(encoding="utf-8"))
    proto = pd.read_csv(PROTO)

    # Build feature dataframe with cohort A (243) and cohort B (282)
    all_feats_union = set()
    for cid, d in gs.items():
        if "error" not in d:
            all_feats_union.update(k for k in d if isinstance(d[k], (int, float)))
    all_feats = sorted(all_feats_union)

    rows_A, rows_B = [], []
    for _, r in proto.iterrows():
        cid = r["case_id"]
        entry = gs.get(cid, {})
        has_graph = "error" not in entry and any(k in entry for k in all_feats)
        base = {"case_id": cid, "label": int(r["label"]), "protocol": r["protocol"]}
        if has_graph:
            feats = {k: float(entry.get(k, 0.0)) for k in all_feats}
            rows_A.append({**base, **feats})
            rows_B.append({**base, **feats, "degraded": 0})
        else:
            # Impute: all zero → mimics "degraded graph" (n_nodes=1, no edges, HU=0)
            feats = {k: 0.0 for k in all_feats}
            rows_B.append({**base, **feats, "degraded": 1})

    df_A = pd.DataFrame(rows_A)
    df_B = pd.DataFrame(rows_B)
    print(f"Cohort A (in-cache): {len(df_A)}  |  Cohort B (full + degraded): {len(df_B)}  "
          f"(degraded={int(df_B['degraded'].sum())})")

    feats = all_feats
    rng = np.random.default_rng(SEED)
    results = {"cohorts": {}, "delta": {}}

    for name, df in (("A_in_cache_243", df_A), ("B_full_282_degraded", df_B)):
        X = df[feats].to_numpy()
        y_d = df["label"].to_numpy()
        df_c = df[df["protocol"] == "contrast"]
        Xc = df_c[feats].to_numpy()
        yc = df_c["label"].to_numpy()
        d_full = cv_auc(X, y_d, "lr")
        d_cont = cv_auc(Xc, yc, "lr")
        d_full_gb = cv_auc(X, y_d, "gb")
        d_cont_gb = cv_auc(Xc, yc, "gb")
        d_full_lo, d_full_hi = boot_ci(d_full[2], rng)
        d_cont_lo, d_cont_hi = boot_ci(d_cont[2], rng)
        # Also protocol within-nonPH
        sub_nph = df[df["label"] == 0].copy()
        sub_nph["is_contrast"] = (sub_nph["protocol"] == "contrast").astype(int)
        if sub_nph["is_contrast"].nunique() == 2:
            Xn = sub_nph[feats].to_numpy()
            yn = sub_nph["is_contrast"].to_numpy()
            p_nph = cv_auc(Xn, yn, "lr")
            p_nph_lo, p_nph_hi = boot_ci(p_nph[2], rng)
        else:
            p_nph = (float("nan"), float("nan"), [])
            p_nph_lo, p_nph_hi = (float("nan"), float("nan"))
        results["cohorts"][name] = {
            "n": int(len(df)),
            "n_contrast": int(len(df_c)),
            "disease_full_lr": {"mean": d_full[0], "ci95": [d_full_lo, d_full_hi]},
            "disease_full_gb": d_full_gb[0],
            "disease_contrast_lr": {"mean": d_cont[0], "ci95": [d_cont_lo, d_cont_hi]},
            "disease_contrast_gb": d_cont_gb[0],
            "protocol_nonph_lr": {"mean": p_nph[0], "ci95": [p_nph_lo, p_nph_hi]},
        }

    a, b = results["cohorts"]["A_in_cache_243"], results["cohorts"]["B_full_282_degraded"]
    results["delta"] = {
        "disease_full_lr": b["disease_full_lr"]["mean"] - a["disease_full_lr"]["mean"],
        "disease_contrast_lr": b["disease_contrast_lr"]["mean"] - a["disease_contrast_lr"]["mean"],
        "protocol_nonph_lr": b["protocol_nonph_lr"]["mean"] - a["protocol_nonph_lr"]["mean"],
    }

    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        "# R8.1 — GCN-feature exclusion sensitivity (full 282 with degraded graphs)",
        "",
        "47-dim graph-aggregate features (n_nodes/mean_degree/node-HU percentiles/edge-attr)",
        "from cache_v2_tri_flat. Cohort B adds 39 missing cases with all-zero (degraded) features.",
        "",
        "| Cohort | n | n_contrast | disease LR full (CI) | disease LR contrast (CI) | protocol within-nonPH LR (CI) |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in results["cohorts"].items():
        df_lr = r["disease_full_lr"]
        dc_lr = r["disease_contrast_lr"]
        pn = r["protocol_nonph_lr"]
        lines.append(
            f"| `{name}` | {r['n']} | {r['n_contrast']} | "
            f"{df_lr['mean']:.3f} [{df_lr['ci95'][0]:.3f}, {df_lr['ci95'][1]:.3f}] | "
            f"{dc_lr['mean']:.3f} [{dc_lr['ci95'][0]:.3f}, {dc_lr['ci95'][1]:.3f}] | "
            f"{pn['mean']:.3f} [{pn['ci95'][0]:.3f}, {pn['ci95'][1]:.3f}] |"
        )
    d = results["delta"]
    lines += [
        "",
        "## Delta (B − A)",
        "",
        f"- Δ disease AUC (full cohort): **{d['disease_full_lr']:+.3f}**",
        f"- Δ disease AUC (contrast-only): **{d['disease_contrast_lr']:+.3f}**",
        f"- Δ protocol AUC (within-nonPH): **{d['protocol_nonph_lr']:+.3f}**",
        "",
        "## Reading",
        "",
        "- If |Δ| on disease-contrast is smaller than the bootstrap CI half-width",
        "  (typically ~0.04 at n=189), the disease claim is robust to the placeholder",
        "  exclusion choice at the feature-classifier level.",
        "- The GCN itself cannot be retrained this round (both GPUs busy). A full",
        "  GCN-training exclusion sensitivity is queued for Round 9 once GPU frees.",
        "",
        "**Caveat**: this is a classifier-on-graph-aggregates proxy for the GCN. A",
        "true GCN-retraining sensitivity analysis may show larger or smaller deltas.",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
