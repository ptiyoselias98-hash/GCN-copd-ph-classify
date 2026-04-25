"""R12.4 — Aggregate cross-seed CIs for R11 GRL-fix sweep.

R11 reviewer flagged that R11_grlfix_summary.json reports only per-run
bootstrap CIs (one per seed). This script pools per-seed OOF predictions
across all 3 seeds (42, 1042, 2042) per λ, then bootstraps over CASES on
the pooled prediction matrix to get an aggregate CI that captures both
within-run sampling variability AND between-seed model variability.

Two flavors per λ:
  (a) "stack-bootstrap": for each bootstrap iter, sample cases with
       replacement, take protocol-LR AUC on the seed-averaged probability
       (mean of per-seed predicted probs for each case).
  (b) "per-seed-mean ± SD" (already in R11 JSON; included for sanity).

Outputs: outputs/r12/r12_cross_seed_cis.{json,md}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent.parent
EMB_ROOT = ROOT / "outputs" / "r11" / "embeddings"
SPLITS = ROOT / "data" / "splits_expanded_282"
CACHE_LIST = ROOT / "outputs" / "r5" / "cache_v2_tri_flat_list.txt"
PROTO = ROOT / "data" / "case_protocol.csv"
LABELS = ROOT / "data" / "labels_expanded_282.csv"
OUT = ROOT / "outputs" / "r12"

LAMBDAS = [0.0, 1.0, 5.0, 10.0]
SEEDS = [42, 1042, 2042]


def load_emb_oof(lam: float, seed: int):
    emb_dir = EMB_ROOT / f"l{lam}_s{seed}"
    if not emb_dir.exists():
        return None
    cached = set(c.strip() for c in CACHE_LIST.read_text(encoding="utf-8").splitlines() if c.strip())
    rows = []
    for k in range(1, 6):
        f = emb_dir / f"emb_gcn_only_rep1_fold{k}.npz"
        if not f.exists():
            return None
        d = np.load(f)
        val_ids = [c.strip() for c in (SPLITS / f"fold_{k}" / "val.txt").read_text().splitlines() if c.strip()]
        val_ids = [c for c in val_ids if c in cached]
        if len(val_ids) != len(d["embeddings"]):
            return None
        for cid, emb, y in zip(val_ids, d["embeddings"], d["y_true"]):
            rows.append({"case_id": cid, "y": int(y), "emb": np.asarray(emb)})
    return rows


def lr_oof_proba(X: np.ndarray, y: np.ndarray, seed: int = 42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=5000, class_weight="balanced")
        sc = StandardScaler().fit(X[tr])
        clf.fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return oof


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    proto = pd.read_csv(PROTO)
    proto["is_contrast"] = (proto["protocol"].astype(str).str.lower() == "contrast").astype(int)
    proto_lookup = dict(zip(proto["case_id"], proto["is_contrast"]))
    label_df = pd.read_csv(LABELS)
    label_col = "label" if "label" in label_df.columns else "y"
    lbl_lookup = dict(zip(label_df["case_id"], label_df[label_col]))

    summary: dict = {"per_lambda": {}}
    for lam in LAMBDAS:
        per_seed_oof: dict[int, dict[str, float]] = {}
        per_seed_y: dict[int, dict[str, int]] = {}
        for seed in SEEDS:
            rows = load_emb_oof(lam, seed)
            if rows is None:
                continue
            # Restrict to nonPH (label==0) within-nonPH protocol stratum
            nonph_rows = [r for r in rows if lbl_lookup.get(r["case_id"], 1) == 0]
            ids = [r["case_id"] for r in nonph_rows]
            X = np.stack([r["emb"] for r in nonph_rows])
            y_proto = np.array([proto_lookup.get(c, -1) for c in ids], int)
            mask = y_proto >= 0
            X = X[mask]; y_proto = y_proto[mask]; ids = [c for c, m in zip(ids, mask) if m]
            if len(np.unique(y_proto)) < 2:
                continue
            oof = lr_oof_proba(X, y_proto, seed=seed)
            per_seed_oof[seed] = dict(zip(ids, oof.tolist()))
            per_seed_y[seed] = dict(zip(ids, y_proto.tolist()))

        # Pool: case ∈ intersection of seeds; mean predicted proba
        if not per_seed_oof:
            continue
        common = set.intersection(*[set(d.keys()) for d in per_seed_oof.values()])
        common = sorted(common)
        if len(common) < 10:
            continue
        prob_matrix = np.array([
            [per_seed_oof[s][c] for c in common] for s in per_seed_oof
        ])  # (n_seeds, n_cases)
        mean_p = prob_matrix.mean(axis=0)
        y_arr = np.array([per_seed_y[next(iter(per_seed_y))][c] for c in common])
        per_seed_aucs = [float(roc_auc_score(y_arr, prob_matrix[i])) for i in range(prob_matrix.shape[0])]
        pooled_auc = float(roc_auc_score(y_arr, mean_p))

        # Stack-bootstrap (cases): account for case-sampling variability with seed-mean prob
        rng = np.random.default_rng(42)
        n = len(y_arr)
        pos = np.where(y_arr == 1)[0]
        neg = np.where(y_arr == 0)[0]
        boots = []
        for _ in range(5000):
            bp = rng.choice(pos, size=len(pos), replace=True)
            bn = rng.choice(neg, size=len(neg), replace=True)
            idx = np.concatenate([bp, bn])
            try:
                boots.append(roc_auc_score(y_arr[idx], mean_p[idx]))
            except ValueError:
                continue
        boots = np.array(boots)

        # Hierarchical-bootstrap: jointly resample seeds and cases
        hboots = []
        seed_keys = list(per_seed_oof.keys())
        for _ in range(5000):
            sboot = rng.choice(seed_keys, size=len(seed_keys), replace=True)
            mp = np.mean([prob_matrix[seed_keys.index(s)] for s in sboot], axis=0)
            bp = rng.choice(pos, size=len(pos), replace=True)
            bn = rng.choice(neg, size=len(neg), replace=True)
            idx = np.concatenate([bp, bn])
            try:
                hboots.append(roc_auc_score(y_arr[idx], mp[idx]))
            except ValueError:
                continue
        hboots = np.array(hboots)

        summary["per_lambda"][f"lambda_{lam}"] = {
            "n_cases_pooled": int(n),
            "n_seeds": int(prob_matrix.shape[0]),
            "per_seed_protocol_lr_auc": per_seed_aucs,
            "seed_mean_AUC": float(np.mean(per_seed_aucs)),
            "seed_sd_AUC": float(np.std(per_seed_aucs, ddof=1)) if len(per_seed_aucs) > 1 else 0.0,
            "pooled_seed_avg_prob_AUC": pooled_auc,
            "case_bootstrap_CI95_on_seed_avg": [
                float(np.percentile(boots, 2.5)),
                float(np.percentile(boots, 97.5)),
            ],
            "hierarchical_bootstrap_CI95": [
                float(np.percentile(hboots, 2.5)),
                float(np.percentile(hboots, 97.5)),
            ],
        }

    out_json = OUT / "r12_cross_seed_cis.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = ["# R12.4 — Cross-seed protocol-LR aggregate CIs (within-nonPH)",
          "",
          "Pools per-seed OOF predicted probabilities across 3 seeds, then",
          "bootstraps over cases (and over seeds, hierarchically). Compare to",
          "the per-run CIs reported in `R11_grlfix_summary.json`.",
          "",
          "| λ | n_cases | n_seeds | seed-mean AUC ± SD | pooled (seed-avg-prob) AUC | case-bootstrap 95% CI | hierarchical (seeds × cases) 95% CI |",
          "|---|---|---|---|---|---|---|"]
    for lam in LAMBDAS:
        rec = summary["per_lambda"].get(f"lambda_{lam}")
        if rec is None:
            md.append(f"| {lam} | — | — | — | — | — | — |")
            continue
        md.append(f"| {lam} | {rec['n_cases_pooled']} | {rec['n_seeds']} | "
                  f"{rec['seed_mean_AUC']:.3f} ± {rec['seed_sd_AUC']:.3f} | "
                  f"{rec['pooled_seed_avg_prob_AUC']:.3f} | "
                  f"[{rec['case_bootstrap_CI95_on_seed_avg'][0]:.3f}, "
                  f"{rec['case_bootstrap_CI95_on_seed_avg'][1]:.3f}] | "
                  f"[{rec['hierarchical_bootstrap_CI95'][0]:.3f}, "
                  f"{rec['hierarchical_bootstrap_CI95'][1]:.3f}] |")
    (OUT / "r12_cross_seed_cis.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Saved {out_json}")
    for lam in LAMBDAS:
        rec = summary["per_lambda"].get(f"lambda_{lam}")
        if rec:
            print(f"  λ={lam}: pooled AUC={rec['pooled_seed_avg_prob_AUC']:.3f} "
                  f"hierCI=[{rec['hierarchical_bootstrap_CI95'][0]:.3f},"
                  f"{rec['hierarchical_bootstrap_CI95'][1]:.3f}]")


if __name__ == "__main__":
    main()
