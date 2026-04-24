"""R10 — Analyse GRL λ sweep: baseline + {0.5, 1.0, 2.0, 5.0}.

For each λ, compute within-nonPH protocol AUC (case-level OOF + bootstrap)
and within-contrast disease AUC. Table + best-operating-point recommendation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent.parent
SPLITS = ROOT / "data" / "splits_expanded_282"
CACHE_LIST = ROOT / "outputs" / "r5" / "cache_v2_tri_flat_list.txt"
PROTO = ROOT / "data" / "case_protocol.csv"
OUT_MD = ROOT / "outputs" / "r10" / "R10_adv_sweep.md"
OUT_JSON = ROOT / "outputs" / "r10" / "R10_adv_sweep.json"

SETTINGS = {
    "baseline (λ=0)": ROOT / "outputs" / "r9" / "embeddings_full",
    "λ=0.5": ROOT / "outputs" / "r10" / "embeddings_adv",
    "λ=1.0": ROOT / "outputs" / "r10" / "embeddings_adv_l1.0",
    "λ=2.0": ROOT / "outputs" / "r10" / "embeddings_adv_l2.0",
    "λ=5.0": ROOT / "outputs" / "r10" / "embeddings_adv_l5.0",
}

SEED = 20260424


def load_emb(emb_dir):
    if not emb_dir.exists():
        return None, None, None
    cached = set(c.strip() for c in CACHE_LIST.read_text(encoding="utf-8").splitlines() if c.strip())
    all_e, all_y, all_c = [], [], []
    for k in range(1, 6):
        f = emb_dir / f"emb_gcn_only_rep1_fold{k}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        val_ids = [c.strip() for c in (SPLITS / f"fold_{k}" / "val.txt").read_text().splitlines() if c.strip()]
        val_ids = [c for c in val_ids if c in cached]
        if len(val_ids) != len(d["embeddings"]):
            continue
        all_e.append(d["embeddings"])
        all_y.extend(d["y_true"].tolist())
        all_c.extend(val_ids)
    if not all_e:
        return None, None, None
    return np.concatenate(all_e), np.array(all_y, int), all_c


def oof_auc_with_boot(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=5000, class_weight="balanced")
        s = StandardScaler().fit(X[tr])
        clf.fit(s.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(s.transform(X[te]))[:, 1]
    auc = roc_auc_score(y, oof)
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))
        try:
            boots.append(roc_auc_score(y[idx], oof[idx]))
        except ValueError:
            pass
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(auc), float(lo), float(hi)


def main():
    proto_map = dict(zip(*zip(*[(r["case_id"], r["protocol"]) for _, r in pd.read_csv(PROTO).iterrows()])))
    results = {}
    for name, emb_dir in SETTINGS.items():
        emb, y, cids = load_emb(emb_dir)
        if emb is None:
            results[name] = {"missing": True}
            continue
        protocols = np.array([proto_map.get(c, "unknown") for c in cids])
        is_contrast = (protocols == "contrast").astype(int)
        nph_mask = y == 0
        Xn, yn = emb[nph_mask], is_contrast[nph_mask]
        cmask = is_contrast == 1
        Xc, yc = emb[cmask], y[cmask]
        proto_auc = oof_auc_with_boot(Xn, yn) if len(yn) and len(set(yn)) == 2 else (float("nan"),)*3
        dis_auc = oof_auc_with_boot(Xc, yc) if len(yc) and len(set(yc)) == 2 else (float("nan"),)*3
        results[name] = {
            "n_nonph": int(nph_mask.sum()),
            "n_contrast": int(cmask.sum()),
            "protocol_auc": {"mean": proto_auc[0], "ci95": [proto_auc[1], proto_auc[2]]},
            "disease_auc": {"mean": dis_auc[0], "ci95": [dis_auc[1], dis_auc[2]]},
        }
        print(f"{name}: protocol {proto_auc[0]:.3f} [{proto_auc[1]:.3f},{proto_auc[2]:.3f}]  disease {dis_auc[0]:.3f}")

    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        "# R10 — GRL λ sweep results",
        "",
        "Protocol AUC WITHIN nonPH + Disease AUC WITHIN contrast (case-level OOF + bootstrap CI)",
        "",
        "| Setting | n nonPH | Protocol AUC (CI) | Disease AUC (CI) |",
        "|---|---|---|---|",
    ]
    for name, r in results.items():
        if "missing" in r:
            lines.append(f"| {name} | — | (embeddings missing) | — |")
            continue
        p = r["protocol_auc"]
        d = r["disease_auc"]
        lines.append(
            f"| {name} | {r['n_nonph']} | "
            f"{p['mean']:.3f} [{p['ci95'][0]:.3f}, {p['ci95'][1]:.3f}] | "
            f"{d['mean']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}] |"
        )
    # Find best operating point
    valid = {n: r for n, r in results.items() if "missing" not in r}
    if valid:
        # Best = lowest protocol AUC subject to disease AUC >= 0.75
        best = min(
            (n for n, r in valid.items() if r["disease_auc"]["mean"] >= 0.75),
            key=lambda n: valid[n]["protocol_auc"]["mean"],
            default=None,
        )
        if best is None:
            best = min(valid, key=lambda n: valid[n]["protocol_auc"]["mean"])
        b = valid[best]
        lines += [
            "",
            f"**Best operating point** (lowest protocol AUC with disease ≥ 0.75 if possible): "
            f"`{best}` — protocol {b['protocol_auc']['mean']:.3f}, disease {b['disease_auc']['mean']:.3f}",
            f"- Target was protocol ≤ 0.60. "
            + ("**✅ MET**" if b["protocol_auc"]["mean"] <= 0.60 else "**❌ NOT MET** — try stronger λ, warm-up schedule, or different architecture."),
        ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
