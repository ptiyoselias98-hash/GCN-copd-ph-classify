"""R10 — Protocol AUC on DEBIASED GCN embeddings (post-GRL training).

Same pipeline as R9_embedding_protocol.py but reads embeddings from the
adversarially-trained arm_a (`outputs/r10/embeddings_adv/`). Compares
against R9 baseline embeddings to show whether adversarial debiasing
reduced within-nonPH protocol AUC.

Case-level bootstrap CI (R9 used fold-level, per Round-9 reviewer flag).
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
EMB_ADV = ROOT / "outputs" / "r10" / "embeddings_adv"
EMB_BASELINE = ROOT / "outputs" / "r9" / "embeddings_full"
SPLITS = ROOT / "data" / "splits_expanded_282"
CACHE_LIST = ROOT / "outputs" / "r5" / "cache_v2_tri_flat_list.txt"
PROTO = ROOT / "data" / "case_protocol.csv"
OUT_MD = ROOT / "outputs" / "r10" / "R10_adv_protocol_test.md"
OUT_JSON = ROOT / "outputs" / "r10" / "R10_adv_protocol_test.json"

SEED = 20260424


def load_emb_with_meta(emb_dir: Path):
    cached = set(c.strip() for c in CACHE_LIST.read_text(encoding="utf-8").splitlines() if c.strip())
    all_emb, all_y, all_cids = [], [], []
    for k in range(1, 6):
        f = emb_dir / f"emb_gcn_only_rep1_fold{k}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        val_ids = [c.strip() for c in (SPLITS / f"fold_{k}" / "val.txt").read_text().splitlines() if c.strip()]
        val_ids = [c for c in val_ids if c in cached]
        if len(val_ids) != len(d["embeddings"]):
            continue
        all_emb.append(d["embeddings"])
        all_y.extend(d["y_true"].tolist())
        all_cids.extend(val_ids)
    return np.concatenate(all_emb, axis=0), np.array(all_y, int), all_cids


def cv_oof_auc(X, y):
    """Per-case OOF predictions + single AUC + case-bootstrap CI."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=5000, class_weight="balanced")
        s = StandardScaler().fit(X[tr])
        clf.fit(s.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(s.transform(X[te]))[:, 1]
    auc = roc_auc_score(y, oof)
    # Case-bootstrap CI
    rng = np.random.default_rng(SEED)
    boots = []
    n = len(y)
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        try:
            boots.append(roc_auc_score(y[idx], oof[idx]))
        except ValueError:
            pass
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(auc), float(lo), float(hi), oof


def main():
    proto = pd.read_csv(PROTO)
    proto_map = dict(zip(proto["case_id"], proto["protocol"]))

    results = {"settings": {}}
    for name, emb_dir in [("baseline", EMB_BASELINE), ("adv_lambda_0.5", EMB_ADV)]:
        if not emb_dir.exists():
            print(f"WARN: {emb_dir} missing")
            continue
        emb, y_disease, cids = load_emb_with_meta(emb_dir)
        protocols = np.array([proto_map.get(c, "unknown") for c in cids])
        is_contrast = (protocols == "contrast").astype(int)
        nph_mask = (y_disease == 0)
        Xn, yn = emb[nph_mask], is_contrast[nph_mask]
        print(f"\n=== {name} ===")
        print(f"  total val cases: {len(cids)}, nonPH: {int(nph_mask.sum())} "
              f"(contrast={int(yn.sum())}, plain={int((1-yn).sum())})")
        proto_auc, p_lo, p_hi, _ = cv_oof_auc(Xn, yn)
        print(f"  protocol within-nonPH (case-bootstrap CI): AUC={proto_auc:.3f} [{p_lo:.3f}, {p_hi:.3f}]")
        cmask = (is_contrast == 1)
        disease_auc, d_lo, d_hi, _ = cv_oof_auc(emb[cmask], y_disease[cmask])
        print(f"  disease within-contrast: AUC={disease_auc:.3f} [{d_lo:.3f}, {d_hi:.3f}]")
        results["settings"][name] = {
            "protocol_within_nonph": {"auc": proto_auc, "ci95": [p_lo, p_hi]},
            "disease_within_contrast": {"auc": disease_auc, "ci95": [d_lo, d_hi]},
            "n_nonph": int(nph_mask.sum()),
            "n_contrast": int(cmask.sum()),
        }

    if "baseline" in results["settings"] and "adv_lambda_0.5" in results["settings"]:
        b = results["settings"]["baseline"]
        a = results["settings"]["adv_lambda_0.5"]
        results["delta"] = {
            "protocol_auc": a["protocol_within_nonph"]["auc"] - b["protocol_within_nonph"]["auc"],
            "disease_auc": a["disease_within_contrast"]["auc"] - b["disease_within_contrast"]["auc"],
        }
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        "# R10 — Adversarial-debiased embedding protocol test",
        "",
        "Compares protocol decodability WITHIN nonPH on (a) baseline arm_a embeddings",
        "(from R9 full-cohort training) vs (b) adversarially-debiased arm_a",
        "embeddings (this round, GRL λ=0.5 on is_contrast).",
        "",
        "Case-level OOF predictions + bootstrap CI over cases (addressing Round-9",
        "reviewer's stats-hygiene flag on fold-level bootstrap).",
        "",
        "| Setting | n nonPH | Protocol AUC (case-boot CI) | Disease AUC (contrast) (case-boot CI) |",
        "|---|---|---|---|",
    ]
    for name, s in results["settings"].items():
        pn = s["protocol_within_nonph"]
        dw = s["disease_within_contrast"]
        lines.append(
            f"| **{name}** | {s['n_nonph']} | "
            f"{pn['auc']:.3f} [{pn['ci95'][0]:.3f}, {pn['ci95'][1]:.3f}] | "
            f"{dw['auc']:.3f} [{dw['ci95'][0]:.3f}, {dw['ci95'][1]:.3f}] |"
        )
    if "delta" in results:
        d = results["delta"]
        target_met = results["settings"]["adv_lambda_0.5"]["protocol_within_nonph"]["auc"] <= 0.60
        lines += [
            "",
            f"**Δ (adv − baseline)**: protocol AUC = **{d['protocol_auc']:+.3f}**, "
            f"disease AUC = **{d['disease_auc']:+.3f}**",
            "",
            f"**Reviewer target**: within-nonPH protocol AUC ≤ 0.60 "
            f"({'✅ MET' if target_met else '❌ NOT MET'})",
            "",
            "## Reading",
            "",
        ]
        if target_met:
            lines += [
                "- Adversarial debiasing successfully reduced protocol leakage below the",
                f"  reviewer's 0.60 ceiling with disease-AUC impact {d['disease_auc']:+.3f}.",
                "- This is the principal Round-8-→-8/10 lever.",
            ]
        else:
            lines += [
                f"- Adversarial debiasing at λ=0.5 moved protocol AUC by {d['protocol_auc']:+.3f}",
                "  but did not yet reach the 0.60 target. Next: sweep λ ∈ {1.0, 2.0, 5.0} and",
                "  potentially warm-up schedule; re-check disease AUC preservation.",
            ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
