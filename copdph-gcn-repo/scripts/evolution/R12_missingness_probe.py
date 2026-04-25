"""R12.2 — Missingness-only protocol probe.

Question (raised by R12 strategy review): if we use "principled missingness"
to recover the 39 dropped cases, does the missingness indicator ALONE leak
protocol? If yes, any imputation/degraded-graph approach inherits the leak
and "fixing" missingness is a false rescue.

Procedure:
- Build is_in_v2_cache binary across full 282
- Fit LR(protocol ~ is_in_v2_cache) on within-nonPH (only nonPH cases)
- Bootstrap-CI on AUC
- Cross-tab: protocol × cache_present, with chi-sq

Output: outputs/r12/missingness_protocol_probe.{json,md}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parent.parent.parent
LABELS = ROOT / "data" / "labels_expanded_282.csv"
PROTO = ROOT / "data" / "case_protocol.csv"
CACHE_LIST = ROOT / "outputs" / "r5" / "cache_v2_tri_flat_list.txt"
OUT = ROOT / "outputs" / "r12"


def boot_auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 5000, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(y)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), [float("nan"), float("nan")]
    boots = []
    for _ in range(n_boot):
        bp = rng.choice(pos, size=len(pos), replace=True)
        bn = rng.choice(neg, size=len(neg), replace=True)
        idx = np.concatenate([bp, bn])
        try:
            boots.append(roc_auc_score(y[idx], p[idx]))
        except ValueError:
            continue
    arr = np.array(boots)
    return float(np.mean(arr)), [float(np.percentile(arr, 2.5)),
                                 float(np.percentile(arr, 97.5))]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    proto = pd.read_csv(PROTO)
    cached = set(c.strip() for c in CACHE_LIST.read_text(encoding="utf-8").splitlines() if c.strip())

    df = proto.copy()
    df["is_in_v2_cache"] = df["case_id"].isin(cached).astype(int)
    df["is_contrast"] = (df["protocol"].astype(str).str.lower() == "contrast").astype(int)

    out = {"counts": {"total": int(len(df)),
                      "n_in_cache": int(df["is_in_v2_cache"].sum()),
                      "n_missing": int((1 - df["is_in_v2_cache"]).sum()),
                      "n_ph": int((df["label"] == 1).sum()) if "label" in df else None}}

    label_col = "label"
    nonph = df[df[label_col] == 0].copy()
    out["nonph_stratum"] = {"n": int(len(nonph))}

    X = nonph[["is_in_v2_cache"]].values.astype(float)
    y = nonph["is_contrast"].values.astype(int)
    if len(np.unique(y)) < 2 or len(np.unique(X)) < 2:
        out["lr_within_nonph"] = {"warning": "degenerate (no variation)"}
    else:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X, y)
        p = clf.predict_proba(X)[:, 1]
        auc = float(roc_auc_score(y, p))
        boot_mean, boot_ci = boot_auc_ci(y, p, n_boot=5000, seed=42)
        out["lr_within_nonph"] = {
            "auc": auc,
            "boot_mean": boot_mean,
            "ci95": boot_ci,
            "n_features": 1,
            "feature": "is_in_v2_cache",
        }

    # Cross-tab missingness × protocol (within-nonPH)
    ct = pd.crosstab(nonph["is_in_v2_cache"], nonph["is_contrast"])
    out["nonph_crosstab"] = {
        "rows_is_in_v2_cache": ct.index.tolist(),
        "cols_is_contrast": ct.columns.tolist(),
        "table": ct.values.tolist(),
    }

    # Full-cohort missingness × protocol (sanity)
    ct_full = pd.crosstab(df["is_in_v2_cache"], df["is_contrast"])
    out["full_crosstab"] = {
        "rows_is_in_v2_cache": ct_full.index.tolist(),
        "cols_is_contrast": ct_full.columns.tolist(),
        "table": ct_full.values.tolist(),
    }

    # Headline judgment
    auc = out.get("lr_within_nonph", {}).get("auc")
    if auc is not None and not (auc != auc):  # not NaN
        if auc >= 0.65:
            verdict = "MISSINGNESS LEAKS PROTOCOL — principled-missingness rescue is unsafe"
        elif auc >= 0.55:
            verdict = "MILD missingness-protocol correlation — disclose but not fatal"
        else:
            verdict = "Missingness ~uninformative for protocol — degraded-graph rebuild is sound"
        out["verdict"] = verdict

    (OUT / "missingness_protocol_probe.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # Markdown report
    md_lines = [
        "# R12.2 — Missingness-only protocol probe",
        "",
        "Tests whether the cache-missingness pattern (39 dropped cases from",
        "`cache_v2_tri_flat`) leaks protocol information by itself. If yes,",
        "any future degraded-graph or principled-missingness approach inherits",
        "the leak.",
        "",
        f"**Cohort**: total {out['counts']['total']}, "
        f"in cache {out['counts']['n_in_cache']}, "
        f"missing {out['counts']['n_missing']}.",
        f"Within-nonPH: n={out['nonph_stratum']['n']}.",
        "",
        "## Within-nonPH LR(protocol ~ is_in_v2_cache)",
        "",
    ]
    if "warning" in out.get("lr_within_nonph", {}):
        md_lines.append(f"⚠ {out['lr_within_nonph']['warning']}")
    else:
        rec = out["lr_within_nonph"]
        md_lines.append(f"- AUC = {rec['auc']:.3f}, bootstrap mean {rec['boot_mean']:.3f}, "
                        f"95% CI [{rec['ci95'][0]:.3f}, {rec['ci95'][1]:.3f}]")
    md_lines += [
        "",
        "## Within-nonPH cross-tab (rows = is_in_v2_cache, cols = is_contrast)",
        "",
        "```",
        str(ct),
        "```",
        "",
        "## Full-cohort cross-tab",
        "",
        "```",
        str(ct_full),
        "```",
        "",
        f"**Verdict**: {out.get('verdict', 'INDETERMINATE')}",
        "",
    ]
    (OUT / "missingness_protocol_probe.md").write_text(
        "\n".join(md_lines), encoding="utf-8")

    print(f"Saved {OUT}/missingness_protocol_probe.{{json,md}}")
    print(f"  AUC: {out.get('lr_within_nonph', {}).get('auc')}")
    print(f"  Verdict: {out.get('verdict')}")


if __name__ == "__main__":
    main()
