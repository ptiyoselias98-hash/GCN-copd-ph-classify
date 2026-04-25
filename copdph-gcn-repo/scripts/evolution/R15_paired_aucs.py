"""R15.A — Paired AUC-difference CIs (R14 reviewer must-fix).

Two paired tests on identical n=68 corrected within-contrast / within-nonPH cases:

  1. Lung-only vs Graph-only disease classifier (within-contrast n=184):
     paired bootstrap on case-level OOF probability differences.
     Reports Δ_AUC and 95% CI on Δ.

  2. CORAL λ=1 vs corrected-GRL λ=10 within-nonPH protocol classifier:
     paired bootstrap on per-case OOF prob differences (n=68 corrected).
     Reports Δ_AUC and 95% CI on Δ.

Outputs: outputs/r15/paired_aucs.{json,md}
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
OUT = ROOT / "outputs" / "r15"
OUT.mkdir(parents=True, exist_ok=True)

GRAPH = ROOT / "outputs" / "r5" / "cache_v2_tri_flat_aggregates.csv"
LUNG = ROOT / "outputs" / "lung_features_v2.csv"
LABELS = ROOT / "data" / "labels_expanded_282.csv"
PROTO = ROOT / "data" / "case_protocol.csv"
SEG_FAILS = ROOT / "outputs" / "r13" / "seg_failures_real.json"
CORAL_EMB_ROOT = ROOT / "outputs" / "r13" / "coral_embeddings"
GRL_EMB_ROOT = ROOT / "outputs" / "r11" / "embeddings"
SPLITS = ROOT / "data" / "splits_expanded_282"
CACHE_LIST = ROOT / "outputs" / "r5" / "cache_v2_tri_flat_list.txt"


def load_seg_fails():
    if not SEG_FAILS.exists():
        return set()
    sf = json.loads(SEG_FAILS.read_text(encoding="utf-8"))
    return {r["case_id"] for r in sf.get("real_fails", []) + sf.get("lung_anomaly", [])}


def oof_lr_proba(X, y, seed=42):
    if len(np.unique(y)) < 2:
        return None
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0)
        clf.fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return oof


def paired_boot_delta_ci(y, p_a, p_b, n_boot=5000, seed=42):
    """Paired bootstrap of (AUC(A) - AUC(B)) over cases (resampled together)."""
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    auc_a = roc_auc_score(y, p_a)
    auc_b = roc_auc_score(y, p_b)
    deltas = []
    for _ in range(n_boot):
        bp = rng.choice(pos, size=len(pos), replace=True)
        bn = rng.choice(neg, size=len(neg), replace=True)
        idx = np.concatenate([bp, bn])
        try:
            d = roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx])
            deltas.append(d)
        except ValueError:
            continue
    arr = np.array(deltas)
    return {
        "auc_a": float(auc_a), "auc_b": float(auc_b),
        "delta_mean": float(arr.mean()),
        "delta_ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
        "p_two_sided_approx": float(2 * min((arr <= 0).mean(), (arr >= 0).mean())),
        "n_bootstraps_kept": int(arr.size),
    }


def lung_vs_graph_paired():
    g_df = pd.read_csv(GRAPH); l_df = pd.read_csv(LUNG)
    labels = pd.read_csv(LABELS); proto = pd.read_csv(PROTO)
    df = labels.merge(proto[["case_id", "protocol"]], on="case_id", how="left")
    fails = load_seg_fails()

    g_cols = [c for c in g_df.columns if c != "case_id" and pd.api.types.is_numeric_dtype(g_df[c])]
    l_cols = [c for c in l_df.columns if c != "case_id" and pd.api.types.is_numeric_dtype(l_df[c])]
    g_df = g_df[["case_id"] + g_cols].rename(columns={c: f"g_{c}" for c in g_cols})
    l_df = l_df[["case_id"] + l_cols].rename(columns={c: f"l_{c}" for c in l_cols})
    g_cols_p = [f"g_{c}" for c in g_cols]; l_cols_p = [f"l_{c}" for c in l_cols]

    full = df.merge(g_df, on="case_id", how="left").merge(l_df, on="case_id", how="left")
    contrast = full[full["protocol"].str.lower() == "contrast"].copy()
    contrast = contrast[~contrast["case_id"].isin(fails)]
    contrast = contrast.dropna(subset=g_cols_p + l_cols_p).reset_index(drop=True)
    y = contrast["label"].values.astype(int)
    print(f"[paired lung-vs-graph] n={len(contrast)} (PH={int((y==1).sum())} nonPH={int((y==0).sum())})")

    p_g = oof_lr_proba(contrast[g_cols_p].values, y, seed=42)
    p_l = oof_lr_proba(contrast[l_cols_p].values, y, seed=42)
    p_b = oof_lr_proba(contrast[g_cols_p + l_cols_p].values, y, seed=42)

    out = {"n": int(len(contrast)),
           "lung_minus_graph": paired_boot_delta_ci(y, p_l, p_g),
           "both_minus_graph": paired_boot_delta_ci(y, p_b, p_g),
           "both_minus_lung": paired_boot_delta_ci(y, p_b, p_l)}
    return out


def load_emb_with_ids(emb_dir: Path):
    """Same loader used by R13_coral_probe — returns (X, y, ids) over 5 folds."""
    cached = set(c.strip() for c in CACHE_LIST.read_text(encoding="utf-8").splitlines() if c.strip())
    all_e, all_y, all_c = [], [], []
    for k in range(1, 6):
        f = emb_dir / f"emb_gcn_only_rep1_fold{k}.npz"
        if not f.exists():
            return None
        d = np.load(f)
        val_ids = [c.strip() for c in (SPLITS / f"fold_{k}" / "val.txt").read_text().splitlines() if c.strip()]
        val_ids = [c for c in val_ids if c in cached]
        if len(val_ids) != len(d["embeddings"]):
            return None
        all_e.append(d["embeddings"])
        all_y.extend(d["y_true"].tolist())
        all_c.extend(val_ids)
    return np.concatenate(all_e), np.array(all_y, int), all_c


def coral_vs_grl_paired(seed=42, lam_coral=1.0, lam_grl=10.0):
    proto = pd.read_csv(PROTO)
    proto["is_contrast"] = (proto["protocol"].astype(str).str.lower() == "contrast").astype(int)
    proto_lookup = dict(zip(proto["case_id"], proto["is_contrast"]))
    label_df = pd.read_csv(LABELS)
    lbl_lookup = dict(zip(label_df["case_id"], label_df["label"]))
    fails = load_seg_fails()

    coral_dir = CORAL_EMB_ROOT / f"coral_l{lam_coral}_s{seed}"
    grl_dir = GRL_EMB_ROOT / f"l{lam_grl}_s{seed}"
    if not coral_dir.exists() or not grl_dir.exists():
        return {"error": f"missing dirs coral={coral_dir.exists()} grl={grl_dir.exists()}"}
    cor = load_emb_with_ids(coral_dir)
    grl = load_emb_with_ids(grl_dir)
    if cor is None or grl is None:
        return {"error": "load failed"}
    X_c, y_c, ids_c = cor
    X_g, y_g, ids_g = grl
    common = sorted(set(ids_c) & set(ids_g))
    idx_c = {c: i for i, c in enumerate(ids_c)}
    idx_g = {c: i for i, c in enumerate(ids_g)}

    keep = [c for c in common if lbl_lookup.get(c, 1) == 0
            and proto_lookup.get(c, -1) >= 0
            and c not in fails]
    if len(keep) < 20:
        return {"error": f"too few common nonPH cases ({len(keep)})"}
    Xc = np.stack([X_c[idx_c[c]] for c in keep])
    Xg = np.stack([X_g[idx_g[c]] for c in keep])
    yp = np.array([proto_lookup[c] for c in keep], int)

    p_c = oof_lr_proba(Xc, yp, seed=seed)
    p_g = oof_lr_proba(Xg, yp, seed=seed)
    if p_c is None or p_g is None:
        return {"error": "oof failed"}
    return {
        "n": int(len(keep)),
        "lam_coral": lam_coral,
        "lam_grl": lam_grl,
        "seed": seed,
        "coral_minus_grl": paired_boot_delta_ci(yp, p_c, p_g),
    }


def main():
    out = {"lung_vs_graph": lung_vs_graph_paired()}
    out["coral_vs_grl_per_seed"] = {}
    for seed in (42, 1042, 2042):
        out["coral_vs_grl_per_seed"][f"seed_{seed}"] = coral_vs_grl_paired(seed=seed)
    (OUT / "paired_aucs.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = ["# R15.A — Paired AUC-difference CIs",
          "",
          "Addresses R14 reviewer flag: 'lung_only > graph_only CIs overlap; reversal needs paired AUC-diff CI'",
          "and 'CORAL λ=1 vs GRL not yet paired on identical cases'.",
          "",
          "## Lung vs Graph (within-contrast disease classifier)",
          "",
          f"n={out['lung_vs_graph']['n']}. Paired bootstrap (5000 iters) of",
          "AUC differences on case-level OOF predictions (same case set, same y).",
          "",
          "| comparison | AUC_a | AUC_b | Δ (a−b) | 95% CI | approx p |",
          "|---|---|---|---|---|---|"]
    for cmp_key, label in [("lung_minus_graph", "lung − graph"),
                            ("both_minus_graph", "(lung+graph) − graph"),
                            ("both_minus_lung", "(lung+graph) − lung")]:
        r = out["lung_vs_graph"][cmp_key]
        md.append(f"| {label} | {r['auc_a']:.3f} | {r['auc_b']:.3f} | "
                  f"{r['delta_mean']:+.3f} | "
                  f"[{r['delta_ci95'][0]:+.3f}, {r['delta_ci95'][1]:+.3f}] | "
                  f"{r['p_two_sided_approx']:.3g} |")

    md += ["",
           "## CORAL λ=1 vs corrected-GRL λ=10 (within-nonPH protocol probe)",
           "",
           "Paired on the intersection of common case_ids in both embedding sets,",
           "restricted to nonPH and excluding seg-failures.",
           "",
           "| seed | n | AUC CORAL | AUC GRL | Δ (CORAL − GRL) | 95% CI | approx p |",
           "|---|---|---|---|---|---|---|"]
    for k, rec in out["coral_vs_grl_per_seed"].items():
        if "error" in rec:
            md.append(f"| {k} | — | — | — | — | — | {rec['error']} |")
            continue
        d = rec["coral_minus_grl"]
        md.append(f"| {rec['seed']} | {rec['n']} | {d['auc_a']:.3f} | {d['auc_b']:.3f} | "
                  f"{d['delta_mean']:+.3f} | "
                  f"[{d['delta_ci95'][0]:+.3f}, {d['delta_ci95'][1]:+.3f}] | "
                  f"{d['p_two_sided_approx']:.3g} |")

    md += ["",
           "## Interpretation",
           "",
           "- For lung vs graph: if `lung − graph` 95% CI excludes 0, the reversal",
           "  is statistically supported; otherwise the marginal CIs were misleading.",
           "- For CORAL vs GRL: if `CORAL − GRL` is significantly negative (lower",
           "  protocol AUC = better deconfounder), CORAL is confirmed as a real",
           "  improvement on the same cases.",
           ""]

    (OUT / "paired_aucs.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved {OUT}/paired_aucs.json + paired_aucs.md")


if __name__ == "__main__":
    main()
