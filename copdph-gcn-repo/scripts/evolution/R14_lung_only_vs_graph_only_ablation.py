"""R14.D — Lung-only vs graph-only vs combined disease classifier ablation.

Quantifies the AUXILIARY role of lung parenchyma in disease classification:
how much of the disease AUC is driven by vascular-graph features alone,
how much by lung-parenchyma features alone, how much by their union?

Endpoint: within-contrast disease classifier (n=189). Removes protocol confound.

Outputs: outputs/r14/ablation_lung_vs_graph.{json,md}
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
OUT = ROOT / "outputs" / "r14"
OUT.mkdir(parents=True, exist_ok=True)

GRAPH = ROOT / "outputs" / "r5" / "cache_v2_tri_flat_aggregates.csv"
LUNG = ROOT / "outputs" / "lung_features_v2.csv"
LABELS = ROOT / "data" / "labels_expanded_282.csv"
PROTO = ROOT / "data" / "case_protocol.csv"
SEG_FAILS = ROOT / "outputs" / "r13" / "seg_failures_real.json"

GRAPH_PREFIX = "g_"
LUNG_PREFIX = "l_"


def boot_auc_ci(y, p, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    if not len(pos) or not len(neg):
        return [float("nan")] * 2
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
    return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]


def oof_lr(X, y, seed=42):
    if len(np.unique(y)) < 2: return None, None
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0)
        clf.fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return float(roc_auc_score(y, oof)), oof


def main():
    g_df = pd.read_csv(GRAPH) if GRAPH.exists() else None
    l_df = pd.read_csv(LUNG) if LUNG.exists() else None
    if g_df is None and l_df is None:
        raise SystemExit("missing both graph + lung CSVs")
    labels = pd.read_csv(LABELS)
    proto = pd.read_csv(PROTO)
    df = labels.merge(proto[["case_id", "protocol"]], on="case_id", how="left")

    fails: set[str] = set()
    if SEG_FAILS.exists():
        sf = json.loads(SEG_FAILS.read_text(encoding="utf-8"))
        for r in sf.get("real_fails", []) + sf.get("lung_anomaly", []):
            fails.add(r["case_id"])

    # Build feature dataframes
    if g_df is not None:
        g_cols = [c for c in g_df.columns if c != "case_id" and pd.api.types.is_numeric_dtype(g_df[c])]
        g_df = g_df[["case_id"] + g_cols].rename(columns={c: GRAPH_PREFIX + c for c in g_cols})
        g_cols_pref = [GRAPH_PREFIX + c for c in g_cols]
    else:
        g_cols_pref = []
    if l_df is not None:
        l_cols = [c for c in l_df.columns if c != "case_id" and pd.api.types.is_numeric_dtype(l_df[c])]
        l_df = l_df[["case_id"] + l_cols].rename(columns={c: LUNG_PREFIX + c for c in l_cols})
        l_cols_pref = [LUNG_PREFIX + c for c in l_cols]
    else:
        l_cols_pref = []

    # Merge into df
    full = df.copy()
    if g_df is not None: full = full.merge(g_df, on="case_id", how="left")
    if l_df is not None: full = full.merge(l_df, on="case_id", how="left")

    # Subset: within-contrast, exclude seg-failures, complete features only
    contrast = full[full["protocol"].str.lower() == "contrast"].copy()
    contrast = contrast[~contrast["case_id"].isin(fails)]
    contrast = contrast.dropna(subset=g_cols_pref + l_cols_pref)
    print(f"Contrast cohort with full features: {len(contrast)} cases "
          f"(PH={int((contrast['label']==1).sum())}, nonPH={int((contrast['label']==0).sum())})")

    y = contrast["label"].values.astype(int)
    if len(np.unique(y)) < 2:
        raise SystemExit("Single class only — abort")

    out: dict = {"n_total": int(len(contrast)),
                 "n_ph": int((y == 1).sum()),
                 "n_nonph": int((y == 0).sum()),
                 "n_excluded_seg_fails": len(fails)}

    feature_sets = {
        "graph_only": g_cols_pref,
        "lung_only": l_cols_pref,
        "graph+lung": g_cols_pref + l_cols_pref,
    }
    for tag, cols in feature_sets.items():
        if not cols:
            continue
        X = contrast[cols].values
        auc, oof = oof_lr(X, y, seed=42)
        if auc is None: continue
        ci = boot_auc_ci(y, oof, n_boot=2000)
        out[tag] = {"n_features": len(cols), "auc": auc, "ci95": ci}
        print(f"  {tag} (n_feats={len(cols)}): AUC={auc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")

    # Per-structure ablation within graph (split graph features by name prefix)
    if g_cols_pref:
        per_struct = {}
        for struct in ("artery", "vein", "airway", "x0", "x1", "x2", "e0", "e1"):
            sub = [c for c in g_cols_pref if struct in c.lower()]
            if not sub: continue
            X = contrast[sub].values
            auc, oof = oof_lr(X, y, seed=42)
            if auc is None: continue
            per_struct[struct] = {"n_features": len(sub), "auc": auc,
                                   "ci95": boot_auc_ci(y, oof, n_boot=1500)}
            print(f"  graph_{struct} (n_feats={len(sub)}): AUC={auc:.3f}")
        out["graph_per_substring"] = per_struct

    (OUT / "ablation_lung_vs_graph.json").write_text(json.dumps(out, indent=2),
                                                        encoding="utf-8")

    md = ["# R14.D — Lung-only vs graph-only disease classifier ablation",
          "",
          "Within-contrast cohort (no protocol confound). 5-fold OOF LR, "
          "case-level bootstrap CI.",
          "",
          f"**n_total**: {out['n_total']} (PH={out['n_ph']}, nonPH={out['n_nonph']}). "
          f"Excluded {out['n_excluded_seg_fails']} seg-failure cases.",
          "",
          "## Headline ablation",
          "",
          "| feature set | n_feats | AUC [95% CI] |",
          "|---|---|---|"]
    for tag in ("graph_only", "lung_only", "graph+lung"):
        if tag not in out: continue
        r = out[tag]
        md.append(f"| {tag} | {r['n_features']} | {r['auc']:.3f} "
                  f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] |")

    if "graph_per_substring" in out:
        md += ["",
               "## Within-graph substructure ablation",
               "",
               "Restricts the graph feature set by substring match in feature name.",
               "",
               "| substring | n_feats | AUC [95% CI] |",
               "|---|---|---|"]
        for k, r in out["graph_per_substring"].items():
            md.append(f"| {k} | {r['n_features']} | {r['auc']:.3f} "
                      f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] |")

    md += ["",
           "## Interpretation",
           "",
           "- If graph_only AUC ≈ graph+lung AUC: lung-parenchyma is largely redundant",
           "  with vascular graph features for disease classification (graph subsumes lung).",
           "- If lung_only AUC > graph_only AUC: lung-parenchyma is the primary disease",
           "  signal carrier (lung dominates); the graph adds little.",
           "- If lung_only ≈ graph_only and graph+lung > both: complementary information.",
           "- Within-graph substring ablation shows which structure (artery/vein/airway)",
           "  carries the most disease-discriminative signal.",
           ""]

    (OUT / "ablation_lung_vs_graph.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Saved {OUT}/ablation_lung_vs_graph.md")


if __name__ == "__main__":
    main()
