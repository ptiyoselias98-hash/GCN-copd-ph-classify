"""A0_cache_qc — audit existing graph cache (cache_tri_v2_unified301 on remote, or
local extracted morph_unified301.csv) for empty/single-node graphs, n_nodes/n_edges
distribution, structure availability.

Since unified-301 cache lives on remote, audit via local morph_unified301 + per_structure_morphometrics
which carry n_nodes/n_edges per structure.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent
MORPH_UNIFIED = ROOT / "outputs" / "r20" / "morph_unified301.csv"
MORPH_LEGACY = ROOT / "outputs" / "r17" / "per_structure_morphometrics.csv"
TDA_LEGACY = ROOT / "outputs" / "r17" / "per_structure_tda.csv"
LUNG_LEGACY = ROOT / "outputs" / "lung_features_v2.csv"
OUT = ROOT / "outputs" / "supplementary" / "A0_data_audit"
OUT.mkdir(parents=True, exist_ok=True)


def per_structure_audit(df, struct):
    n_nodes_col = f"{struct}_n_nodes"
    n_edges_col = f"{struct}_n_edges"
    n_branches_col = f"{struct}_n_branches"
    out = {"structure": struct, "n_total_cases": int(len(df))}
    if n_nodes_col in df.columns:
        nodes = df[n_nodes_col].dropna()
        out.update({
            "n_with_nonzero_nodes": int((nodes > 0).sum()),
            "n_empty_graph": int((nodes == 0).sum()),
            "n_single_node": int((nodes == 1).sum()),
            "n_nodes_p10_p50_p90": [float(np.percentile(nodes, p)) for p in (10, 50, 90)] if len(nodes) else None,
        })
    if n_edges_col in df.columns:
        edges = df[n_edges_col].dropna()
        out["n_edges_p10_p50_p90"] = [float(np.percentile(edges, p)) for p in (10, 50, 90)] if len(edges) else None
    if n_branches_col in df.columns:
        b = df[n_branches_col].dropna()
        out["n_branches_p10_p50_p90"] = [float(np.percentile(b, p)) for p in (10, 50, 90)] if len(b) else None
    return out


def main():
    qc = {"feature_files_found": {}}
    # unified-301 morph (R20.G)
    if MORPH_UNIFIED.exists():
        m = pd.read_csv(MORPH_UNIFIED)
        qc["feature_files_found"]["morph_unified301"] = {
            "n_cases": int(len(m)), "n_columns": int(m.shape[1])}
        qc["per_structure_unified301"] = [per_structure_audit(m, s)
                                            for s in ("artery", "vein", "airway")]
    # legacy morph (R17, HiPaS-style)
    if MORPH_LEGACY.exists():
        m = pd.read_csv(MORPH_LEGACY)
        qc["feature_files_found"]["morph_legacy_R17"] = {
            "n_cases": int(len(m)), "n_columns": int(m.shape[1])}
        qc["per_structure_legacy_R17"] = [per_structure_audit(m, s)
                                           for s in ("artery", "vein", "airway")]
    # TDA
    if TDA_LEGACY.exists():
        t = pd.read_csv(TDA_LEGACY)
        qc["feature_files_found"]["tda_legacy_R17.5"] = {
            "n_cases": int(len(t)), "n_columns": int(t.shape[1])}
    # lung
    if LUNG_LEGACY.exists():
        l = pd.read_csv(LUNG_LEGACY)
        qc["feature_files_found"]["lung_features_v2"] = {
            "n_cases": int(len(l)), "n_columns": int(l.shape[1])}

    (OUT / "cache_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")

    # Histograms
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    if MORPH_UNIFIED.exists():
        m = pd.read_csv(MORPH_UNIFIED)
        for col, struct in enumerate(("artery", "vein", "airway")):
            ax = axes[0, col]
            n_col = f"{struct}_n_nodes"
            if n_col in m.columns:
                ax.hist(m[n_col].dropna(), bins=30, color="#3b82f6", alpha=0.7,
                        edgecolor="black")
                ax.set_xlabel(f"{struct} n_nodes")
                ax.set_ylabel("frequency")
                ax.set_title(f"unified-301 {struct} n_nodes (n={len(m)})")
                ax.grid(alpha=0.3)
            ax = axes[1, col]
            e_col = f"{struct}_n_edges"
            if e_col in m.columns:
                ax.hist(m[e_col].dropna(), bins=30, color="#ef4444", alpha=0.7,
                        edgecolor="black")
                ax.set_xlabel(f"{struct} n_edges")
                ax.set_ylabel("frequency")
                ax.set_title(f"unified-301 {struct} n_edges")
                ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "cache_qc_histograms.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(json.dumps(qc, indent=2))
    print(f"saved {OUT}/cache_qc.{{json,histograms.png}}")


if __name__ == "__main__":
    main()
