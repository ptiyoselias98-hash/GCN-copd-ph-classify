"""R14.B — Multi-structure (artery + vein + airway) phenotype clustering.

Final-goal scientific question (user 2026-04-25):
  在COPD向COPD-PH转归过程中，肺血管影像表型如何演化？
  肺实质与气道表型在其中起到怎样的辅助作用？

This script approaches the question by:
  1. Loading per-case 47-feature graph aggregates from cache_v2_tri_flat
     (artery, vein, airway each → ~15 features × 3 structures + 2 globals).
  2. Joining R14.A lung parenchyma features (LAA-950, mean HU, etc.) when
     available — these ARE the lung-parenchyma auxiliary phenotypes.
  3. UMAP + KMeans + GaussianMixture clustering on the joined feature
     space — identifies "transition" clusters that bridge nonPH and PH.
  4. Per-cluster: PH/nonPH composition, mean phenotype values, top-3
     features by abs Z-score → the "typical structural pattern" per cluster.

Skip seg-failure cases. Within-contrast restriction available via flag
to remove protocol confound.

Outputs: outputs/r14/multistruct_clusters_{full,contrast_only}.{json,md,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent.parent
OUT = ROOT / "outputs" / "r14"
OUT.mkdir(parents=True, exist_ok=True)
TRI_DIR = ROOT / "outputs" / "r5"
GRAPH_FEATS = ROOT / "outputs" / "r5" / "cache_v2_tri_flat_aggregates.csv"
LUNG_FEATS_LEGACY = ROOT / "outputs" / "lung_features_v2.csv"
LUNG_FEATS_NEW = OUT / "lung_parenchyma_features.csv"
LABELS = ROOT / "data" / "labels_expanded_282.csv"
PROTO = ROOT / "data" / "case_protocol.csv"
SEG_FAILS = ROOT / "outputs" / "r13" / "seg_failures_real.json"


def load_graph_features() -> pd.DataFrame | None:
    """Try loading the existing 47-feature graph aggregate CSV."""
    candidates = [
        ROOT / "outputs" / "r5" / "cache_v2_tri_flat_aggregates.csv",
        ROOT / "outputs" / "r5" / "v2_aggregates.csv",
        ROOT / "outputs" / "_v2_aggregates.csv",
    ]
    for p in candidates:
        if p.exists():
            print(f"[graph_feats] loading {p}")
            return pd.read_csv(p)
    print(f"[graph_feats] no aggregate CSV found; tried {[str(p) for p in candidates]}")
    return None


def cluster_and_summarize(X: np.ndarray, ids: list[str], df_meta: pd.DataFrame,
                           feat_names: list[str], k: int = 4, suffix: str = "full"):
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    # Try UMAP, fall back to PCA if not installed
    try:
        import umap
        reducer = umap.UMAP(n_components=2, n_neighbors=min(15, len(X) - 1),
                            random_state=42)
        Z = reducer.fit_transform(Xs)
        proj_name = "UMAP"
    except Exception as exc:
        from sklearn.decomposition import PCA
        Z = PCA(n_components=2, random_state=42).fit_transform(Xs)
        proj_name = f"PCA (UMAP unavailable: {str(exc)[:60]})"

    # KMeans + GMM
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
    gmm = GaussianMixture(n_components=k, random_state=42, covariance_type="diag").fit(Xs)
    df_meta = df_meta.copy()
    df_meta["cluster_km"] = km.labels_
    df_meta["cluster_gmm"] = gmm.predict(Xs)

    # Per-cluster composition + top features (KMeans)
    cluster_summary = []
    for c in range(k):
        mask = df_meta["cluster_km"] == c
        sub = df_meta.loc[mask]
        if len(sub) == 0:
            continue
        nph = (sub["label"] == 0).sum(); ph = (sub["label"] == 1).sum()
        ph_pct = 100.0 * ph / max(nph + ph, 1)
        # Top features by mean Z-score absolute value within cluster
        z_means = Xs[mask].mean(axis=0)
        order = np.argsort(np.abs(z_means))[::-1][:5]
        top_feats = [(feat_names[i], float(z_means[i])) for i in order]
        cluster_summary.append({
            "cluster": int(c),
            "n": int(len(sub)),
            "n_nonph": int(nph),
            "n_ph": int(ph),
            "ph_pct": float(ph_pct),
            "top_features": top_feats,
        })

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    label_arr = df_meta["label"].values
    proto_arr = df_meta["protocol"].fillna("").values

    # Left: colour by label
    for lbl, marker, color in [(0, "o", "#3b82f6"), (1, "x", "#ef4444")]:
        m = label_arr == lbl
        ax[0].scatter(Z[m, 0], Z[m, 1], c=color, marker=marker,
                      label=f"label={lbl}", alpha=0.6, s=30)
    # Annotate cluster centres
    for c in range(k):
        m = df_meta["cluster_km"].values == c
        if m.any():
            cx, cy = Z[m, 0].mean(), Z[m, 1].mean()
            ax[0].annotate(f"C{c}", (cx, cy), fontsize=14, fontweight="bold",
                            ha="center", va="center",
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
    ax[0].set_title(f"{proj_name} — by label (PH=red, nonPH=blue)")
    ax[0].legend()

    # Right: colour by KMeans cluster
    cmap = plt.get_cmap("tab10")
    for c in range(k):
        m = df_meta["cluster_km"].values == c
        ax[1].scatter(Z[m, 0], Z[m, 1], c=[cmap(c)], label=f"C{c} (n={m.sum()})",
                      alpha=0.6, s=30)
    ax[1].set_title(f"KMeans clusters (k={k})")
    ax[1].legend()
    plt.tight_layout()
    fig_path = OUT / f"multistruct_clusters_{suffix}.png"
    plt.savefig(fig_path, dpi=140); plt.close(fig)

    return {
        "n_cases": len(df_meta),
        "k": k,
        "projection": proj_name,
        "kmeans_summary": cluster_summary,
        "fig": str(fig_path),
    }, df_meta


def main():
    graph_df = load_graph_features()
    if LUNG_FEATS_NEW.exists():
        lung_df = pd.read_csv(LUNG_FEATS_NEW)
        lung_src = "R14.A new"
    elif LUNG_FEATS_LEGACY.exists():
        lung_df = pd.read_csv(LUNG_FEATS_LEGACY)
        lung_src = "lung_features_v2.csv (existing)"
    else:
        lung_df = None
        lung_src = "(none)"
    print(f"[lung_feats] using {lung_src}")
    labels = pd.read_csv(LABELS)
    proto = pd.read_csv(PROTO)
    df = labels.merge(proto[["case_id", "protocol"]], on="case_id", how="left")
    fails: set[str] = set()
    if SEG_FAILS.exists():
        sf = json.loads(SEG_FAILS.read_text(encoding="utf-8"))
        for r in sf.get("real_fails", []) + sf.get("lung_anomaly", []):
            fails.add(r["case_id"])

    if graph_df is None and lung_df is None:
        print("ERROR: neither graph aggregates nor lung features available")
        return

    feat_dfs = []
    feat_names = []
    if graph_df is not None:
        gid = "case_id" if "case_id" in graph_df.columns else "patient_id"
        graph_df = graph_df.rename(columns={gid: "case_id"})
        gcols = [c for c in graph_df.columns if c != "case_id"
                 and pd.api.types.is_numeric_dtype(graph_df[c])]
        feat_names += [f"g_{c}" for c in gcols]
        feat_dfs.append(graph_df[["case_id"] + gcols].rename(
            columns={c: f"g_{c}" for c in gcols}))
    if lung_df is not None:
        # Try multiple feature column conventions (R14 new vs lung_features_v2)
        lcols_candidates = [
            "LAA_950_pct", "LAA_910_pct", "HAA_700_pct",
            "mean_HU", "sd_HU", "skew_HU", "kurt_HU",
            "total_lung_vol_mL", "left_right_asymmetry", "lower_upper_ratio",
            # legacy lung_features_v2 columns:
            "paren_LAA_950_frac", "paren_LAA_910_frac", "paren_LAA_856_frac",
            "paren_mean_HU", "paren_std_HU", "paren_HU_p25", "paren_HU_p75", "paren_HU_p95",
            "lung_vol_mL", "vessel_airway_over_lung",
            "apical_LAA_950_frac", "basal_LAA_950_frac",
            "apical_basal_LAA950_gradient",
            "artery_vol_mL", "vein_vol_mL", "airway_vol_mL",
        ]
        lcols = [c for c in lcols_candidates if c in lung_df.columns]
        feat_names += [f"l_{c}" for c in lcols]
        feat_dfs.append(lung_df[["case_id"] + lcols].rename(
            columns={c: f"l_{c}" for c in lcols}))

    # Inner join feature dataframes by case_id
    full = feat_dfs[0]
    for fd in feat_dfs[1:]:
        full = full.merge(fd, on="case_id", how="inner")
    full = full.merge(df[["case_id", "label", "protocol"]], on="case_id", how="left")
    full = full[~full["case_id"].isin(fails)]
    full = full.dropna(subset=feat_names)

    print(f"[merged] {len(full)} cases with full feature vector ({len(feat_names)} features)")

    # Run on full cohort + within-contrast
    summaries = {}
    X = full[feat_names].values
    summaries["full"], _ = cluster_and_summarize(X, full["case_id"].tolist(),
                                                  full[["case_id", "label", "protocol"]],
                                                  feat_names, k=4, suffix="full")

    contrast = full[full["protocol"].str.lower() == "contrast"].reset_index(drop=True)
    if len(contrast) >= 30:
        Xc = contrast[feat_names].values
        summaries["contrast_only"], _ = cluster_and_summarize(
            Xc, contrast["case_id"].tolist(),
            contrast[["case_id", "label", "protocol"]],
            feat_names, k=3, suffix="contrast_only")

    (OUT / "multistruct_clusters.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# R14.B — Multi-structure phenotype clusters",
          "",
          "Joins per-case graph aggregates (artery + vein + airway from",
          "`cache_v2_tri_flat`) with lung parenchyma features (R14.A) to",
          "build a single multi-modal feature vector per case, then UMAP +",
          "KMeans clusters them. Each cluster represents a 'typical structural",
          "pattern' on the COPD ↔ COPD-PH spectrum.",
          ""]
    for tag, s in summaries.items():
        md += [f"## Cohort: {tag} (n={s['n_cases']})",
               "",
               f"Projection: {s['projection']} | KMeans k={s['k']} | Fig: `{s['fig']}`",
               "",
               "| cluster | n | nonPH | PH | PH% | top features (z-score) |",
               "|---|---|---|---|---|---|"]
        for c in s["kmeans_summary"]:
            top = "; ".join(f"{n}={z:+.2f}" for n, z in c["top_features"])
            md.append(f"| C{c['cluster']} | {c['n']} | {c['n_nonph']} | {c['n_ph']} | "
                      f"{c['ph_pct']:.1f}% | {top} |")
        md.append("")

    (OUT / "multistruct_clusters.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Saved {OUT}/multistruct_clusters.md")


if __name__ == "__main__":
    main()
