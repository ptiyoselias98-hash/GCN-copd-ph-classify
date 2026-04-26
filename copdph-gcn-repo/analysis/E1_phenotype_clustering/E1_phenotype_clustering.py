"""E1_phenotype_clustering — Phase E1: corrected-signature clustering.

Within-contrast n=190 PRIMARY. Use top-50 features by |Cohen's d| from C1 T1
(disease-relevant signatures) + top-30 by |ρ_mPAP| from C1 T2 (severity-relevant)
deduplicated, RobustScaler, then UMAP/PCA + KMeans/GMM k=2..5.

Per codex pass-1 mitigations:
- ARI permutation null vs PH label (1000 perms)
- Cluster names assigned AFTER blinded enrichment inspection (post-hoc, here documented)
- Report all attempted k/algorithm combos, including failures
- Pre-specified winner: max silhouette × stability bootstrap

Output:
  outputs/supplementary/E1_phenotype_clustering/cluster_assignments.csv
  outputs/supplementary/E1_phenotype_clustering/cluster_summary.csv
  outputs/supplementary/E1_phenotype_clustering/cluster_signature_heatmap.png
  outputs/supplementary/E1_phenotype_clustering/umap_clusters.png
  outputs/supplementary/E1_phenotype_clustering/medoid_cases.csv
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).parent.parent.parent
SIG = ROOT / "outputs" / "supplementary" / "B1_graph_signature" / "graph_signatures_patient_level.csv"
T1 = ROOT / "outputs" / "supplementary" / "C1_signature_severity" / "signature_group_stats.csv"
T2 = ROOT / "outputs" / "supplementary" / "C1_signature_severity" / "mpap_correlation_table.csv"
OUT = ROOT / "outputs" / "supplementary" / "E1_phenotype_clustering"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(SIG)
    df = df[df["is_contrast_only_subset"]].reset_index(drop=True)
    assert len(df) == 190, f"E1 cohort drift: expected within-contrast n=190 from A0 lock, got {len(df)}"
    t1 = pd.read_csv(T1).sort_values("abs_d", ascending=False)
    t2 = pd.read_csv(T2).sort_values("abs_rho", ascending=False)
    feat_pool = list(dict.fromkeys(t1.head(50)["feature"].tolist() + t2.head(30)["feature"].tolist()))
    feat_pool = [f for f in feat_pool if f in df.columns]
    print(f"E1 cohort: n={len(df)} (PH={int((df.label==1).sum())}, "
          f"nonPH={int((df.label==0).sum())}); feat panel = top-{len(feat_pool)} dedup")
    X = df[feat_pool].fillna(0).values.astype(float)
    X = RobustScaler().fit_transform(X)
    y = df["label"].values

    # Sweep k=2..5 × KMeans/GMM
    rows = []; assigns = {}
    for k in [2, 3, 4, 5]:
        for algo in ["KMeans", "GMM"]:
            try:
                if algo == "KMeans":
                    mdl = KMeans(n_clusters=k, n_init=20, random_state=42).fit(X)
                    labels = mdl.labels_
                else:
                    mdl = GaussianMixture(n_components=k, random_state=42, n_init=10).fit(X)
                    labels = mdl.predict(X)
                sil = float(silhouette_score(X, labels)) if len(set(labels)) > 1 else float("nan")
                db = float(davies_bouldin_score(X, labels)) if len(set(labels)) > 1 else float("nan")
                ari = float(adjusted_rand_score(y, labels))
                # 1000-perm null for ARI
                rng = np.random.default_rng(42)
                null_aris = []
                for _ in range(1000):
                    null_aris.append(adjusted_rand_score(rng.permutation(y), labels))
                null_aris = np.array(null_aris)
                ari_p = float((np.abs(null_aris) >= np.abs(ari)).mean())
                # Bootstrap stability: resample 80% of cases, re-cluster, compute ARI vs original
                stab_aris = []
                for s in range(50):
                    idx = rng.choice(len(X), size=int(0.8*len(X)), replace=False)
                    if algo == "KMeans":
                        m2 = KMeans(n_clusters=k, n_init=10, random_state=s).fit(X[idx])
                        l2 = m2.labels_
                    else:
                        m2 = GaussianMixture(n_components=k, random_state=s, n_init=5).fit(X[idx])
                        l2 = m2.predict(X[idx])
                    stab_aris.append(adjusted_rand_score(labels[idx], l2))
                rows.append({
                    "k": k, "algo": algo,
                    "silhouette": sil, "davies_bouldin": db,
                    "ari_vs_label": ari, "ari_perm_p": ari_p,
                    "stability_ari_mean": float(np.mean(stab_aris)),
                    "stability_ari_std": float(np.std(stab_aris)),
                })
                assigns[f"{algo}_k{k}"] = labels.tolist()
            except Exception as e:
                rows.append({"k": k, "algo": algo, "error": str(e)[:60]})

    res_df = pd.DataFrame(rows)
    res_df.to_csv(OUT / "cluster_summary.csv", index=False)
    print("\n=== cluster sweep ===")
    print(res_df.to_string())

    # Pre-specified winner: max silhouette × stability_ari_mean
    res_valid = res_df[res_df["silhouette"].notna()].copy()
    res_valid["score"] = res_valid["silhouette"] * res_valid["stability_ari_mean"]
    winner = res_valid.sort_values("score", ascending=False).iloc[0]
    print(f"\nWINNER: k={int(winner['k'])} {winner['algo']} sil={winner['silhouette']:.3f} "
          f"stability={winner['stability_ari_mean']:.3f} ARI-perm-p={winner['ari_perm_p']:.3g}")

    # Save winner assignment + per-cluster summary
    win_key = f"{winner['algo']}_k{int(winner['k'])}"
    df["cluster"] = assigns[win_key]
    df_out = df[["case_id", "label", "measured_mpap", "measured_mpap_flag",
                  "cluster"] + feat_pool[:20]].copy()  # show top-20 features for inspection
    df_out.to_csv(OUT / "cluster_assignments.csv", index=False)

    # Per-cluster stats: PH%, mPAP distribution, top discriminating features
    cluster_summary = []
    for c in sorted(set(df["cluster"])):
        sub = df[df["cluster"] == c]
        mp = sub.loc[sub["measured_mpap_flag"], "measured_mpap"].astype(float).values
        cluster_summary.append({
            "cluster": int(c), "n": int(len(sub)),
            "PH_pct": float((sub["label"] == 1).mean()),
            "n_PH": int((sub["label"] == 1).sum()),
            "n_nonPH": int((sub["label"] == 0).sum()),
            "mpap_mean": float(mp.mean()) if len(mp) else None,
            "mpap_median": float(np.median(mp)) if len(mp) else None,
            "n_mpap_resolved": int(len(mp)),
        })
    cs_df = pd.DataFrame(cluster_summary)
    print("\n=== per-cluster summary ===")
    print(cs_df.to_string())
    cs_df.to_csv(OUT / "cluster_per_cluster_summary.csv", index=False)

    # PCA-2D for visualization
    pca = PCA(n_components=2).fit(X)
    Z = pca.transform(X)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    for c in sorted(set(df["cluster"])):
        m = df["cluster"] == c
        ax.scatter(Z[m, 0], Z[m, 1], label=f"cluster {c} (n={int(m.sum())}, "
                                            f"PH={int(((df.loc[m,'label']==1)).sum())})",
                   s=30, alpha=0.7, edgecolors="black", linewidths=0.3)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"E1 — Phenotype clustering: {win_key} (n=190 within-contrast)\n"
                 f"sil={winner['silhouette']:.3f} stab-ARI={winner['stability_ari_mean']:.3f} "
                 f"ARI-vs-PH={winner['ari_vs_label']:.3f} perm-p={winner['ari_perm_p']:.3g}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Heatmap: cluster × top-15 features (z-score)
    ax = axes[1]
    top15 = feat_pool[:15]
    cluster_means = []
    for c in sorted(set(df["cluster"])):
        m = df["cluster"] == c
        cluster_means.append([df.loc[m, f].mean() for f in top15])
    cluster_means = np.array(cluster_means)
    # Z-score columns
    cm_z = (cluster_means - cluster_means.mean(0)) / (cluster_means.std(0) + 1e-9)
    im = ax.imshow(cm_z, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_yticks(range(len(set(df["cluster"]))))
    ax.set_yticklabels([f"C{c}" for c in sorted(set(df["cluster"]))])
    ax.set_xticks(range(len(top15)))
    ax.set_xticklabels([f[:18] for f in top15], rotation=70, fontsize=7)
    plt.colorbar(im, ax=ax, label="z-score (across clusters)")
    ax.set_title(f"Cluster mean z-scores on top-15 D1+C1 features")
    plt.tight_layout()
    plt.savefig(OUT / "umap_clusters.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save signature heatmap separately (top-30 features × clusters)
    fig, ax = plt.subplots(figsize=(14, max(4, len(set(df["cluster"])) * 0.6)))
    top30 = feat_pool[:30]
    cluster_means_30 = []
    for c in sorted(set(df["cluster"])):
        m = df["cluster"] == c
        cluster_means_30.append([df.loc[m, f].mean() for f in top30])
    cm_z_30 = (np.array(cluster_means_30) - np.array(cluster_means_30).mean(0)) / (np.array(cluster_means_30).std(0) + 1e-9)
    im = ax.imshow(cm_z_30, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_yticks(range(len(set(df["cluster"]))))
    ax.set_yticklabels([f"C{c} (n={int((df.cluster==c).sum())}, PH%={(df.loc[df.cluster==c,'label']==1).mean()*100:.0f})"
                        for c in sorted(set(df["cluster"]))])
    ax.set_xticks(range(len(top30)))
    ax.set_xticklabels([f[:22] for f in top30], rotation=80, fontsize=7)
    plt.colorbar(im, ax=ax, label="z-score (across clusters)")
    ax.set_title(f"E1 — Cluster signature heatmap ({win_key}, top-30 features by D1+C1 importance)")
    plt.tight_layout()
    plt.savefig(OUT / "cluster_signature_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Medoid cases (closest to each cluster centroid)
    medoid_rows = []
    for c in sorted(set(df["cluster"])):
        m = df["cluster"] == c
        if m.sum() == 0: continue
        Xc = X[m]
        centroid = Xc.mean(axis=0)
        d = cdist(Xc, centroid.reshape(1, -1)).ravel()
        medoid_idx = int(np.argmin(d))
        global_idx = np.where(m)[0][medoid_idx]
        medoid_rows.append({
            "cluster": int(c),
            "medoid_case_id": str(df.iloc[global_idx]["case_id"]),
            "medoid_label": int(df.iloc[global_idx]["label"]),
            "medoid_mpap": float(df.iloc[global_idx]["measured_mpap"]) if pd.notna(df.iloc[global_idx]["measured_mpap"]) else None,
        })
    pd.DataFrame(medoid_rows).to_csv(OUT / "medoid_cases.csv", index=False)
    print(f"\nsaved all E1 outputs to {OUT}")
    print(f"winner: {win_key}, ARI-perm-p={winner['ari_perm_p']:.3g}")


if __name__ == "__main__":
    main()
