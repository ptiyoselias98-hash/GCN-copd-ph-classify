"""R15.B — Clustering stability sweep (R14 reviewer must-fix).

R14 endotypes used fixed k=3, fixed seed=42. Reviewer flagged: no k-sweep,
no silhouette, no consensus ARI/NMI, no cluster-vs-baseline enrichment test.

This script:
  - Sweeps k = 2..6 on the contrast-only n=184 multi-modal feature vector
  - Reports silhouette score, Calinski-Harabasz, Davies-Bouldin per k
  - Computes consensus ARI across 30 random KMeans seeds per k
  - Tests cluster-PH-enrichment vs the 85.9% PH baseline (binomial)

Outputs: outputs/r15/clustering_stability.{json,md}
         outputs/r15/fig_r15_stability.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                              davies_bouldin_score, silhouette_score)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent.parent
OUT = ROOT / "outputs" / "r15"
OUT.mkdir(parents=True, exist_ok=True)
GRAPH = ROOT / "outputs" / "r5" / "cache_v2_tri_flat_aggregates.csv"
LUNG = ROOT / "outputs" / "lung_features_v2.csv"
LABELS = ROOT / "data" / "labels_expanded_282.csv"
PROTO = ROOT / "data" / "case_protocol.csv"
SEG_FAILS = ROOT / "outputs" / "r13" / "seg_failures_real.json"


def main():
    g_df = pd.read_csv(GRAPH); l_df = pd.read_csv(LUNG)
    labels = pd.read_csv(LABELS); proto = pd.read_csv(PROTO)
    fails = set()
    if SEG_FAILS.exists():
        sf = json.loads(SEG_FAILS.read_text(encoding="utf-8"))
        fails = {r["case_id"] for r in sf.get("real_fails", []) + sf.get("lung_anomaly", [])}

    g_cols = [c for c in g_df.columns if c != "case_id" and pd.api.types.is_numeric_dtype(g_df[c])]
    l_cols = [c for c in l_df.columns if c != "case_id" and pd.api.types.is_numeric_dtype(l_df[c])]
    g_df = g_df[["case_id"] + g_cols].rename(columns={c: f"g_{c}" for c in g_cols})
    l_df = l_df[["case_id"] + l_cols].rename(columns={c: f"l_{c}" for c in l_cols})
    feat_cols = [f"g_{c}" for c in g_cols] + [f"l_{c}" for c in l_cols]

    df = labels.merge(proto[["case_id", "protocol"]], on="case_id", how="left") \
        .merge(g_df, on="case_id", how="left").merge(l_df, on="case_id", how="left")
    contrast = df[df["protocol"].str.lower() == "contrast"].copy()
    contrast = contrast[~contrast["case_id"].isin(fails)]
    contrast = contrast.dropna(subset=feat_cols).reset_index(drop=True)
    X = StandardScaler().fit_transform(contrast[feat_cols].values)
    y = contrast["label"].values.astype(int)
    base_ph = float(y.mean())
    print(f"n={len(X)} (PH={int((y==1).sum())} nonPH={int((y==0).sum())}, baseline PH%={base_ph:.3f})")

    out = {"n": int(len(X)), "n_features": int(X.shape[1]),
           "baseline_ph_frac": base_ph, "k_sweep": {}}

    ks = list(range(2, 7))
    metrics_rows = []
    for k in ks:
        # Single canonical clustering at seed=42
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        sil = float(silhouette_score(X, km.labels_))
        ch = float(calinski_harabasz_score(X, km.labels_))
        db = float(davies_bouldin_score(X, km.labels_))

        # Consensus ARI across 30 seeds
        seeds = list(range(30))
        labelings = []
        for s in seeds:
            kk = KMeans(n_clusters=k, random_state=s, n_init=10).fit(X)
            labelings.append(kk.labels_)
        aris = []
        for i in range(len(labelings)):
            for j in range(i + 1, len(labelings)):
                aris.append(adjusted_rand_score(labelings[i], labelings[j]))
        consensus_ari = float(np.mean(aris))

        # Cluster PH-enrichment vs baseline (binomial test on each cluster)
        enr = []
        for c in range(k):
            mask = km.labels_ == c
            if not mask.any():
                continue
            n_c = int(mask.sum()); n_ph = int(y[mask].sum())
            try:
                p = float(binomtest(n_ph, n_c, p=base_ph).pvalue)
            except Exception:
                p = float("nan")
            enr.append({"cluster": int(c), "n": n_c,
                        "n_ph": n_ph, "ph_frac": float(n_ph / n_c),
                        "binomial_p_vs_baseline": p,
                        "enriched": bool(p < 0.05 and n_ph / n_c > base_ph),
                        "depleted": bool(p < 0.05 and n_ph / n_c < base_ph)})
        out["k_sweep"][f"k={k}"] = {
            "silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db,
            "consensus_ari_30seeds": consensus_ari,
            "cluster_enrichment": enr,
        }
        n_significant = sum(1 for e in enr if e["enriched"] or e["depleted"])
        metrics_rows.append({"k": k, "silhouette": sil, "ari": consensus_ari,
                             "db": db, "n_significant_clusters": n_significant})
        print(f"  k={k}: sil={sil:.3f} ch={ch:.0f} db={db:.3f} "
              f"consensus_ARI={consensus_ari:.3f} sig_clusters={n_significant}/{k}")

    # Pick best k by consensus ARI (most stable)
    best = max(out["k_sweep"].values(), key=lambda v: v["consensus_ari_30seeds"])
    out["best_k_by_ari"] = max(metrics_rows, key=lambda r: r["ari"])

    # Fig
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    rec = pd.DataFrame(metrics_rows)
    axes[0].plot(rec["k"], rec["silhouette"], "o-", c="#3b82f6")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("Silhouette score (higher better)")
    axes[0].set_title("Silhouette across k"); axes[0].grid(alpha=0.3)
    axes[1].plot(rec["k"], rec["ari"], "o-", c="#10b981")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("Consensus ARI (30 seeds, higher better)")
    axes[1].set_title("Stability across seeds"); axes[1].grid(alpha=0.3)
    axes[2].plot(rec["k"], rec["db"], "o-", c="#ef4444")
    axes[2].set_xlabel("k"); axes[2].set_ylabel("Davies-Bouldin (lower better)")
    axes[2].set_title("Cluster separation"); axes[2].grid(alpha=0.3)
    plt.suptitle(f"R15.B clustering stability (contrast-only n={len(X)}, baseline PH%={base_ph:.1%})", y=1.02)
    plt.tight_layout()
    fig_path = OUT / "fig_r15_stability.png"
    plt.savefig(fig_path, dpi=140, bbox_inches="tight"); plt.close()

    (OUT / "clustering_stability.json").write_text(json.dumps(out, indent=2),
                                                      encoding="utf-8")

    md = ["# R15.B — Clustering stability sweep",
          "",
          f"Contrast-only cohort (n={out['n']}, n_features={out['n_features']}, "
          f"baseline PH%={base_ph:.1%}). Sweep k=2..6.",
          "",
          "| k | silhouette | calinski-harabasz | davies-bouldin | consensus ARI (30 seeds) | sig-enriched/depleted clusters |",
          "|---|---|---|---|---|---|"]
    for k, r in zip(ks, metrics_rows):
        ksk = out["k_sweep"][f"k={k}"]
        n_sig = r["n_significant_clusters"]
        md.append(f"| {k} | {ksk['silhouette']:.3f} | {ksk['calinski_harabasz']:.0f} | "
                  f"{ksk['davies_bouldin']:.3f} | {ksk['consensus_ari_30seeds']:.3f} | "
                  f"{n_sig}/{k} |")

    md += ["",
           f"**Best k by consensus ARI**: k={out['best_k_by_ari']['k']} "
           f"(ARI={out['best_k_by_ari']['ari']:.3f}, silhouette="
           f"{out['best_k_by_ari']['silhouette']:.3f}).",
           "",
           "## Cluster-PH enrichment per k (binomial test vs baseline)",
           ""]
    for k in ks:
        ksk = out["k_sweep"][f"k={k}"]
        md.append(f"### k={k}")
        md.append("")
        md.append("| cluster | n | n_PH | PH% | binom p (vs baseline) | enriched | depleted |")
        md.append("|---|---|---|---|---|---|---|")
        for e in ksk["cluster_enrichment"]:
            md.append(f"| C{e['cluster']} | {e['n']} | {e['n_ph']} | "
                      f"{e['ph_frac']:.1%} | {e['binomial_p_vs_baseline']:.3g} | "
                      f"{'✓' if e['enriched'] else ''} | {'✓' if e['depleted'] else ''} |")
        md.append("")

    md.append("![stability sweep](fig_r15_stability.png)")
    (OUT / "clustering_stability.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Saved {OUT}/clustering_stability.json + .md + fig_r15_stability.png")


if __name__ == "__main__":
    main()
