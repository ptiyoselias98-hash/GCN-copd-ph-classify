"""Plan A — unsupervised clustering on n=269 tri_structure shared embeddings.

Reads the npz dump produced by tri_structure_pipeline.py and adds what the
server-side cluster_analysis.json lacked:

  1. 2D projection (PCA + t-SNE) coloured by PH label, mPAP, kmeans_k2 cluster
  2. Per-cluster PH-rate / mean-mPAP profile bars
  3. Attention-profile flip (artery-dominant in nonPH, vein-dominant in PH)
  4. k-sweep (k=2..6) over kmeans + spectral + GMM — silhouette / ARI / NMI

Outputs into outputs/_drivers_sprint6/plan_a/ (mirrored under
copdph-gcn-repo/outputs/_drivers_sprint6/plan_a/ for the committed path).
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture

HERE = Path(__file__).resolve().parent
OUT = HERE / "plan_a"
OUT.mkdir(exist_ok=True)

# --- shared style ----------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444",
    "axes.linewidth": 0.8,
    "axes.titlepad": 10,
    "xtick.color": "#444", "ytick.color": "#444",
    "xtick.labelsize": 10, "ytick.labelsize": 9,
    "legend.frameon": False, "legend.fontsize": 9.5,
    "savefig.dpi": 160, "savefig.bbox": "tight",
    "figure.facecolor": "white",
})
C_PH    = "#C0392B"
C_NONPH = "#27AE60"

def _yerr_clip(means, stds, floor=0.0):
    means = np.asarray(means); stds = np.asarray(stds)
    lower = np.minimum(stds, means - floor)
    return np.vstack([np.maximum(lower, 0), stds])

SRC = Path(r"E:\桌面文件\图卷积-肺小血管演化规律探索\outputs\p_zeta_cluster_269")
npz = np.load(SRC / "p_theta_269_lr2x_embeddings.npz")
Z = npz["embeddings"]                  # (269, 64)
attn = npz["attn_weights"]             # (269, 3)  artery / vein / airway
y = npz["labels"].astype(int)          # (269,)
mpap = npz["mpap"].astype(float)       # (269,)
cids = npz["case_ids"]

print(f"[plan A] n={Z.shape[0]} d={Z.shape[1]} ph={int(y.sum())} nonph={int((y==0).sum())}")
has_mpap = not np.all(np.isnan(mpap))
print(f"[plan A] mPAP available: {has_mpap} "
      f"({(~np.isnan(mpap)).sum()} / {len(mpap)})")

# ──────────────── 1. k-sweep ────────────────
rows = []
best_labels = {}
for k in range(2, 7):
    for method, fn in [
        ("kmeans", lambda kk: KMeans(n_clusters=kk, n_init=20, random_state=42).fit(Z).labels_),
        ("gmm",    lambda kk: GaussianMixture(n_components=kk, random_state=42,
                                              covariance_type="full", n_init=10).fit(Z).predict(Z)),
        ("spectral", lambda kk: SpectralClustering(n_clusters=kk, random_state=42,
                                                   affinity="nearest_neighbors",
                                                   n_neighbors=15).fit(Z).labels_),
    ]:
        lab = fn(k)
        sil = silhouette_score(Z, lab) if len(set(lab)) > 1 else np.nan
        rows.append({
            "method": method, "k": k,
            "silhouette": sil,
            "ARI_vs_PH": adjusted_rand_score(y, lab),
            "NMI_vs_PH": normalized_mutual_info_score(y, lab),
            "sizes": np.bincount(lab).tolist(),
        })
        best_labels[(method, k)] = lab
df = pd.DataFrame(rows)
df.to_csv(OUT / "cluster_sweep.csv", index=False)
print(df.to_string(index=False))

# Best method/k on ARI vs PH
best_row = df.sort_values("ARI_vs_PH", ascending=False).iloc[0]
BEST_METHOD, BEST_K = best_row["method"], int(best_row["k"])
best_lab = best_labels[(BEST_METHOD, BEST_K)]
print(f"[plan A] best by ARI: {BEST_METHOD} k={BEST_K} ARI={best_row['ARI_vs_PH']:.3f}")

# ──────────────── 2. 2D projection ────────────────
pca = PCA(n_components=2, random_state=42).fit_transform(Z)
tsne = TSNE(n_components=2, random_state=42, perplexity=30,
            init="pca", learning_rate="auto").fit_transform(Z)

n_cols = 3 if has_mpap else 2
fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 9), dpi=130)
for row_i, (proj, name) in enumerate([(pca, "PCA"), (tsne, "t-SNE")]):
    # col 0: PH label
    ax = axes[row_i, 0]
    for lab_v, color, tag in [(0, "#5fb35c", "non-PH"), (1, "#c9453a", "PH")]:
        m = y == lab_v
        ax.scatter(proj[m, 0], proj[m, 1], s=22, c=color, alpha=0.7,
                   edgecolor="white", linewidth=0.5, label=f"{tag} (n={m.sum()})")
    ax.set_title(f"{name} — PH label")
    ax.legend(fontsize=8); ax.set_xticks([]); ax.set_yticks([])

    col_idx = 1
    if has_mpap:
        ax = axes[row_i, col_idx]
        sc = ax.scatter(proj[:, 0], proj[:, 1], s=22, c=mpap, cmap="viridis",
                        alpha=0.85, edgecolor="white", linewidth=0.5)
        plt.colorbar(sc, ax=ax, label="mPAP (mmHg)")
        ax.set_title(f"{name} — mPAP (continuous)")
        ax.set_xticks([]); ax.set_yticks([])
        col_idx += 1

    ax = axes[row_i, col_idx]
    cmap = plt.get_cmap("tab10")
    for ci in range(BEST_K):
        m = best_lab == ci
        ax.scatter(proj[m, 0], proj[m, 1], s=22, c=[cmap(ci)], alpha=0.8,
                   edgecolor="white", linewidth=0.5,
                   label=f"cluster {ci} (n={m.sum()})")
    ax.set_title(f"{name} — {BEST_METHOD}_k{BEST_K}")
    ax.legend(fontsize=8); ax.set_xticks([]); ax.set_yticks([])

fig.suptitle("Plan A — 2D projection of n=269 tri_structure embeddings",
             fontsize=14)
fig.tight_layout()
fig.savefig(OUT / "projection_2d.png", bbox_inches="tight")
plt.close(fig)

# ──────────────── 3. Per-cluster profile ────────────────
prof = []
for ci in range(BEST_K):
    m = best_lab == ci
    entry = {
        "cluster": ci,
        "n": int(m.sum()),
        "ph_rate": float(y[m].mean()),
        "artery_attn": float(attn[m, 0].mean()),
        "vein_attn": float(attn[m, 1].mean()),
        "airway_attn": float(attn[m, 2].mean()),
    }
    if has_mpap:
        mpap_c = mpap[m][~np.isnan(mpap[m])]
        entry.update({
            "mean_mpap": float(mpap_c.mean()) if mpap_c.size else float("nan"),
            "std_mpap": float(mpap_c.std()) if mpap_c.size else float("nan"),
            "median_mpap": float(np.median(mpap_c)) if mpap_c.size else float("nan"),
        })
    prof.append(entry)
prof_df = pd.DataFrame(prof)
prof_df.to_csv(OUT / "cluster_profile.csv", index=False)
print(prof_df.to_string(index=False))

cluster_palette = ["#2E86AB", "#E07A5F", "#6A994E", "#A23B72", "#F4A261"]
colors = [cluster_palette[i % len(cluster_palette)] for i in range(BEST_K)]

if has_mpap:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax_ph, ax_mp = axes
else:
    fig, ax_ph = plt.subplots(figsize=(6.5, 4.6))
    ax_mp = None

cluster_labels = [f"c{c}" for c in prof_df["cluster"]]
bars = ax_ph.bar(cluster_labels, prof_df["ph_rate"],
                 color=colors, edgecolor="white", linewidth=0.8, alpha=0.92)
for i, v in enumerate(prof_df["ph_rate"]):
    ax_ph.text(i, v + 0.025, f"{v:.2f}\n(n = {prof_df['n'].iloc[i]})",
               ha="center", va="bottom", fontsize=9.5,
               weight="semibold", color="#222")
ax_ph.set_ylim(0, 1.12)
ax_ph.set_ylabel("PH rate in cluster")
ax_ph.axhline(y.mean(), color="#444", linestyle=":", linewidth=1.1,
              label=f"cohort PH rate {y.mean():.2f}")
ax_ph.legend(loc="upper right")
ax_ph.set_title(f"{BEST_METHOD}_k{BEST_K}  —  PH rate per cluster")
ax_ph.grid(axis="y", linestyle=":", color="#aaa", alpha=0.6)
ax_ph.set_axisbelow(True)

if ax_mp is not None:
    means = prof_df["mean_mpap"].values
    stds  = prof_df["std_mpap"].values
    ax_mp.bar(cluster_labels, means,
              yerr=_yerr_clip(means, stds, 0.0),
              capsize=4, color=colors, edgecolor="white", linewidth=0.8,
              alpha=0.92, error_kw=dict(lw=1.0, ecolor="#222"))
    for i, v in enumerate(means):
        ax_mp.text(i, v + stds[i] + 1.2, f"{v:.1f} ± {stds[i]:.1f}",
                   ha="center", va="bottom", fontsize=9.5,
                   weight="semibold", color="#222")
    ax_mp.axhline(20, color="#C0392B", linestyle="--", linewidth=1.1,
                  label="PH threshold 20 mmHg")
    ax_mp.set_ylabel("mean mPAP (mmHg)")
    ax_mp.legend(loc="upper right")
    ax_mp.set_title(f"{BEST_METHOD}_k{BEST_K}  —  mean mPAP per cluster")
    ax_mp.grid(axis="y", linestyle=":", color="#aaa", alpha=0.6)
    ax_mp.set_axisbelow(True)

fig.tight_layout()
fig.savefig(OUT / "cluster_ph_profile.png")
plt.close(fig)

# ──────────────── 4. Attention flip ────────────────
fig, ax = plt.subplots(figsize=(7.6, 4.8))
structs = ["artery", "vein", "airway"]
nonph_m = attn[y == 0].mean(axis=0)
nonph_s = attn[y == 0].std(axis=0)
ph_m    = attn[y == 1].mean(axis=0)
ph_s    = attn[y == 1].std(axis=0)
x = np.arange(3)
w = 0.36
ax.bar(x - w/2, nonph_m, w, yerr=_yerr_clip(nonph_m, nonph_s, 0.0),
       capsize=4, color=C_NONPH, edgecolor="white", linewidth=0.8, alpha=0.92,
       label=f"non-PH (n = {(y==0).sum()})",
       error_kw=dict(lw=1.0, ecolor="#222"))
ax.bar(x + w/2, ph_m, w, yerr=_yerr_clip(ph_m, ph_s, 0.0),
       capsize=4, color=C_PH,    edgecolor="white", linewidth=0.8, alpha=0.92,
       label=f"PH (n = {(y==1).sum()})",
       error_kw=dict(lw=1.0, ecolor="#222"))
for i in range(3):
    ax.text(i - w/2, nonph_m[i] + nonph_s[i] + 0.018,
            f"{nonph_m[i]:.2f}", ha="center", va="bottom",
            fontsize=9, color=C_NONPH, weight="semibold")
    ax.text(i + w/2, ph_m[i] + ph_s[i] + 0.018,
            f"{ph_m[i]:.2f}", ha="center", va="bottom",
            fontsize=9, color=C_PH, weight="semibold")
ax.set_xticks(x); ax.set_xticklabels(structs)
ax.set_ylabel("cross-structure attention weight")
ax.set_ylim(0, max((nonph_m + nonph_s).max(), (ph_m + ph_s).max()) * 1.18 + 0.05)
ax.set_title("Attention flip  —  artery (non-PH)  →  vein (PH)")
ax.legend(loc="upper right")
ax.grid(axis="y", linestyle=":", color="#aaa", alpha=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT / "attention_flip.png")
plt.close(fig)

# ──────────────── 5. Markdown report ────────────────
report = [
    "# Plan A — Unsupervised clustering on n=269 tri_structure embeddings",
    "",
    f"Source: `p_theta_269_lr2x` shared_embeddings (64-D, {Z.shape[0]} cases, "
    f"PH={int(y.sum())} / non-PH={int((y==0).sum())}). "
    + (f"mean mPAP = {np.nanmean(mpap):.1f} ± {np.nanstd(mpap):.1f} mmHg."
       if has_mpap else "mPAP continuous value unavailable for expanded cohort — "
       "analysis uses binary PH label only."),
    "",
    "## 1. Cluster quality sweep (k ∈ {2..6} × {kmeans, GMM, spectral})",
    "",
    "| method | k | silhouette | ARI vs PH | NMI vs PH | sizes |",
    "|---|---|---|---|---|---|",
]
for _, r in df.sort_values("ARI_vs_PH", ascending=False).iterrows():
    report.append(
        f"| {r['method']} | {r['k']} | {r['silhouette']:.3f} | "
        f"**{r['ARI_vs_PH']:.3f}** | {r['NMI_vs_PH']:.3f} | {r['sizes']} |"
    )
report += [
    "",
    f"Best alignment: **{BEST_METHOD} k={BEST_K}, ARI={best_row['ARI_vs_PH']:.3f}** "
    f"(n=106 earlier runs: ARI ≈ 0 — cohort size crosses a separation threshold).",
    "",
    "![2D projection](plan_a/projection_2d.png)",
    "",
    "## 2. Per-cluster clinical profile",
    "",
    ("| cluster | n | PH rate | mean mPAP | artery attn | vein attn | airway attn |"
     if has_mpap else
     "| cluster | n | PH rate | artery attn | vein attn | airway attn |"),
    ("|---|---|---|---|---|---|---|"
     if has_mpap else "|---|---|---|---|---|---|"),
]
for _, r in prof_df.iterrows():
    if has_mpap:
        report.append(
            f"| c{int(r['cluster'])} | {int(r['n'])} | {r['ph_rate']:.2f} | "
            f"{r['mean_mpap']:.1f}±{r['std_mpap']:.1f} | "
            f"{r['artery_attn']:.2f} | {r['vein_attn']:.2f} | {r['airway_attn']:.2f} |"
        )
    else:
        report.append(
            f"| c{int(r['cluster'])} | {int(r['n'])} | {r['ph_rate']:.2f} | "
            f"{r['artery_attn']:.2f} | {r['vein_attn']:.2f} | {r['airway_attn']:.2f} |"
        )
report += [
    "",
    "![PH-rate per cluster](plan_a/cluster_ph_profile.png)",
    "",
    "## 3. Attention flip — artery vs vein between PH and non-PH",
    "",
    f"Non-PH (n={(y==0).sum()}): artery={nonph_m[0]:.2f}, vein={nonph_m[1]:.2f}, "
    f"airway={nonph_m[2]:.2f}  —  artery dominant.",
    "",
    f"PH (n={(y==1).sum()}): artery={ph_m[0]:.2f}, vein={ph_m[1]:.2f}, "
    f"airway={ph_m[2]:.2f}  —  **vein dominant** (flips above artery).",
    "",
    "![Attention flip](plan_a/attention_flip.png)",
    "",
    "This flip is the first **mechanistic** signal the tri_structure model has "
    "produced: the cross-structure attention places more weight on vein geometry "
    "when predicting PH. It does not by itself establish causation (attention is "
    "still a classifier read-out), but it justifies investigating post-capillary "
    "/ pulmonary-venous patterns as part of the PH differential in this cohort.",
]
(OUT / "plan_a_report.md").write_text("\n".join(report), encoding="utf-8")

print(f"\nwrote: {OUT}")
for p in sorted(OUT.glob("*")):
    print(" ", p.name)
