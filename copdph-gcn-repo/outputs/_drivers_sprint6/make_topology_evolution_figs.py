"""Topology-evolution figures — the honest story.

We probed unsupervised PH-vs-nonPH separability using three label-free
topology views (A: WL kernel, B: graph stats, C: GAE self-supervised)
on the 269-case cohort and, separately, a "clean-segmentation" subset of
141 cases where all three structures have non-trivial trees.

Panel 1 — best-ARI bar by method, raw n=269 vs filtered n=141.
Panel 2 — the 3-cluster structure at the raw "best" operating point
          (C_GAE spectral k=3), showing that the split follows
          segmentation-completeness, not topology.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(r"E:\桌面文件\图卷积-肺小血管演化规律探索\outputs\p_zeta_cluster_269\topology_evolution")
OUT  = ROOT

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
    "xtick.labelsize": 9.5, "ytick.labelsize": 9,
    "legend.frameon": False, "legend.fontsize": 9,
    "savefig.dpi": 160, "savefig.bbox": "tight",
    "figure.facecolor": "white",
})

C_RAW  = "#E07A5F"   # warm — n=269 (confounded)
C_FLT  = "#2E86AB"   # cool — n=141 (clean)


def _best_by_exp(rows):
    out = {}
    for r in rows:
        e = r["experiment"]
        if e not in out or r["ARI_vs_PH"] > out[e]["ARI_vs_PH"]:
            out[e] = r
    return out


raw  = json.loads((ROOT / "topo_summary.json").read_text())
flt  = json.loads((ROOT / "topo_summary_filtered.json").read_text())

raw_best = _best_by_exp(raw["rows"])
flt_best = _best_by_exp(flt["rows"])

exps   = ["A_WL", "B_stats", "C_GAE"]
labels = ["A — WL kernel", "B — graph stats", "C — GAE (SSL)"]

raw_ari = [raw_best[e]["ARI_vs_PH"] for e in exps]
flt_ari = [flt_best[e]["ARI_vs_PH"] for e in exps]
raw_tag = [f"{raw_best[e]['method']} k={raw_best[e]['k']}" for e in exps]
flt_tag = [f"{flt_best[e]['method']} k={flt_best[e]['k']}" for e in exps]

x = np.arange(len(exps))
w = 0.36

fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))

# ---------------- Panel 1 — ARI raw vs filtered
ax = axes[0]
n_raw = raw["n_cases"]
n_flt = flt["n_kept"]
ax.bar(x - w/2, raw_ari, w, color=C_RAW, edgecolor="white", linewidth=0.6, alpha=0.92,
       label=f"raw n={n_raw}  (105 non-PH / 164 PH)")
ax.bar(x + w/2, flt_ari, w, color=C_FLT, edgecolor="white", linewidth=0.6, alpha=0.92,
       label=f"filtered n={n_flt}  (21 non-PH / 120 PH)")

for i, (a, tag) in enumerate(zip(raw_ari, raw_tag)):
    ax.text(x[i] - w/2, a + 0.012, f"{a:.3f}",
            ha="center", va="bottom", fontsize=9, color=C_RAW, weight="semibold")
    ax.text(x[i] - w/2, -0.035, tag,
            ha="center", va="top", fontsize=8, color="#444")
for i, (a, tag) in enumerate(zip(flt_ari, flt_tag)):
    ax.text(x[i] + w/2, a + 0.012, f"{a:.3f}",
            ha="center", va="bottom", fontsize=9, color=C_FLT, weight="semibold")
    ax.text(x[i] + w/2, -0.035, tag,
            ha="center", va="top", fontsize=8, color="#444")

ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(-0.07, 0.70)
ax.axhline(0.0, color="#888", lw=0.7)
ax.set_ylabel("ARI vs PH label  (best over k, method)")
ax.set_title("Unsupervised topology ↔ PH separability\n"
             "collapses once degenerate-segmentation cases are removed",
             fontsize=11.5)
ax.legend(loc="upper right")
ax.grid(axis="y", linestyle=":", color="#aaa", alpha=0.6)
ax.set_axisbelow(True)

# ---------------- Panel 2 — raw best cluster profile vs segmentation size
# Use the raw best (C_GAE spectral k=3): cluster sizes [189, 23, 57].
# Pull per-cluster profile to show artery_n_nodes, airway_n_nodes, vein_n_nodes.
prof = json.loads((ROOT / "topo_best_cluster_profile.json").read_text())
best = prof["best"]
clusters = prof["profile"]
clusters = sorted(clusters, key=lambda c: -c["n"])  # largest first

ax = axes[1]
names = [f"cluster {c['cluster']}\nn={c['n']}, PH={c['ph_rate']*100:.1f}%"
         for c in clusters]
art   = [c["artery_n_nodes"] for c in clusters]
vein  = [c["vein_n_nodes"]   for c in clusters]
air   = [c["airway_n_nodes"] for c in clusters]

xc = np.arange(len(clusters))
bw = 0.24

ax.bar(xc - bw, art,  bw, color="#C0392B", edgecolor="white", linewidth=0.6, label="artery nodes")
ax.bar(xc,      vein, bw, color="#2E86AB", edgecolor="white", linewidth=0.6, label="vein nodes")
ax.bar(xc + bw, air,  bw, color="#27AE60", edgecolor="white", linewidth=0.6, label="airway nodes")

for i, (a, v, b) in enumerate(zip(art, vein, air)):
    ax.text(xc[i] - bw, a + 3, f"{a:.0f}", ha="center", va="bottom", fontsize=8.5, color="#C0392B")
    ax.text(xc[i],      v + 3, f"{v:.0f}", ha="center", va="bottom", fontsize=8.5, color="#2E86AB")
    ax.text(xc[i] + bw, b + 3, f"{b:.0f}", ha="center", va="bottom", fontsize=8.5, color="#27AE60")

ax.set_xticks(xc); ax.set_xticklabels(names)
ax.set_ylabel("mean per-structure node count")
ax.set_title(f"What the raw best cluster (C_GAE spectral k=3, ARI={best['ARI_vs_PH']:.3f}) is actually separating\n"
             "→ clusters are segmentation-completeness buckets, not topology phenotypes",
             fontsize=11.5)
ax.legend(loc="upper right")
ax.grid(axis="y", linestyle=":", color="#aaa", alpha=0.6)
ax.set_axisbelow(True)

plt.tight_layout()
out = OUT / "topo_evolution_raw_vs_filtered.png"
plt.savefig(out)
print(f"wrote {out}")

# Also drop a copy next to the other Sprint-6 figures for the README
out2 = Path(r"E:\桌面文件\图卷积-肺小血管演化规律探索\copdph-gcn-repo\outputs\_drivers_sprint6") / "topo_evolution_raw_vs_filtered.png"
plt.savefig(out2)
print(f"wrote {out2}")
