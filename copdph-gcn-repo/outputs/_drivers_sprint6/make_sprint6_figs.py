"""Generate sprint 6 figures from the committed CV result dumps.

Outputs (same dir as this script):
  sprint6_auc_bar.png          horizontal AUC ± std across 10 variants
  sprint6_lr_sensitivity.png   AUC vs LR, two cohort sizes
  sprint6_radar.png            6-metric radar across 5 configs
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

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
    "xtick.color": "#444",
    "ytick.color": "#444",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})
C269 = "#2E86AB"   # blue (n=269)
C106 = "#E07A5F"   # warm orange (n=106)
CBASE = "#3A3A3A"  # baseline

with open(REPO / "_cv_results_6jobs.json", "r", encoding="utf-8") as f:
    six = json.load(f)
with open(REPO / "_cv_results_lrsweep.json", "r", encoding="utf-8") as f:
    lrs = json.load(f)

rows = []
for name, blk in {**six, **lrs}.items():
    rows.append({
        "name": name,
        "n": blk["n_cases"],
        "auc": blk["mean_metrics"]["auc"]["mean"],
        "auc_std": blk["mean_metrics"]["auc"]["std"],
    })
rows.sort(key=lambda r: r["auc"])

BASELINE = 0.944  # arm_a_ensemble (sprint 5)

# --- Figure 1: AUC bar chart -----------------------------------------------
fig, ax = plt.subplots(figsize=(10.2, 5.6))
labels = [r["name"] for r in rows]
aucs = np.array([r["auc"] for r in rows])
stds = np.array([r["auc_std"] for r in rows])
colors = [C106 if r["n"] == 106 else C269 for r in rows]

y = np.arange(len(rows))
bars = ax.barh(y, aucs, xerr=stds, color=colors, edgecolor="white",
               linewidth=0.8, alpha=0.92,
               error_kw=dict(ecolor="#222", lw=1.0, capsize=3))
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9.5)
ax.set_xlabel("AUC  (5-fold × 3-rep CV)")
ax.set_xlim(0.55, 1.04)
ax.axvline(BASELINE, color=CBASE, linestyle="--", linewidth=1.1,
           label=f"arm_a_ensemble baseline ({BASELINE:.3f})")
for i, r in enumerate(rows):
    ax.text(r["auc"] + r["auc_std"] + 0.008, i,
            f"{r['auc']:.3f} ± {r['auc_std']:.3f}",
            va="center", fontsize=8.5, color="#222")

legend_handles = [
    Patch(facecolor=C269, edgecolor="white", label="n = 269 (expanded)"),
    Patch(facecolor=C106, edgecolor="white", label="n = 106 (gold)"),
]
# baseline line handle
from matplotlib.lines import Line2D
legend_handles.append(Line2D([0], [0], color=CBASE, lw=1.2, ls="--",
                             label=f"arm_a_ensemble ({BASELINE:.3f})"))
ax.legend(handles=legend_handles, loc="lower right", fontsize=9,
          handlelength=2.0)
ax.set_title("Sprint 6 — tri-structure GCN AUC across 10 variants")
ax.grid(axis="x", linestyle=":", color="#aaa", alpha=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(HERE / "sprint6_auc_bar.png")
plt.close(fig)

# --- Figure 2: LR sensitivity ----------------------------------------------
lr_points = {
    269: [
        (5e-4, lrs["p_theta_269_lrhalf"]["mean_metrics"]["auc"]["mean"],
               lrs["p_theta_269_lrhalf"]["mean_metrics"]["auc"]["std"]),
        (1e-3, six["p_zeta_tri_282"]["mean_metrics"]["auc"]["mean"],
               six["p_zeta_tri_282"]["mean_metrics"]["auc"]["std"]),
        (2e-3, lrs["p_theta_269_lr2x"]["mean_metrics"]["auc"]["mean"],
               lrs["p_theta_269_lr2x"]["mean_metrics"]["auc"]["std"]),
    ],
    106: [
        (5e-4, lrs["p_theta_106_lrhalf"]["mean_metrics"]["auc"]["mean"],
               lrs["p_theta_106_lrhalf"]["mean_metrics"]["auc"]["std"]),
        (2e-3, lrs["p_theta_106_lr2x"]["mean_metrics"]["auc"]["mean"],
               lrs["p_theta_106_lr2x"]["mean_metrics"]["auc"]["std"]),
    ],
}

fig, ax = plt.subplots(figsize=(8.2, 4.8))
for n, pts, color, marker in [
    (269, lr_points[269], C269, "o"),
    (106, lr_points[106], C106, "s"),
]:
    xs = [p[0] for p in pts]
    ys = np.array([p[1] for p in pts])
    es = np.array([p[2] for p in pts])
    ax.errorbar(xs, ys, yerr=es, fmt=marker + "-", color=color,
                linewidth=2.2, markersize=10, capsize=4,
                markeredgecolor="white", markeredgewidth=1.0,
                label=f"n = {n}", elinewidth=1.1)
    for x_, y_, e_ in zip(xs, ys, es):
        ax.annotate(f"{y_:.3f}", xy=(x_, y_ + e_),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8.5, color=color, weight="semibold")

ax.axhline(BASELINE, color=CBASE, linestyle="--", linewidth=1.1,
           label=f"arm_a_ensemble ({BASELINE:.3f})")
ax.set_xscale("log")
ax.set_xticks([5e-4, 1e-3, 2e-3])
ax.set_xticklabels(["5e-4", "1e-3", "2e-3"])
ax.set_xlabel("learning rate (Adam, cosine schedule)")
ax.set_ylabel("AUC  (5-fold × 3-rep CV)")
ax.set_ylim(0.55, 1.0)
ax.set_title("Sprint 6 — LR sensitivity (tri_structure, pool=mean, mpap_aux)")
ax.legend(loc="lower right")
ax.grid(linestyle=":", color="#aaa", alpha=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(HERE / "sprint6_lr_sensitivity.png")
plt.close(fig)

# --- Figure 3: 6-metric radar ----------------------------------------------
def get_metrics(blk):
    m = blk["mean_metrics"]
    return [m["auc"]["mean"], m["accuracy"]["mean"], m["sensitivity"]["mean"],
            m["specificity"]["mean"], m["f1"]["mean"], m["precision"]["mean"]]

configs = [
    ("p_theta_269_lr2x  (lr=2e-3, n=269)", get_metrics(lrs["p_theta_269_lr2x"]), "#1B4F72"),
    ("p_zeta_sig  (signature, n=269)",      get_metrics(six["p_zeta_sig"]),     "#2E86AB"),
    ("p_zeta_tri_282  (default, n=269)",    get_metrics(six["p_zeta_tri_282"]), "#7FB3D5"),
    ("p_eta_pool_attn  (best n=106)",       get_metrics(six["p_eta_pool_attn"]), "#E07A5F"),
    ("p_theta_106_lrhalf  (worst n=106)",   get_metrics(lrs["p_theta_106_lrhalf"]), "#A23B2C"),
]

labels_r = ["AUC", "Accuracy", "Sensitivity", "Specificity", "F1", "Precision"]
N = len(labels_r)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8.5, 8.0), subplot_kw=dict(polar=True))
fig.patch.set_facecolor("white")
for name, vals, color in configs:
    v = vals + vals[:1]
    ax.plot(angles, v, "-o", color=color, linewidth=2.0, markersize=4.5,
            label=name)
    ax.fill(angles, v, color=color, alpha=0.10)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels_r, fontsize=11)
ax.set_ylim(0.5, 1.0)
ax.set_yticks([0.6, 0.7, 0.8, 0.9])
ax.set_yticklabels(["0.6", "0.7", "0.8", "0.9"], fontsize=8, color="#666")
ax.set_rlabel_position(90)
ax.grid(True, linestyle=":", color="#aaa", alpha=0.6)
ax.spines["polar"].set_color("#bbb")
ax.spines["polar"].set_linewidth(0.8)

# emphasised 0.9 reference ring
ax.plot(angles, [0.9] * (N + 1), "--", color="#222", linewidth=0.9, alpha=0.45)

ax.set_title("Sprint 6 — 6-metric radar  (5 configs, 5-fold × 3-rep CV)",
             fontsize=13, pad=22)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18),
          fontsize=9.5, ncol=1, frameon=False)
fig.tight_layout()
fig.savefig(HERE / "sprint6_radar.png")
plt.close(fig)

print("wrote:")
print(" ", HERE / "sprint6_auc_bar.png")
print(" ", HERE / "sprint6_lr_sensitivity.png")
print(" ", HERE / "sprint6_radar.png")
