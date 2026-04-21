"""Generate sprint 6 figures from the committed CV result dumps.

Outputs (same dir as this script):
  sprint6_auc_bar.png          horizontal AUC ± std across 10 variants
  sprint6_lr_sensitivity.png   AUC vs LR, two cohort sizes
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]  # copdph-gcn-repo/

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

BASELINE = 0.944  # arm_a_ensemble yesterday

# --- Figure 1: AUC bar chart
fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=130)
labels = [r["name"] for r in rows]
aucs = [r["auc"] for r in rows]
stds = [r["auc_std"] for r in rows]
colors = ["#d98b5f" if r["n"] == 106 else "#4a7ab8" for r in rows]

y = np.arange(len(rows))
ax.barh(y, aucs, xerr=stds, color=colors, edgecolor="black", linewidth=0.6,
        error_kw=dict(ecolor="black", lw=1, capsize=3))
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("AUC (5-fold × 3-rep CV)")
ax.set_xlim(0.55, 0.98)
ax.axvline(BASELINE, color="#222", linestyle="--", linewidth=1,
           label=f"arm_a_ensemble baseline ({BASELINE:.3f})")
for i, r in enumerate(rows):
    ax.text(r["auc"] + r["auc_std"] + 0.005, i,
            f"{r['auc']:.3f}±{r['auc_std']:.3f}",
            va="center", fontsize=8)

from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor="#4a7ab8", edgecolor="black", label="n = 269 (expanded)"),
    Patch(facecolor="#d98b5f", edgecolor="black", label="n = 106 (gold)"),
]
ax.legend(handles=legend_handles + ax.get_legend_handles_labels()[0],
          loc="lower right", fontsize=9)
ax.set_title("Sprint 6 — tri-structure GCN AUC across 10 variants")
ax.grid(axis="x", linestyle=":", alpha=0.5)
fig.tight_layout()
fig.savefig(HERE / "sprint6_auc_bar.png")
plt.close(fig)

# --- Figure 2: LR sensitivity (pool=mean, mpap_aux, use_sig=False)
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

fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=130)
for n, pts, color, marker in [
    (269, lr_points[269], "#4a7ab8", "o"),
    (106, lr_points[106], "#d98b5f", "s"),
]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    es = [p[2] for p in pts]
    ax.errorbar(xs, ys, yerr=es, fmt=marker + "-", color=color,
                linewidth=2, markersize=9, capsize=4,
                label=f"n = {n}")
    for x, y, e in zip(xs, ys, es):
        ax.text(x, y + e + 0.006, f"{y:.3f}", ha="center", fontsize=8,
                color=color)

ax.axhline(BASELINE, color="#222", linestyle="--", linewidth=1,
           label=f"arm_a_ensemble ({BASELINE:.3f})")
ax.set_xscale("log")
ax.set_xticks([5e-4, 1e-3, 2e-3])
ax.set_xticklabels(["5e-4", "1e-3", "2e-3"])
ax.set_xlabel("learning rate (Adam, cosine schedule)")
ax.set_ylabel("AUC (5-fold × 3-rep CV)")
ax.set_ylim(0.55, 0.98)
ax.set_title("Sprint 6 — LR sensitivity (tri_structure, pool=mean, mpap_aux)")
ax.legend(loc="lower right", fontsize=9)
ax.grid(linestyle=":", alpha=0.5)
fig.tight_layout()
fig.savefig(HERE / "sprint6_lr_sensitivity.png")
plt.close(fig)

# --- Figure 3: 6-metric radar — 5 configs head-to-head
def get_metrics(blk):
    m = blk["mean_metrics"]
    return [
        m["auc"]["mean"],
        m["accuracy"]["mean"],
        m["sensitivity"]["mean"],
        m["specificity"]["mean"],
        m["f1"]["mean"],
        m["precision"]["mean"],
    ]

configs = [
    ("p_theta_269_lr2x (lr=2e-3, n=269)", get_metrics(lrs["p_theta_269_lr2x"]), "#1a4e8a"),
    ("p_zeta_sig (signature, n=269)", get_metrics(six["p_zeta_sig"]), "#4a7ab8"),
    ("p_zeta_tri_282 (default, n=269)", get_metrics(six["p_zeta_tri_282"]), "#8ab4e2"),
    ("p_eta_pool_attn (best n=106)", get_metrics(six["p_eta_pool_attn"]), "#d98b5f"),
    ("p_theta_106_lrhalf (worst n=106)", get_metrics(lrs["p_theta_106_lrhalf"]), "#c9453a"),
]

labels_r = ["AUC", "Accuracy", "Sensitivity", "Specificity", "F1", "Precision"]
N = len(labels_r)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=130,
                       subplot_kw=dict(polar=True))
for name, vals, color in configs:
    v = vals + vals[:1]
    ax.plot(angles, v, "-", color=color, linewidth=2, label=name)
    ax.fill(angles, v, color=color, alpha=0.08)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels_r, fontsize=11)
ax.set_ylim(0.5, 1.0)
ax.set_yticks([0.6, 0.7, 0.8, 0.9, 0.95])
ax.set_yticklabels(["0.6", "0.7", "0.8", "0.9", "0.95"], fontsize=8)
ax.set_rlabel_position(90)
ax.grid(True, linestyle=":", alpha=0.6)

# emphasise 0.9 reference ring
circle_vals = [0.9] * (N + 1)
ax.plot(angles, circle_vals, "--", color="#222", linewidth=1, alpha=0.5)

ax.set_title("Sprint 6 — 6-metric radar (5 configs, 5-fold × 3-rep CV)",
             fontsize=13, pad=18)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18),
          fontsize=9, ncol=1, frameon=False)
fig.tight_layout()
fig.savefig(HERE / "sprint6_radar.png", bbox_inches="tight")
plt.close(fig)

print("wrote:")
print(" ", HERE / "sprint6_auc_bar.png")
print(" ", HERE / "sprint6_lr_sensitivity.png")
print(" ", HERE / "sprint6_radar.png")
