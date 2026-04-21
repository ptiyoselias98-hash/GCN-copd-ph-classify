"""Plan C figure — separate-structure-with-attention (p_theta_269_lr2x) vs
joint heterograph with companion edges (Plan C). Both n=269, 5-fold CV."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT    = Path(r"E:\桌面文件\图卷积-肺小血管演化规律探索\outputs\p_zeta_cluster_269")
P_THETA = ROOT / "p_theta_269_lr2x_cv_results.json"
PLAN_C  = ROOT / "plan_c" / "cv_results.json"
OUT     = ROOT / "plan_c"

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
C_THETA = "#1F77B4"   # blue
C_C     = "#9467BD"   # purple

p_theta = json.load(open(P_THETA))["mean_metrics"]
plan_c  = json.load(open(PLAN_C))["aggregate"]

metrics = ["auc", "accuracy", "sensitivity", "specificity", "f1", "precision"]
labels  = ["AUC", "Accuracy", "Sensitivity", "Specificity", "F1", "Precision"]
pt_means = np.array([p_theta[m]["mean"] for m in metrics])
pt_stds  = np.array([p_theta[m]["std"]  for m in metrics])
pc_means = np.array([plan_c[m]["mean"]  for m in metrics])
pc_stds  = np.array([plan_c[m]["std"]   for m in metrics])

fig, ax = plt.subplots(figsize=(10.0, 5.0))
x = np.arange(len(metrics))
w = 0.36

ax.bar(x - w/2, pt_means, w, yerr=pt_stds, capsize=3,
       color=C_THETA, edgecolor="white", linewidth=0.8, alpha=0.92,
       error_kw=dict(lw=1.0, ecolor="#222"),
       label="p_theta_269_lr2x  —  separate towers + cross-attention")
ax.bar(x + w/2, pc_means, w, yerr=pc_stds, capsize=3,
       color=C_C,     edgecolor="white", linewidth=0.8, alpha=0.92,
       error_kw=dict(lw=1.0, ecolor="#222"),
       label="Plan C  —  joint heterograph (companion edges)")

# value labels above the error caps
for i in range(len(metrics)):
    top_t = pt_means[i] + pt_stds[i]
    top_c = pc_means[i] + pc_stds[i]
    ax.text(x[i] - w/2, top_t + 0.014, f"{pt_means[i]:.3f}",
            ha="center", va="bottom", fontsize=8.5,
            color=C_THETA, weight="semibold")
    ax.text(x[i] + w/2, top_c + 0.014, f"{pc_means[i]:.3f}",
            ha="center", va="bottom", fontsize=8.5,
            color=C_C, weight="semibold")
    # delta annotation under the metric label
    delta = pc_means[i] - pt_means[i]
    color_d = "#C0392B" if delta < 0 else "#27AE60"
    ax.text(x[i], 0.555, f"Δ {delta:+.3f}", ha="center", va="bottom",
            fontsize=8.5, color=color_d)

ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0.55, 1.08)
ax.set_ylabel("metric value  (5-fold CV mean ± std)")
ax.set_title("Plan C  —  separate-structure cross-attention  vs  joint heterograph  (n = 269)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=1)
ax.grid(axis="y", linestyle=":", color="#aaa", alpha=0.6)
ax.set_axisbelow(True)

plt.tight_layout()
out = OUT / "plan_c_vs_p_theta.png"
plt.savefig(out)
print(f"wrote {out}")

print()
print(f"{'metric':12s} {'p_theta':18s} {'plan C':18s} {'Δ (C - theta)':>14s}")
for m, l, a, sa, b, sb in zip(metrics, labels, pt_means, pt_stds, pc_means, pc_stds):
    print(f"{l:12s} {a:.3f}±{sa:.3f}        {b:.3f}±{sb:.3f}        {b-a:+.3f}")
