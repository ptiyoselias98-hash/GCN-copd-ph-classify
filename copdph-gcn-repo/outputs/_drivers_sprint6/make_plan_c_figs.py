"""Plan C figure — separate-structure-with-attention (p_theta_269_lr2x) vs
joint heterograph with companion edges (Plan C). Both n=269, 5-fold CV."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(r"E:\桌面文件\图卷积-肺小血管演化规律探索\outputs\p_zeta_cluster_269")
P_THETA = ROOT / "p_theta_269_lr2x_cv_results.json"
PLAN_C  = ROOT / "plan_c" / "cv_results.json"
OUT     = ROOT / "plan_c"

p_theta = json.load(open(P_THETA))["mean_metrics"]
plan_c  = json.load(open(PLAN_C))["aggregate"]

metrics = ["auc", "accuracy", "sensitivity", "specificity", "f1", "precision"]
labels  = ["AUC", "Acc", "Sens", "Spec", "F1", "Prec"]
pt_means = [p_theta[m]["mean"] for m in metrics]
pt_stds  = [p_theta[m]["std"]  for m in metrics]
pc_means = [plan_c[m]["mean"]  for m in metrics]
pc_stds  = [plan_c[m]["std"]   for m in metrics]

fig, ax = plt.subplots(figsize=(8.5, 4.4))
x = np.arange(len(metrics))
w = 0.36
ax.bar(x - w/2, pt_means, w, yerr=pt_stds, capsize=3, color="#1f77b4",
       label="p_theta_269_lr2x — separate + cross-attention", alpha=0.9)
ax.bar(x + w/2, pc_means, w, yerr=pc_stds, capsize=3, color="#9467bd",
       label="Plan C — joint heterograph (companion edges)", alpha=0.9)
for i, (a, b) in enumerate(zip(pt_means, pc_means)):
    ax.text(x[i] - w/2, a + 0.018, f"{a:.3f}", ha="center", fontsize=8)
    ax.text(x[i] + w/2, b + 0.018, f"{b:.3f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0.55, 1.05)
ax.set_ylabel("metric value (5-fold CV mean ± std)")
ax.set_title("Plan C — separate-structure cross-attention vs joint heterograph (n=269)")
ax.legend(loc="lower right", fontsize=9)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
out = OUT / "plan_c_vs_p_theta.png"
plt.savefig(out, dpi=140)
print(f"wrote {out}")

# print delta
print()
print(f"{'metric':12s} {'p_theta':18s} {'plan C':18s} {'Δ (C - theta)':>14s}")
for m, l, a, sa, b, sb in zip(metrics, labels, pt_means, pt_stds, pc_means, pc_stds):
    print(f"{l:12s} {a:.3f}±{sa:.3f}        {b:.3f}±{sb:.3f}        {b-a:+.3f}")
