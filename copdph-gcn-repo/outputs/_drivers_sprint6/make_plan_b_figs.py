"""Plan B figures — per-tertile A:B diameter ratio PH vs non-PH."""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(r"E:\桌面文件\图卷积-肺小血管演化规律探索\outputs\p_zeta_cluster_269\plan_b")
PER_CASE = ROOT / "plan_b_per_case.csv"
OUT = ROOT

rows = []
with open(PER_CASE, newline="") as f:
    for r in csv.DictReader(f):
        rows.append(r)

def col(key, mask=None):
    vals = []
    for r in rows:
        v = r[key]
        if v == "" or v.lower() == "nan":
            vals.append(np.nan)
        else:
            vals.append(float(v))
    vals = np.array(vals, dtype=float)
    if mask is not None:
        vals = vals[mask]
    return vals

labels = col("label")
ph = labels == 1
nonph = labels == 0

# Panel 1: fraction(A:B > 1) by tertile, PH vs non-PH
tertiles = ["upper", "middle", "lower"]
ph_means = []; ph_stds = []
non_means = []; non_stds = []
pvals_mw = []
from scipy.stats import mannwhitneyu
for t in tertiles:
    v = col(f"{t}_frac_gt1")
    vp = v[ph]; vp = vp[~np.isnan(vp)]
    vn = v[nonph]; vn = vn[~np.isnan(vn)]
    ph_means.append(vp.mean()); ph_stds.append(vp.std())
    non_means.append(vn.mean()); non_stds.append(vn.std())
    try:
        _, p = mannwhitneyu(vp, vn, alternative="two-sided")
    except Exception:
        p = float("nan")
    pvals_mw.append(p)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
x = np.arange(len(tertiles))
w = 0.36
ax.bar(x - w/2, ph_means, w, yerr=ph_stds, capsize=3, color="#d62728",
       label=f"PH (n={int(ph.sum())})", alpha=0.85)
ax.bar(x + w/2, non_means, w, yerr=non_stds, capsize=3, color="#2ca02c",
       label=f"non-PH (n={int(nonph.sum())})", alpha=0.85)
for i, p in enumerate(pvals_mw):
    if p < 0.05:
        mark = "*"
    elif p < 0.1:
        mark = "·"
    else:
        mark = ""
    y = max(ph_means[i] + ph_stds[i], non_means[i] + non_stds[i]) + 0.03
    ax.text(x[i], y, f"p={p:.3f}{mark}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(["upper Z", "middle Z", "lower Z"])
ax.set_ylabel("fraction of paired branches with A:B > 1")
ax.set_title("Plan B — cross-structure A:B ratio by Z-tertile\n(lower Z-tertile trends toward more A>B in PH)")
ax.legend(loc="upper right")
ax.set_ylim(0, 1.0)
ax.grid(axis="y", alpha=0.3)

# Panel 2: mean A:B ratio by tertile
ph_m = []; ph_s = []
non_m = []; non_s = []
pvals_m = []
for t in tertiles:
    v = col(f"{t}_AB_mean")
    vp = v[ph]; vp = vp[~np.isnan(vp)]
    vn = v[nonph]; vn = vn[~np.isnan(vn)]
    ph_m.append(vp.mean()); ph_s.append(vp.std())
    non_m.append(vn.mean()); non_s.append(vn.std())
    try:
        _, p = mannwhitneyu(vp, vn, alternative="two-sided")
    except Exception:
        p = float("nan")
    pvals_m.append(p)

ax = axes[1]
ax.bar(x - w/2, ph_m, w, yerr=ph_s, capsize=3, color="#d62728",
       label=f"PH (n={int(ph.sum())})", alpha=0.85)
ax.bar(x + w/2, non_m, w, yerr=non_s, capsize=3, color="#2ca02c",
       label=f"non-PH (n={int(nonph.sum())})", alpha=0.85)
ax.axhline(1.0, ls="--", color="k", alpha=0.5, lw=1)
for i, p in enumerate(pvals_m):
    mark = "*" if p < 0.05 else ("·" if p < 0.1 else "")
    y = max(ph_m[i] + ph_s[i], non_m[i] + non_s[i]) + 0.05
    ax.text(x[i], y, f"p={p:.2f}{mark}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(["upper Z", "middle Z", "lower Z"])
ax.set_ylabel("mean artery:airway diameter ratio")
ax.set_title("Plan B — mean A:B ratio by Z-tertile")
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
out = OUT / "plan_b_ab_ratio_by_tertile.png"
plt.savefig(out, dpi=140)
print(f"wrote {out}")
