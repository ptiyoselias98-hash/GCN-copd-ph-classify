"""Plan B figures — per-tertile A:B diameter ratio, PH vs non-PH."""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

ROOT = Path(r"E:\桌面文件\图卷积-肺小血管演化规律探索\outputs\p_zeta_cluster_269\plan_b")
PER_CASE = ROOT / "plan_b_per_case.csv"
OUT = ROOT

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
C_PH    = "#C0392B"
C_NONPH = "#27AE60"

rows = []
with open(PER_CASE, newline="") as f:
    for r in csv.DictReader(f):
        rows.append(r)

def col(key, mask=None):
    vals = []
    for r in rows:
        v = r[key]
        vals.append(float(v) if v not in ("", "nan", "NaN") else np.nan)
    vals = np.array(vals, dtype=float)
    return vals if mask is None else vals[mask]

labels = col("label")
ph = labels == 1
nonph = labels == 0

def _stats(key):
    v = col(key)
    vp = v[ph]; vp = vp[~np.isnan(vp)]
    vn = v[nonph]; vn = vn[~np.isnan(vn)]
    try:
        _, p = mannwhitneyu(vp, vn, alternative="two-sided")
    except Exception:
        p = float("nan")
    return vp.mean(), vp.std(), vn.mean(), vn.std(), p

def _yerr_clip(means, stds, floor=0.0):
    """Asymmetric yerr that doesn't dip below `floor`."""
    means = np.asarray(means); stds = np.asarray(stds)
    lower = np.minimum(stds, means - floor)
    return np.vstack([np.maximum(lower, 0), stds])

tertiles = ["upper", "middle", "lower"]
xticks   = ["upper Z", "middle Z", "lower Z"]
x = np.arange(len(tertiles))
w = 0.36

ph_n  = int(ph.sum()); nph_n = int(nonph.sum())

# ---- aggregate stats
frac_ph_m, frac_ph_s, frac_nph_m, frac_nph_s, frac_p = [], [], [], [], []
for t in tertiles:
    m_p, s_p, m_n, s_n, p = _stats(f"{t}_frac_gt1")
    frac_ph_m.append(m_p); frac_ph_s.append(s_p)
    frac_nph_m.append(m_n); frac_nph_s.append(s_n); frac_p.append(p)

mean_ph_m, mean_ph_s, mean_nph_m, mean_nph_s, mean_p = [], [], [], [], []
for t in tertiles:
    m_p, s_p, m_n, s_n, p = _stats(f"{t}_AB_mean")
    mean_ph_m.append(m_p); mean_ph_s.append(s_p)
    mean_nph_m.append(m_n); mean_nph_s.append(s_n); mean_p.append(p)

fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))

# Panel A — fraction(A:B > 1)
ax = axes[0]
ax.bar(x - w/2, frac_ph_m, w,
       yerr=_yerr_clip(frac_ph_m, frac_ph_s, 0.0),
       capsize=3, color=C_PH,    edgecolor="white", linewidth=0.6, alpha=0.92,
       label=f"PH (n={ph_n})", error_kw=dict(lw=1.0, ecolor="#222"))
ax.bar(x + w/2, frac_nph_m, w,
       yerr=_yerr_clip(frac_nph_m, frac_nph_s, 0.0),
       capsize=3, color=C_NONPH, edgecolor="white", linewidth=0.6, alpha=0.92,
       label=f"non-PH (n={nph_n})", error_kw=dict(lw=1.0, ecolor="#222"))

# value labels above the bars (interior of error caps)
for i in range(len(tertiles)):
    ax.text(x[i] - w/2, frac_ph_m[i]  + 0.012, f"{frac_ph_m[i]:.2f}",
            ha="center", va="bottom", fontsize=8.5, color=C_PH, weight="semibold")
    ax.text(x[i] + w/2, frac_nph_m[i] + 0.012, f"{frac_nph_m[i]:.2f}",
            ha="center", va="bottom", fontsize=8.5, color=C_NONPH, weight="semibold")

# p-value bracket above each tertile
ax.set_ylim(0, 1.10)
for i, p in enumerate(frac_p):
    top = max(frac_ph_m[i] + frac_ph_s[i], frac_nph_m[i] + frac_nph_s[i])
    y_brk = min(top + 0.04, 1.02)
    ax.plot([x[i] - w/2, x[i] + w/2], [y_brk, y_brk], color="#444", lw=0.9)
    mark = "*" if p < 0.05 else ("·" if p < 0.10 else "")
    ax.text(x[i], y_brk + 0.012, f"p = {p:.3f}{mark}",
            ha="center", va="bottom", fontsize=9, color="#222")

ax.set_xticks(x); ax.set_xticklabels(xticks)
ax.set_ylabel("fraction of paired branches with A:B > 1")
ax.set_title("Plan B — fraction(A:B > 1) by Z-tertile\n(lower-Z tertile trends toward more A > B in PH)",
             fontsize=11.5)
ax.grid(axis="y", linestyle=":", color="#aaa", alpha=0.6)
ax.set_axisbelow(True)

# Panel B — mean A:B ratio
ax = axes[1]
ax.bar(x - w/2, mean_ph_m, w,
       yerr=_yerr_clip(mean_ph_m, mean_ph_s, 0.0),
       capsize=3, color=C_PH,    edgecolor="white", linewidth=0.6, alpha=0.92,
       label=f"PH (n={ph_n})", error_kw=dict(lw=1.0, ecolor="#222"))
ax.bar(x + w/2, mean_nph_m, w,
       yerr=_yerr_clip(mean_nph_m, mean_nph_s, 0.0),
       capsize=3, color=C_NONPH, edgecolor="white", linewidth=0.6, alpha=0.92,
       label=f"non-PH (n={nph_n})", error_kw=dict(lw=1.0, ecolor="#222"))
ax.axhline(1.0, ls="--", color="#444", alpha=0.6, lw=1, label="A:B = 1 (equal)")
for i in range(len(tertiles)):
    ax.text(x[i] - w/2, mean_ph_m[i]  + 0.04, f"{mean_ph_m[i]:.2f}",
            ha="center", va="bottom", fontsize=8.5, color=C_PH, weight="semibold")
    ax.text(x[i] + w/2, mean_nph_m[i] + 0.04, f"{mean_nph_m[i]:.2f}",
            ha="center", va="bottom", fontsize=8.5, color=C_NONPH, weight="semibold")

# y-lim chosen so labels and brackets fit
ymax = max(np.array(mean_ph_m) + np.array(mean_ph_s)).max() if len(mean_ph_m) else 2
ymax = max(ymax, max(np.array(mean_nph_m) + np.array(mean_nph_s)).max() if len(mean_nph_m) else 2)
ax.set_ylim(0, ymax + 0.45)

for i, p in enumerate(mean_p):
    top = max(mean_ph_m[i] + mean_ph_s[i], mean_nph_m[i] + mean_nph_s[i])
    y_brk = top + 0.10
    ax.plot([x[i] - w/2, x[i] + w/2], [y_brk, y_brk], color="#444", lw=0.9)
    mark = "*" if p < 0.05 else ("·" if p < 0.10 else "")
    ax.text(x[i], y_brk + 0.04, f"p = {p:.2f}{mark}",
            ha="center", va="bottom", fontsize=9, color="#222")

ax.set_xticks(x); ax.set_xticklabels(xticks)
ax.set_ylabel("mean artery : airway diameter ratio")
ax.set_title("Plan B — mean A:B ratio by Z-tertile\n(bulk centred near 1.1; signal lives in the tail)",
             fontsize=11.5)
ax.grid(axis="y", linestyle=":", color="#aaa", alpha=0.6)
ax.set_axisbelow(True)

# single shared figure legend below both panels
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
fig.legend(handles=[
    Patch(facecolor=C_PH,    edgecolor="white", label=f"PH  (n = {ph_n})"),
    Patch(facecolor=C_NONPH, edgecolor="white", label=f"non-PH  (n = {nph_n})"),
    Line2D([0], [0], color="#444", ls="--", lw=1, label="A:B = 1 (equal)"),
], loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False,
   fontsize=10)

plt.tight_layout(rect=[0, 0.06, 1, 1])
out = OUT / "plan_b_ab_ratio_by_tertile.png"
plt.savefig(out)
print(f"wrote {out}")
