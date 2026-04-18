"""Generate 6-metric xlsx + radar PNGs from sprint2_results.json."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
data = json.loads((HERE / "sprint2_results.json").read_text())
KEYS = ["AUC", "Accuracy", "Precision", "Sensitivity", "F1", "Specificity"]

rows = []
for fs, modes in data.items():
    for mode, r in modes.items():
        row = {"feat_set": fs, "mode": mode, "pooled_AUC": r["pooled_AUC"]}
        for k in KEYS:
            row[f"{k}_mean"] = r["mean"][k]
            row[f"{k}_std"] = r["std"][k]
        rows.append(row)
df = pd.DataFrame(rows)
df.to_excel(HERE / "sprint2_metrics.xlsx", index=False)
print(f"saved {HERE / 'sprint2_metrics.xlsx'}")

# Radar: one per mode, comparing baseline vs enhanced
angles = np.linspace(0, 2 * np.pi, len(KEYS), endpoint=False).tolist()
angles += angles[:1]

fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))
for ax, mode in zip(axes, ["radiomics_only", "gcn_only", "hybrid"]):
    for fs, color in [("baseline", "#1f77b4"), ("enhanced", "#d62728")]:
        vals = [data[fs][mode]["mean"][k] for k in KEYS]
        vals += vals[:1]
        ax.plot(angles, vals, color=color, linewidth=2, label=fs)
        ax.fill(angles, vals, color=color, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(KEYS, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title(mode, y=1.08, fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
plt.tight_layout()
plt.savefig(HERE / "sprint2_radar.png", dpi=150, bbox_inches="tight")
print(f"saved {HERE / 'sprint2_radar.png'}")

# Combined bar chart of AUC for quick glance
fig, ax = plt.subplots(figsize=(10, 5))
modes = ["radiomics_only", "gcn_only", "hybrid"]
x = np.arange(len(modes))
w = 0.35
for i, fs in enumerate(["baseline", "enhanced"]):
    means = [data[fs][m]["mean"]["AUC"] for m in modes]
    stds = [data[fs][m]["std"]["AUC"] for m in modes]
    ax.bar(x + (i - 0.5) * w, means, w, yerr=stds, capsize=4, label=fs)
ax.set_xticks(x); ax.set_xticklabels(modes)
ax.set_ylabel("AUC"); ax.set_ylim(0, 1); ax.legend()
ax.set_title("Sprint2 v2: baseline vs enhanced (node 13D + graph-level 12D globals)")
plt.tight_layout()
plt.savefig(HERE / "sprint2_auc_bar.png", dpi=150, bbox_inches="tight")
print(f"saved {HERE / 'sprint2_auc_bar.png'}")
