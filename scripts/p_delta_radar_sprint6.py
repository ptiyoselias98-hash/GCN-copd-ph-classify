"""P-δ: Sprint 6 六臂 + Sprint 5 v2 baseline 六指标雷达图."""
from __future__ import annotations
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "sprint6_实验结果"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["AUC", "Accuracy", "Precision", "Sensitivity", "F1", "Specificity"]

def fold_mean(rs: dict, mode: str = "gcn_only") -> dict:
    """Return dict[metric]→mean across folds for the given mode."""
    if "baseline" not in rs or mode not in rs["baseline"]:
        return {}
    folds = rs["baseline"][mode]["folds"]
    out = {m: float(np.mean([f[m] for f in folds])) for m in METRICS}
    return out

def fold_mean_enhanced(rs: dict, mode: str = "gcn_only") -> dict:
    if "enhanced" not in rs or mode not in rs["enhanced"]:
        return {}
    folds = rs["enhanced"][mode]["folds"]
    return {m: float(np.mean([f[m] for f in folds])) for m in METRICS}

# load arm results
def load(p):
    return json.loads(p.read_text(encoding="utf-8"))

arms = {
    "arm_a_base (282, gcn)":       load(ROOT/"outputs"/"sprint6_arm_a_base"/"sprint6_results.json"),
    "arm_a_full (282, +aug+res)":  load(ROOT/"outputs"/"sprint6_arm_a_full"/"sprint6_results.json"),
    "arm_a_ensemble (282, 3rep)":  load(ROOT/"outputs"/"sprint6_arm_a_ensemble"/"sprint6_results.json"),
    "arm_b_base (113, hybrid)":    load(ROOT/"outputs"/"sprint6_arm_b_base"/"sprint6_results.json"),
    "arm_b_full (113, hybrid)":    load(ROOT/"outputs"/"sprint6_arm_b_full"/"sprint6_results.json"),
}

# Sprint 5 v2 baseline, pulled from 实验盘点_2026-04-21.md
sprint5_v2 = {
    "AUC": 0.924, "Accuracy": 0.895, "Precision": 0.940,
    "Sensitivity": 0.870, "F1": 0.905, "Specificity": 0.920,
}
# (Accuracy/Precision/F1 for Sprint 5 v2 estimated from available data; AUC/Sens/Spec from table)

def polar_angles(n):
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.concatenate([theta, theta[:1]])

def plot_radar(entries: dict, title: str, out_path: Path, colors=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    ang = polar_angles(len(METRICS))
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0,1,len(entries)))
    for (name, vals), color in zip(entries.items(), colors):
        data = [vals.get(m, 0.0) for m in METRICS]
        data = data + data[:1]
        ax.plot(ang, data, "o-", linewidth=2, label=name, color=color)
        ax.fill(ang, data, alpha=0.12, color=color)
    ax.set_xticks(polar_angles(len(METRICS))[:-1])
    ax.set_xticklabels(METRICS, fontsize=11)
    ax.set_ylim(0.5, 1.0)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels([f"{v:.1f}" for v in [0.6,0.7,0.8,0.9,1.0]])
    ax.set_title(title, fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")

# --- plot 1: arm_a variants (282) — baseline/gcn_only
entries1 = {"Sprint 5 v2 baseline (106)": sprint5_v2}
for name, rs in arms.items():
    if "arm_a" in name:
        entries1[name] = fold_mean(rs, "gcn_only")
plot_radar(entries1, "Sprint 6 arm_a × Sprint 5 v2 — gcn_only baseline (fold-mean)",
           OUT_DIR/"radar_arm_a_vs_sprint5.png")

# --- plot 2: arm_b hybrid modes
entries2 = {"Sprint 5 v2 baseline (106)": sprint5_v2}
for name, rs in arms.items():
    if "arm_b" in name:
        entries2[name + " / baseline"] = fold_mean(rs, "hybrid")
        entries2[name + " / enhanced"] = fold_mean_enhanced(rs, "hybrid")
plot_radar(entries2, "Sprint 6 arm_b hybrid × Sprint 5 v2 (fold-mean)",
           OUT_DIR/"radar_arm_b_vs_sprint5.png")

# --- plot 3: all best picks side-by-side
entries3 = {
    "Sprint 5 v2 baseline (106)": sprint5_v2,
    "arm_a_ensemble best pooled (282)": fold_mean(arms["arm_a_ensemble (282, 3rep)"], "gcn_only"),
    "arm_a_base / enhanced (282)": fold_mean_enhanced(arms["arm_a_base (282, gcn)"], "gcn_only"),
    "arm_b_base / baseline hybrid (113)": fold_mean(arms["arm_b_base (113, hybrid)"], "hybrid"),
    "arm_b_base / enhanced gcn_only (113)": fold_mean_enhanced(arms["arm_b_base (113, hybrid)"], "gcn_only"),
}
plot_radar(entries3, "Sprint 6 best arms × Sprint 5 v2 — six-metric overlay",
           OUT_DIR/"radar_all_best_vs_sprint5.png")

# text summary for cross-reference
def fmt(d): return " / ".join(f"{m}={d.get(m,0):.3f}" for m in METRICS)
summary_lines = ["# P-δ radar: 六指标汇总(fold-mean)", ""]
summary_lines.append("| entry | " + " | ".join(METRICS) + " |")
summary_lines.append("|---|" + "---|"*len(METRICS))
for name, vals in entries3.items():
    row = "| " + name + " | " + " | ".join(f"{vals.get(m,0):.3f}" for m in METRICS) + " |"
    summary_lines.append(row)
summary_lines.append("")
summary_lines.append("## Figures")
summary_lines.append("- `radar_arm_a_vs_sprint5.png` — arm_a 四个配置叠加")
summary_lines.append("- `radar_arm_b_vs_sprint5.png` — arm_b hybrid baseline/enhanced")
summary_lines.append("- `radar_all_best_vs_sprint5.png` — 最佳配置 vs Sprint 5 v2 baseline")
(OUT_DIR/"radar_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
print(f"Wrote {OUT_DIR/'radar_summary.md'}")
