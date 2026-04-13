"""Generate radar chart + Excel summary from sprint2_results.json."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

SRC = Path(__file__).parent / "outputs" / "sprint2_results.json"
OUT_DIR = Path(r"C:\Users\cheng\Desktop\图卷积-肺小血管演化规律探索\outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["Accuracy", "AUC", "F1", "Specificity", "Sensitivity", "Precision"]
METRIC_LABELS = ["ACC", "AUC", "F1", "Specificity", "Sensitivity", "Precision"]
MODES = ["radiomics_only", "gcn_only", "hybrid"]
FEATS = ["baseline", "enhanced"]


def load() -> dict:
    with open(SRC, "r", encoding="utf-8") as f:
        return json.load(f)


def build_excel(data: dict, path: Path) -> None:
    per_fold_rows = []
    summary_rows = []
    for feat in FEATS:
        for mode in MODES:
            block = data[feat][mode]
            for i, fold in enumerate(block["folds"], 1):
                row = {"feat_set": feat, "mode": mode, "fold": i}
                row.update({m: fold[m] for m in METRICS})
                per_fold_rows.append(row)
            mean_row = {"feat_set": feat, "mode": mode, "stat": "mean"}
            std_row = {"feat_set": feat, "mode": mode, "stat": "std"}
            mean_row.update({m: block["mean"][m] for m in METRICS})
            std_row.update({m: block["std"][m] for m in METRICS})
            mean_row["pooled_AUC"] = block.get("pooled_AUC")
            summary_rows.append(mean_row)
            summary_rows.append(std_row)

    df_folds = pd.DataFrame(per_fold_rows)
    df_sum = pd.DataFrame(summary_rows)

    pretty_rows = []
    for feat in FEATS:
        for mode in MODES:
            b = data[feat][mode]
            row = {"feat_set": feat, "mode": mode}
            for m in METRICS:
                row[m] = f"{b['mean'][m]:.4f}±{b['std'][m]:.4f}"
            row["pooled_AUC"] = f"{b.get('pooled_AUC', float('nan')):.4f}"
            pretty_rows.append(row)
    df_pretty = pd.DataFrame(pretty_rows)

    with pd.ExcelWriter(path, engine="openpyxl") as w:
        df_pretty.to_excel(w, sheet_name="summary_pretty", index=False)
        df_sum.to_excel(w, sheet_name="mean_std", index=False)
        df_folds.to_excel(w, sheet_name="per_fold", index=False)


def radar_axes(ax, n):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(METRIC_LABELS, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="gray")
    ax.grid(True, linestyle=":", alpha=0.6)
    return angles


def plot_one(ax, data, feat, title):
    angles = radar_axes(ax, len(METRICS))
    ref = [0.9] * len(METRICS) + [0.9]
    ax.plot(angles, ref, color="gray", linestyle=":", linewidth=0.9, alpha=0.7, zorder=2)
    ax.text(np.deg2rad(8), 0.9, "0.9", fontsize=8, color="gray",
            ha="left", va="center")

    colors = {"radiomics_only": "#1f77b4", "gcn_only": "#2ca02c", "hybrid": "#d62728"}
    for mode in MODES:
        mean = data[feat][mode]["mean"]
        vals = [mean[m] for m in METRICS]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=mode, color=colors[mode])
        ax.fill(angles, vals, alpha=0.12, color=colors[mode])
    ax.set_title(title, fontsize=12, pad=18, fontweight="bold")


def plot_radar(data: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw=dict(polar=True))
    plot_one(axes[0], data, "baseline", "Baseline (12D node features)")
    plot_one(axes[1], data, "enhanced", "Enhanced (16D, +4 vessel features)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Sprint 2 — 5-fold CV mean metrics", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_radar_combined(data: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    angles = radar_axes(ax, len(METRICS))
    ref = [0.9] * len(METRICS) + [0.9]
    ax.plot(angles, ref, color="gray", linestyle=":", linewidth=0.9, alpha=0.7, zorder=2)
    ax.text(np.deg2rad(8), 0.9, "0.9", fontsize=8, color="gray",
            ha="left", va="center")

    palette = {
        ("baseline", "radiomics_only"): ("#1f77b4", "-"),
        ("baseline", "gcn_only"):       ("#2ca02c", "-"),
        ("baseline", "hybrid"):         ("#d62728", "-"),
        ("enhanced", "radiomics_only"): ("#1f77b4", "--"),
        ("enhanced", "gcn_only"):       ("#2ca02c", "--"),
        ("enhanced", "hybrid"):         ("#d62728", "--"),
    }
    for feat in FEATS:
        for mode in MODES:
            mean = data[feat][mode]["mean"]
            vals = [mean[m] for m in METRICS]
            vals += vals[:1]
            color, ls = palette[(feat, mode)]
            ax.plot(angles, vals, linewidth=1.8, label=f"{feat}/{mode}", color=color, linestyle=ls)
    ax.set_title("Sprint 2 — All configs (solid=baseline / dashed=enhanced)",
                 fontsize=12, pad=22, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05), fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data = load()
    xlsx = OUT_DIR / "sprint2_metrics.xlsx"
    png_split = OUT_DIR / "sprint2_radar.png"
    png_combined = OUT_DIR / "sprint2_radar_combined.png"
    build_excel(data, xlsx)
    plot_radar(data, png_split)
    plot_radar_combined(data, png_combined)
    print(f"[ok] {xlsx}")
    print(f"[ok] {png_split}")
    print(f"[ok] {png_combined}")


if __name__ == "__main__":
    main()
