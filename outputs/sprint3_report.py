"""Generate sprint3 reports: per-arm xlsx + radar, plus combined comparison bar.

Also compares against sprint2_v2 to visualize the P0 improvement delta.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).parent
ARMS = ["focal_local4", "focal_all", "wce_local4"]
MODES = ["radiomics_only", "gcn_only", "hybrid"]
FEATS = ["baseline", "enhanced"]
KEYS = ["AUC", "Accuracy", "Precision", "Sensitivity", "F1", "Specificity"]


def load(arm: str) -> dict:
    return json.loads((HERE / f"sprint3_{arm}" / "sprint3_results.json").read_text())


def strip_cfg(data: dict) -> dict:
    return {k: v for k, v in data.items() if k != "_config"}


def per_arm_xlsx(arm: str, data: dict) -> Path:
    rows = []
    for fs in FEATS:
        for mode in MODES:
            r = data[fs][mode]
            row = {"feat_set": fs, "mode": mode, "pooled_AUC": r["pooled_AUC"]}
            for k in KEYS:
                row[f"{k}_mean"] = r["mean"][k]
                row[f"{k}_std"] = r["std"][k]
            # Youden thresholds per fold
            row["threshold_mean"] = float(np.mean([f["threshold"] for f in r["folds"]]))
            rows.append(row)
    df = pd.DataFrame(rows)
    out = HERE / f"sprint3_{arm}" / "sprint3_metrics.xlsx"
    df.to_excel(out, index=False)
    return out


def per_arm_radar(arm: str, data: dict) -> Path:
    angles = np.linspace(0, 2 * np.pi, len(KEYS), endpoint=False).tolist() + [0.0]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))
    for ax, mode in zip(axes, MODES):
        for fs, color in [("baseline", "#1f77b4"), ("enhanced", "#d62728")]:
            vals = [data[fs][mode]["mean"][k] for k in KEYS]
            vals += vals[:1]
            ax.plot(angles, vals, color=color, linewidth=2, label=fs)
            ax.fill(angles, vals, color=color, alpha=0.15)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(KEYS, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 0.9, 1.0], angle=30, fontsize=7)
        ax.grid(True, color="gray", alpha=0.55, linewidth=0.7)
        ax.set_title(mode, y=1.08, fontsize=12)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    fig.suptitle(f"Sprint 3 — {arm}", fontsize=14, y=1.02)
    plt.tight_layout()
    out = HERE / f"sprint3_{arm}" / "sprint3_radar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    return out


def combined_bar(arms_data: dict, sprint2: dict) -> Path:
    """Compare enhanced/hybrid across arms + sprint2 baseline/hybrid + enhanced/hybrid."""
    labels = KEYS
    x = np.arange(len(labels))
    series = [
        ("sprint2 base/hybrid", sprint2["baseline"]["hybrid"]["mean"], "#888"),
        ("sprint2 enh/hybrid",  sprint2["enhanced"]["hybrid"]["mean"], "#bbb"),
    ]
    palette = ["#1f77b4", "#2ca02c", "#d62728"]
    for arm, color in zip(ARMS, palette):
        series.append((f"sp3 {arm} enh/hyb",
                       arms_data[arm]["enhanced"]["hybrid"]["mean"], color))
    w = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (name, m, color) in enumerate(series):
        vals = [m[k] for k in labels]
        ax.bar(x + (i - (len(series) - 1) / 2) * w, vals, w, label=name, color=color)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.05)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0])
    ax.axhline(0.9, color="gray", linewidth=0.8, alpha=0.55, zorder=0)
    ax.set_ylabel("Score")
    ax.set_title("Sprint 3 P0 vs Sprint 2 v2 — enhanced/hybrid across arms "
                 "(all 6 metrics, 5-fold CV mean)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = HERE / "sprint3_combined_bar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    return out


def combined_radar(arms_data: dict) -> Path:
    """6-panel radar: (arm × mode) for enhanced feat set only, to show which
    mode + arm wins where."""
    angles = np.linspace(0, 2 * np.pi, len(KEYS), endpoint=False).tolist() + [0.0]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))
    palette = {"focal_local4": "#1f77b4", "focal_all": "#2ca02c", "wce_local4": "#d62728"}
    for ax, mode in zip(axes, MODES):
        for arm in ARMS:
            vals = [arms_data[arm]["enhanced"][mode]["mean"][k] for k in KEYS]
            vals += vals[:1]
            ax.plot(angles, vals, color=palette[arm], linewidth=2, label=arm)
            ax.fill(angles, vals, color=palette[arm], alpha=0.12)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(KEYS, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 0.9, 1.0], angle=30, fontsize=7)
        ax.grid(True, color="gray", alpha=0.55, linewidth=0.7)
        ax.set_title(f"enhanced / {mode}", y=1.08, fontsize=12)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    fig.suptitle("Sprint 3 — enhanced feature set, 3 arms compared", fontsize=14, y=1.02)
    plt.tight_layout()
    out = HERE / "sprint3_arms_radar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    return out


def summary_table(arms_data: dict, sprint2: dict) -> Path:
    rows = []
    for name, d in [("sprint2_v2", sprint2)]:
        for fs in FEATS:
            for mode in MODES:
                r = d[fs][mode]
                row = {"run": name, "feat_set": fs, "mode": mode,
                       "pooled_AUC": r["pooled_AUC"]}
                for k in KEYS:
                    row[k] = r["mean"][k]
                rows.append(row)
    for arm, d in arms_data.items():
        for fs in FEATS:
            for mode in MODES:
                r = d[fs][mode]
                row = {"run": f"sp3_{arm}", "feat_set": fs, "mode": mode,
                       "pooled_AUC": r["pooled_AUC"]}
                for k in KEYS:
                    row[k] = r["mean"][k]
                rows.append(row)
    df = pd.DataFrame(rows)
    out = HERE / "sprint3_vs_sprint2.xlsx"
    df.to_excel(out, index=False)
    return out


def main():
    arms_data = {a: strip_cfg(load(a)) for a in ARMS}
    sprint2 = json.loads((HERE / "sprint2_v2" / "sprint2_results.json").read_text())

    for arm in ARMS:
        print("xlsx:", per_arm_xlsx(arm, arms_data[arm]))
        print("radar:", per_arm_radar(arm, arms_data[arm]))
    print("combined bar:", combined_bar(arms_data, sprint2))
    print("arms radar:", combined_radar(arms_data))
    print("summary xlsx:", summary_table(arms_data, sprint2))

    # console summary
    print("\n" + "=" * 130)
    print(f"{'run':<20} {'feat':<10} {'mode':<16} " + " ".join(f"{k:>10}" for k in KEYS) + "  pooled")
    print("-" * 130)
    for name, d in [("sprint2_v2", sprint2)] + [(f"sp3_{a}", arms_data[a]) for a in ARMS]:
        for fs in FEATS:
            for mode in MODES:
                r = d[fs][mode]
                m = r["mean"]
                print(f"{name:<20} {fs:<10} {mode:<16} "
                      + " ".join(f"{m[k]:>10.4f}" for k in KEYS)
                      + f"  {r['pooled_AUC']:.4f}")


if __name__ == "__main__":
    main()
