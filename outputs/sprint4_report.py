"""Sprint 4 report: gated fusion (4a) + A/V node flag (4b) vs sprint3 winner.

Emits:
    outputs/sprint4a_gated/sprint4_metrics.xlsx  + radar PNG
    outputs/sprint4b_av/   sprint4_metrics.xlsx  + radar PNG
    outputs/sprint4_combined_bar.png
    outputs/sprint4_arms_radar.png
    outputs/sprint4_vs_sprint3.xlsx
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).parent
ARMS = [("sprint3_focal_local4", "sp3 focal_local4"),
        ("sprint4a_gated",       "sp4a gated"),
        ("sprint4b_av",          "sp4b A/V")]
MODES = ["radiomics_only", "gcn_only", "hybrid"]
FEATS = ["baseline", "enhanced"]
KEYS = ["AUC", "Accuracy", "Precision", "Sensitivity", "F1", "Specificity"]


def load(sub: str) -> dict | None:
    p = HERE / sub / "sprint3_results.json"
    if not p.exists():
        print(f"MISS {p}")
        return None
    return {k: v for k, v in json.loads(p.read_text()).items() if k != "_config"}


def per_arm_xlsx(sub: str, data: dict) -> Path:
    rows = []
    for fs in FEATS:
        for mode in MODES:
            r = data[fs][mode]
            row = {"feat_set": fs, "mode": mode, "pooled_AUC": r["pooled_AUC"]}
            for k in KEYS:
                row[f"{k}_mean"] = r["mean"][k]
                row[f"{k}_std"] = r["std"][k]
            rows.append(row)
    df = pd.DataFrame(rows)
    out = HERE / sub / "sprint4_metrics.xlsx"
    df.to_excel(out, index=False)
    return out


def per_arm_radar(sub: str, data: dict) -> Path:
    angles = np.linspace(0, 2 * np.pi, len(KEYS), endpoint=False).tolist() + [0.0]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))
    for ax, mode in zip(axes, MODES):
        for fs, c in [("baseline", "#1f77b4"), ("enhanced", "#d62728")]:
            vals = [data[fs][mode]["mean"][k] for k in KEYS]
            vals += vals[:1]
            ax.plot(angles, vals, color=c, linewidth=2, label=fs)
            ax.fill(angles, vals, color=c, alpha=0.15)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(KEYS, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 0.9, 1.0], angle=30, fontsize=7)
        ax.grid(True, color="gray", alpha=0.55, linewidth=0.7)
        ax.set_title(mode, y=1.08, fontsize=12)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    fig.suptitle(f"Sprint 4 — {sub}", fontsize=14, y=1.02)
    plt.tight_layout()
    out = HERE / sub / "sprint4_radar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    return out


def combined_bar(arms: list[tuple[str, str, dict]]) -> Path:
    labels = KEYS
    x = np.arange(len(labels))
    palette = ["#888", "#1f77b4", "#d62728"]
    series = []
    for (sub, nice, d), color in zip(arms, palette):
        series.append((f"{nice} enh/hyb", d["enhanced"]["hybrid"]["mean"], color))
    w = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (name, m, color) in enumerate(series):
        vals = [m[k] for k in labels]
        ax.bar(x + (i - (len(series) - 1) / 2) * w, vals, w,
               label=name, color=color)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.05)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0])
    ax.axhline(0.9, color="gray", linewidth=0.8, alpha=0.55, zorder=0)
    ax.set_ylabel("Score")
    ax.set_title("Sprint 4 — gated fusion & A/V node flag vs sprint3 winner "
                 "(enhanced/hybrid, 5-fold CV mean)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = HERE / "sprint4_combined_bar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    return out


def arms_radar(arms: list[tuple[str, str, dict]]) -> Path:
    angles = np.linspace(0, 2 * np.pi, len(KEYS), endpoint=False).tolist() + [0.0]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))
    palette = ["#888", "#1f77b4", "#d62728"]
    for ax, mode in zip(axes, MODES):
        for (sub, nice, d), color in zip(arms, palette):
            vals = [d["enhanced"][mode]["mean"][k] for k in KEYS]
            vals += vals[:1]
            ax.plot(angles, vals, color=color, linewidth=2, label=nice)
            ax.fill(angles, vals, color=color, alpha=0.10)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(KEYS, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 0.9, 1.0], angle=30, fontsize=7)
        ax.grid(True, color="gray", alpha=0.55, linewidth=0.7)
        ax.set_title(f"enhanced / {mode}", y=1.08, fontsize=12)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    fig.suptitle("Sprint 4 — enhanced feature set, 3 arms compared",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    out = HERE / "sprint4_arms_radar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    return out


def summary_xlsx(arms: list[tuple[str, str, dict]]) -> Path:
    rows = []
    for sub, nice, d in arms:
        for fs in FEATS:
            for mode in MODES:
                r = d[fs][mode]
                row = {"run": nice, "feat_set": fs, "mode": mode,
                       "pooled_AUC": r["pooled_AUC"]}
                for k in KEYS:
                    row[k] = r["mean"][k]
                rows.append(row)
    df = pd.DataFrame(rows)
    out = HERE / "sprint4_vs_sprint3.xlsx"
    df.to_excel(out, index=False)
    return out


def main() -> None:
    arms = []
    for sub, nice in ARMS:
        d = load(sub)
        if d is not None:
            arms.append((sub, nice, d))
    if len(arms) < 2:
        print("need ≥2 arms to make a comparison")
        return
    for sub, _, d in arms:
        if sub.startswith("sprint4"):
            print("xlsx:", per_arm_xlsx(sub, d))
            print("radar:", per_arm_radar(sub, d))
    print("combined bar:", combined_bar(arms))
    print("arms radar:", arms_radar(arms))
    print("summary xlsx:", summary_xlsx(arms))

    print("\n" + "=" * 130)
    print(f"{'run':<24} {'feat':<10} {'mode':<16} "
          + " ".join(f"{k:>10}" for k in KEYS) + "  pooled")
    print("-" * 130)
    for sub, nice, d in arms:
        for fs in FEATS:
            for mode in MODES:
                r = d[fs][mode]; m = r["mean"]
                print(f"{nice:<24} {fs:<10} {mode:<16} "
                      + " ".join(f"{m[k]:>10.4f}" for k in KEYS)
                      + f"  {r['pooled_AUC']:.4f}")


if __name__ == "__main__":
    main()
