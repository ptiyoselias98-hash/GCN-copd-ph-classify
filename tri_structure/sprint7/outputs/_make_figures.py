"""Generate Sprint 7 summary figures for the README.

Outputs under tri_structure/sprint7/outputs/figures/:
  - sweep_edrop_auc.png      : AUC vs edge_drop_p
  - phase2_vs_phase1_auc.png : grouped bar Phase1 / v2 / sprint7 across 6 metrics
  - attention_mpap_sprint7.png : |r| of per-structure attention vs mPAP
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


def _auc(path):
    with open(path) as f:
        d = json.load(f)
    m = d["mean_metrics"]["auc"]
    return m["mean"], m["std"]


def sweep_figure():
    ps = ["0.0", "0.05", "0.10", "0.15"]
    xs = [float(p) for p in ps]
    ys, es = [], []
    for p in ps:
        mu, sd = _auc(ROOT / f"sprint7_sweep_edrop/sweep_edrop_p{p}/cv_results.json")
        ys.append(mu)
        es.append(sd)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(xs, ys, yerr=es, marker="o", capsize=4, color="#1f77b4", lw=2)
    best_i = int(np.argmax(ys))
    ax.scatter([xs[best_i]], [ys[best_i]], s=150, facecolors="none",
               edgecolors="#d62728", lw=2.5, zorder=5, label=f"best p={xs[best_i]}")
    ax.set_xlabel("edge_drop_p")
    ax.set_ylabel("AUC (5-fold, 1 repeat)")
    ax.set_title("Sprint 7 Task 3 — edge-dropout sweep")
    ax.set_xticks(xs)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower center")
    ax.set_ylim(0.45, 0.95)
    fig.tight_layout()
    out = FIG / "sweep_edrop_auc.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def phase_compare_figure():
    with open(ROOT / "sprint7_phase2/cv_results.json") as f:
        s7 = json.load(f)["mean_metrics"]

    # Phase 1 and v2 numbers from README (15-fold CV, n=106)
    phase1 = {"auc": (0.880, 0.093), "accuracy": (0.877, 0.089),
              "sensitivity": (0.869, 0.113), "specificity": (0.900, 0.119),
              "f1": (0.910, 0.070), "precision": (0.964, 0.041)}
    v2 = {"auc": (0.734, 0.142), "accuracy": (0.739, 0.115),
          "sensitivity": (0.708, 0.144), "specificity": (0.824, 0.166),
          "f1": (0.794, 0.101), "precision": (0.926, 0.064)}
    sprint7 = {k: (s7[k]["mean"], s7[k]["std"]) for k in phase1}

    metrics = list(phase1.keys())
    labels = ["AUC", "Acc", "Sens", "Spec", "F1", "Prec"]
    x = np.arange(len(metrics))
    w = 0.26

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (name, d, color) in enumerate([
        ("Phase 1 (mean pool)", phase1, "#2ca02c"),
        ("v2 (attn + signatures)", v2, "#ff7f0e"),
        ("Sprint 7 (tri-cache + reg.)", sprint7, "#1f77b4"),
    ]):
        means = [d[k][0] for k in metrics]
        stds = [d[k][1] for k in metrics]
        ax.bar(x + (i - 1) * w, means, w, yerr=stds, capsize=3,
               color=color, label=name, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("metric value (mean ± std)")
    ax.set_title("Sprint 7 vs Phase 1 vs v2 — 6-metric comparison (n=106)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = FIG / "phase2_vs_phase1_auc.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def attention_figure():
    # From sprint7_phase2.log lines 79-81 (absolute values)
    sprint7 = {"artery": 0.073, "airway": 0.042}
    # From tri_structure/RESULTS.md (Phase 1 vein not reported; omit)
    phase1 = {"artery": 0.486, "airway": 0.468}

    structures = ["artery", "airway"]
    x = np.arange(len(structures))
    w = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w / 2, [phase1[s] for s in structures], w,
           color="#2ca02c", label="Phase 1", alpha=0.85)
    ax.bar(x + w / 2, [sprint7[s] for s in structures], w,
           color="#1f77b4", label="Sprint 7", alpha=0.85)
    ax.axhline(0.40, color="#d62728", ls="--", lw=1.2,
               label="plan investigation threshold (0.40)")
    ax.set_xticks(x)
    ax.set_xticklabels(structures)
    ax.set_ylabel("|r(attention, mPAP)|")
    ax.set_title("Per-structure attention × mPAP — Phase 1 vs Sprint 7")
    ax.set_ylim(0, 0.6)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = FIG / "attention_mpap_sprint7.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    sweep_figure()
    phase_compare_figure()
    attention_figure()
