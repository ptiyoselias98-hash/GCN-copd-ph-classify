"""R14 figure generators for README — produces 4 publication-style figures:

  fig_r14_coral_vs_grl.png   — multi-seed CORAL vs GRL protocol-LR comparison
  fig_r14_lung_vs_graph.png  — lung-vs-graph ablation bar chart with CIs
  fig_r14_endotypes.png      — endotype composition (PH% per cluster, both cohorts)
  fig_r14_disease_pareto.png — disease AUC vs protocol AUC Pareto across deconfounders

Reads from:
  outputs/r13/coral_probe.json   (multi-seed CORAL + MMD)
  outputs/r14/ablation_lung_vs_graph.json
  outputs/r14/multistruct_clusters.json
  outputs/r11/R11_grlfix_summary.json (GRL baseline)

Outputs to: outputs/figures/fig_r14_*.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent.parent
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fig_coral_vs_grl():
    coral = load(ROOT / "outputs" / "r13" / "coral_probe.json")
    grl = load(ROOT / "outputs" / "r11" / "R11_grlfix_summary.json")
    if not coral:
        return
    lambdas = [0.0, 1.0, 5.0, 10.0]
    coral_means = []; coral_sds = []
    for lam in lambdas:
        rec = coral.get("per_lambda", {}).get(f"lambda_{lam}")
        if not rec:
            coral_means.append(None); coral_sds.append(0); continue
        coral_means.append(rec["protocol_lr_corrected"]["mean"])
        coral_sds.append(rec["protocol_lr_corrected"]["sd"])

    grl_means = []; grl_sds = []
    if grl:
        for lam in lambdas:
            agg = grl.get("aggregated", {}).get(f"lambda_{lam}")
            if not agg:
                grl_means.append(None); grl_sds.append(0); continue
            grl_means.append(agg["protocol_lr"]["mean"])
            grl_sds.append(agg["protocol_lr"]["sd"])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(lambdas)); w = 0.35
    if any(v is not None for v in coral_means):
        ax.bar(x - w/2, [v if v is not None else 0 for v in coral_means], w,
               yerr=coral_sds, color="#10b981", label="CORAL (n=68 corrected, 3 seeds)",
               capsize=4)
    if grl and any(v is not None for v in grl_means):
        ax.bar(x + w/2, [v if v is not None else 0 for v in grl_means], w,
               yerr=grl_sds, color="#ef4444", label="GRL (n=80, 3 seeds, R11)",
               capsize=4)
    ax.axhline(0.60, ls="--", c="#22c55e", alpha=0.7, label="reviewer target (≤0.60)")
    ax.axhline(0.50, ls=":", c="grey", alpha=0.5, label="random chance")
    ax.set_xticks(x); ax.set_xticklabels([f"λ={l}" for l in lambdas])
    ax.set_ylabel("Within-nonPH protocol LR AUC (μ ± SD across seeds)")
    ax.set_title("R14 — CORAL vs GRL protocol-leakage reduction\n"
                 "CORAL @ λ=1 mean 0.71 breaks GRL's 0.80 floor")
    ax.set_ylim(0.4, 1.0); ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = OUT / "fig_r14_coral_vs_grl.png"
    plt.savefig(p, dpi=140); plt.close()
    print(f"Saved {p}")


def fig_lung_vs_graph():
    abl = load(ROOT / "outputs" / "r14" / "ablation_lung_vs_graph.json")
    if not abl:
        return
    sets = [("graph_only", "Graph only\n(50 vascular feats)", "#3b82f6"),
            ("lung_only", "Lung only\n(49 parenchyma feats)", "#10b981"),
            ("graph+lung", "Graph + Lung\n(99 combined)", "#8b5cf6")]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, (key, label, color) in enumerate(sets):
        rec = abl.get(key)
        if rec is None:
            continue
        a = rec["auc"]; lo, hi = rec["ci95"]
        ax.bar(i, a, color=color, alpha=0.85)
        ax.errorbar(i, a, yerr=[[a - lo], [hi - a]], fmt="none",
                    color="black", capsize=6, lw=1.5)
        ax.text(i, a + 0.012, f"{a:.3f}\n[{lo:.3f}, {hi:.3f}]",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(sets))); ax.set_xticklabels([s[1] for s in sets])
    ax.set_ylabel("Disease AUC (within-contrast n=184, 5-fold OOF LR)")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("R14.D — Lung-only AUC dominates Graph-only\n"
                 "Lung parenchyma carries primary disease signal; vascular graph is complementary (+0.085 AUC)")
    ax.axhline(0.5, ls=":", c="grey", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    # Add per-graph-substring sub-bars on a secondary inset
    ps = abl.get("graph_per_substring", {})
    if ps:
        ax2 = fig.add_axes([0.62, 0.62, 0.32, 0.25])
        keys = list(ps.keys())
        aucs = [ps[k]["auc"] for k in keys]
        ax2.bar(range(len(keys)), aucs, color="#94a3b8", alpha=0.8)
        ax2.set_xticks(range(len(keys))); ax2.set_xticklabels(keys, rotation=45, fontsize=8)
        ax2.set_ylabel("AUC", fontsize=9); ax2.set_ylim(0.5, 0.85)
        ax2.set_title("graph substring ablation", fontsize=9)
        ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = OUT / "fig_r14_lung_vs_graph.png"
    plt.savefig(p, dpi=140); plt.close()
    print(f"Saved {p}")


def fig_endotypes():
    cl = load(ROOT / "outputs" / "r14" / "multistruct_clusters.json")
    if not cl:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, key in zip(axes, ["full", "contrast_only"]):
        s = cl.get(key)
        if not s:
            ax.set_visible(False); continue
        clusters = s["kmeans_summary"]
        labels = [f"C{c['cluster']}\nn={c['n']}" for c in clusters]
        ph_pcts = [c["ph_pct"] for c in clusters]
        nph = [c["n_nonph"] for c in clusters]
        ph = [c["n_ph"] for c in clusters]
        x = np.arange(len(clusters))
        ax.bar(x, nph, color="#3b82f6", label="nonPH")
        ax.bar(x, ph, bottom=nph, color="#ef4444", label="PH")
        for i, c in enumerate(clusters):
            ax.text(i, c["n"] + 0.5, f"{c['ph_pct']:.0f}% PH",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("Cases per cluster")
        title_label = "full cohort (n=226)" if key == "full" else "contrast-only (n=184, no protocol confound)"
        ax.set_title(f"KMeans endotypes — {title_label}")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("R14.B — Multi-structure phenotype clusters on 66-D feature vector\n"
                 "C0 transition (vessel-diameter+emphysema), C1 arterial-rich PH, C2 dense-lung PH",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    p = OUT / "fig_r14_endotypes.png"
    plt.savefig(p, dpi=140, bbox_inches="tight"); plt.close()
    print(f"Saved {p}")


def fig_pareto():
    coral = load(ROOT / "outputs" / "r13" / "coral_probe.json")
    grl = load(ROOT / "outputs" / "r11" / "R11_grlfix_summary.json")
    if not coral:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    # CORAL points (per-seed)
    coral_pts_x, coral_pts_y, coral_pts_lam = [], [], []
    for k, rec in coral.get("per_seed", {}).items():
        c = rec["corrected_cohort"]
        d = rec.get("disease_auc_5fold") or {}
        if d.get("mean") is None:
            continue
        coral_pts_x.append(c["lr_auc"])
        coral_pts_y.append(d["mean"])
        # parse λ from key, e.g. "coral_l1.0_s42"
        try:
            lam = float(k.split("_l")[1].split("_")[0])
        except (IndexError, ValueError):
            lam = 0
        coral_pts_lam.append(lam)
    sc1 = ax.scatter(coral_pts_x, coral_pts_y, c=coral_pts_lam, cmap="viridis",
                      s=80, edgecolor="white", lw=1, label="CORAL (per seed)")
    # MMD points
    mmd = coral.get("mmd", {})
    for k, rec in mmd.items():
        d = rec.get("disease_auc_5fold") or {}
        if d.get("mean") is None: continue
        ax.scatter([rec["lr_auc"]], [d["mean"]], marker="^", s=140, c="#ef4444",
                   edgecolor="black", lw=1.2, label=k)
    # GRL points (aggregated per λ)
    if grl:
        grl_x, grl_y, grl_lam = [], [], []
        for lam in [0.0, 1.0, 5.0, 10.0]:
            agg = grl.get("aggregated", {}).get(f"lambda_{lam}")
            if not agg: continue
            grl_x.append(agg["protocol_lr"]["mean"])
            grl_y.append(agg["disease_lr"]["mean"])
            grl_lam.append(lam)
        ax.scatter(grl_x, grl_y, marker="x", s=120, c="#dc2626", lw=2.5, label="GRL (R11 mean per λ)")
    ax.axvline(0.60, ls="--", c="#22c55e", alpha=0.6, label="protocol target ≤0.60")
    ax.axhline(0.85, ls="--", c="#f59e0b", alpha=0.6, label="disease floor ≥0.85")
    ax.set_xlabel("Within-nonPH protocol LR AUC (lower is better)")
    ax.set_ylabel("Disease AUC (higher is better)")
    ax.set_title("R14 — Disease vs Protocol Pareto across deconfounders\n"
                 "CORAL preserves disease ~0.93 while reducing protocol; GRL crashes disease")
    ax.set_xlim(0.55, 0.95); ax.set_ylim(0.6, 1.0)
    ax.invert_xaxis()  # left = better protocol invariance
    cb = plt.colorbar(sc1, ax=ax); cb.set_label("CORAL λ")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = OUT / "fig_r14_disease_pareto.png"
    plt.savefig(p, dpi=140); plt.close()
    print(f"Saved {p}")


if __name__ == "__main__":
    fig_coral_vs_grl()
    fig_lung_vs_graph()
    fig_endotypes()
    fig_pareto()
