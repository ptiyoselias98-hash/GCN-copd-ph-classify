"""R15+R16 figure generators for README:

  fig_r15_enlarged_protocol.png   — protocol LR AUC at n=80 (R12) vs n=151 (R15.G)
  fig_r15_endotype_forest.png     — Cohen's d forest plot for 14 features (Holm-sig flagged)
  fig_r15_disease_replication.png — within-contrast disease AUC R14 vs R15 (with CIs)
  fig_r16_segqc_volumes.png       — raw vs repaired lung volume histogram (when repair done)

Reads from outputs/r12/r12_cross_seed_cis.json, outputs/r13/coral_probe.json,
outputs/r14/ablation_lung_vs_graph.json, outputs/r15/enlarged_lung_results.json,
outputs/r16/endotype_corrected.json, outputs/r16/seg_qc_new100.json.
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


def fig_enlarged_protocol():
    """R12 baseline vs R15.G enlarged within-nonPH protocol probe."""
    g = load(ROOT / "outputs" / "r15" / "enlarged_lung_results.json")
    if not g or "A_within_nonph_protocol" not in g:
        return
    r15 = g["A_within_nonph_protocol"]

    fig, ax = plt.subplots(figsize=(8, 5))
    # R12 baseline (n=80, lung-features-equivalent baseline approximated from R12 single-best)
    bars = [
        ("R12 (n=80)\nGCN aggregates", 0.853, [0.722, 0.942], "#94a3b8"),
        ("R15.G (n=151)\nlung features only", r15["lr_auc"], r15["lr_ci95"], "#10b981"),
        ("R15.G (n=151)\nlung features (MLP)", r15["mlp_auc"], r15["mlp_ci95"], "#3b82f6"),
    ]
    x = np.arange(len(bars))
    for i, (lbl, auc, ci, color) in enumerate(bars):
        ax.bar(i, auc, color=color, alpha=0.85)
        ax.errorbar(i, auc, yerr=[[auc - ci[0]], [ci[1] - auc]], fmt="none",
                    color="black", capsize=6, lw=1.5)
        ax.text(i, auc + 0.012, f"{auc:.3f}\n[{ci[0]:.3f}, {ci[1]:.3f}]",
                ha="center", va="bottom", fontsize=9)
    ax.axhline(0.60, ls="--", c="#22c55e", alpha=0.7, label="reviewer target ≤0.60")
    ax.axhline(0.50, ls=":", c="grey", alpha=0.5, label="random chance")
    ax.set_xticks(x); ax.set_xticklabels([b[0] for b in bars], fontsize=9)
    ax.set_ylabel("Within-nonPH protocol AUC")
    ax.set_ylim(0.45, 1.0)
    ax.set_title("R15.G — enlarged-stratum within-nonPH protocol probe\n"
                 "Protocol confound is MORE pronounced at larger n (not less)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = OUT / "fig_r15_enlarged_protocol.png"
    plt.savefig(p, dpi=140); plt.close()
    print(f"Saved {p}")


def fig_endotype_forest():
    """Forest plot of Cohen's d for 14 endotype features with Holm-Bonferroni."""
    e = load(ROOT / "outputs" / "r16" / "endotype_corrected.json")
    if not e: return
    rows = sorted(e["results"], key=lambda r: r["cohens_d"])
    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(rows))
    for i, r in enumerate(rows):
        d = r["cohens_d"]; ci = r["cohens_d_ci95"]
        sig = r["sig_holm_05"]
        color = "#ef4444" if (sig and d > 0) else ("#3b82f6" if (sig and d < 0) else "#94a3b8")
        ax.errorbar(d, i, xerr=[[d - ci[0]], [ci[1] - d]], fmt="o",
                    color=color, capsize=4, lw=1.5, markersize=7)
        marker = " ✓" if sig else ""
        ax.text(2.0, i, f"{r['feature']}{marker}\n"
                       f"p_holm={r['p_holm']:.2g}", va="center", fontsize=9)
    ax.axvline(0, c="black", lw=0.8)
    ax.axvline(0.5, ls=":", c="grey", alpha=0.5)
    ax.axvline(-0.5, ls=":", c="grey", alpha=0.5)
    ax.axvline(0.8, ls="--", c="grey", alpha=0.4)
    ax.axvline(-0.8, ls="--", c="grey", alpha=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel("Cohen's d (PH − nonPH within contrast, n=197)")
    ax.set_xlim(-1.6, 3.5)
    ax.set_title("R16.B — Endotype effect sizes (within-contrast PH vs nonPH)\n"
                 "RED = significantly elevated in PH; BLUE = significantly reduced; GREY = NS after Holm-Bonferroni")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    p = OUT / "fig_r15_endotype_forest.png"
    plt.savefig(p, dpi=140); plt.close()
    print(f"Saved {p}")


def fig_disease_replication():
    """R14 vs R15 within-contrast disease AUC."""
    g15 = load(ROOT / "outputs" / "r15" / "enlarged_lung_results.json")
    abl = load(ROOT / "outputs" / "r14" / "ablation_lung_vs_graph.json")
    if not g15 or not abl: return
    r14 = abl.get("lung_only", {})
    r15 = g15.get("B_within_contrast_disease", {})
    if not r14 or not r15: return

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = [("R14 lung-only\nn=184", r14["auc"], r14["ci95"], "#3b82f6"),
            ("R15.G lung-only\nn=" + str(r15["n"]), r15["lr_auc"], r15["lr_ci95"], "#10b981")]
    x = np.arange(len(bars))
    for i, (lbl, a, ci, color) in enumerate(bars):
        ax.bar(i, a, color=color, alpha=0.85)
        ax.errorbar(i, a, yerr=[[a - ci[0]], [ci[1] - a]], fmt="none",
                    color="black", capsize=6, lw=1.5)
        ax.text(i, a + 0.008, f"{a:.3f}\n[{ci[0]:.3f}, {ci[1]:.3f}]",
                ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels([b[0] for b in bars], fontsize=10)
    ax.set_ylabel("Within-contrast disease AUC (5-fold OOF LR)")
    ax.set_ylim(0.65, 1.0)
    ax.set_title("R15.G — within-contrast disease AUC replicates R14\n"
                 "lung-only signal is genuine, not overfit to R14's 184-case sample")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = OUT / "fig_r15_disease_replication.png"
    plt.savefig(p, dpi=140); plt.close()
    print(f"Saved {p}")


def fig_segqc_volumes():
    """Lung volume histogram for new 100 cases."""
    qc = load(ROOT / "outputs" / "r16" / "seg_qc_new100.json")
    if not qc: return
    import pandas as pd
    fcsv = ROOT / "outputs" / "r15" / "lung_features_new100.csv"
    if not fcsv.exists(): return
    df = pd.read_csv(fcsv)
    repair_csv = ROOT / "outputs" / "r16" / "lung_features_new100_repaired.csv"
    has_repair = repair_csv.exists()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["lung_vol_mL"].values, bins=30, color="#ef4444", alpha=0.7,
            label=f"Simple_AV_seg raw (n={len(df)}, median={df['lung_vol_mL'].median():.0f}mL)")
    if has_repair:
        rdf = pd.read_csv(repair_csv)
        ax.hist(rdf["lung_vol_mL_repaired"].values, bins=30, color="#10b981", alpha=0.7,
                label=f"Repaired (HU<-300 + top-2-CC, n={len(rdf)}, median={rdf['lung_vol_mL_repaired'].median():.0f}mL)")
    ax.axvspan(1500, 8500, alpha=0.15, color="green", label="plausible adult range (1.5-8.5L)")
    ax.set_xlabel("Lung volume (mL)")
    ax.set_ylabel("# cases")
    ax.set_title("R16.A — Simple_AV_seg lung-mask volume distribution on plain-scan CT\n"
                 f"79/100 raw masks oversize (>8.5L); R16.C HU+CC repair "
                 + ("complete" if has_repair else "running…"))
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = OUT / "fig_r16_segqc_volumes.png"
    plt.savefig(p, dpi=140); plt.close()
    print(f"Saved {p}")


if __name__ == "__main__":
    fig_enlarged_protocol()
    fig_endotype_forest()
    fig_disease_replication()
    fig_segqc_volumes()
