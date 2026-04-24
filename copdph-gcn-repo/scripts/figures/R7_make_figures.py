"""R7 — Build publication-style figure suite for the README.

Outputs to copdph-gcn-repo/outputs/figures/:
  fig1_aris_score_progression.png  — Round 1-6 hostile-review score
  fig2_protocol_decoder_bars.png   — protocol AUC across feature sets (full vs within-nonPH)
  fig3_paired_delong_forest.png    — Δ AUC forest plot for arm comparisons
  fig4_hipas_directions.png        — PH vs nonPH artery/vein abundance
  fig5_cache_coverage_sankey.png   — 282 → 243 case flow
  fig6_classifier_heatmap.png      — feature_set × endpoint matrix
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
OUT_DIR = ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.titleweight"] = "bold"


# ---------- Fig 1: ARIS score progression ----------
def fig1():
    rounds = [1, 2, 3, 4, 5, 6]
    scores = [2, 3, 4, 5, 6, 5]
    labels = [
        "R1\nbaseline",
        "R2\n+W2/W6\n+protocol\nablation",
        "R3\n+E3/R3\n+REPRODUCE",
        "R4\n+within-nonPH\n+overlay\n+exclusion",
        "R5\n+DeLong CI\n+GCN-input\nprotocol",
        "R6\n+paired DeLong\n+OOF csv\n+env lock",
    ]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(rounds, scores, "-o", linewidth=2.5, markersize=12, color="tab:blue")
    ax.axhline(8, ls="--", color="tab:green", alpha=0.7, label="target = 8/10")
    ax.axhline(6, ls=":", color="tab:orange", alpha=0.5, label="almost = 6/10")
    for r, s, lab in zip(rounds, scores, labels):
        ax.annotate(f"{s}", xy=(r, s), xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=11, fontweight="bold")
        ax.annotate(lab, xy=(r, s), xytext=(0, -28), textcoords="offset points",
                    ha="center", fontsize=8, color="dimgray")
    ax.set_xticks(rounds)
    ax.set_xticklabels([f"R{r}" for r in rounds])
    ax.set_ylim(0, 10.5)
    ax.set_xlabel("ARIS hostile review round")
    ax.set_ylabel("Score (1–10, hard-mode)")
    ax.set_title("ARIS adversarial review progression — codex-mcp gpt-5.2 high-reasoning")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fp = OUT_DIR / "fig1_aris_score_progression.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"  {fp.name}")


# ---------- Fig 2: protocol decoder bars ----------
def fig2():
    # Data from R4.1 within-nonPH and R3 full-cohort
    feature_sets = ["v1_whole_lung", "v2_paren_only", "v2_paren_LAA", "v2_spatial_paren",
                    "v2_per_struct_vols", "v2_vessel_ratios", "v2_combined_no_HU"]
    full_lr = [1.000, 0.857, 0.591, 0.732, 0.524, 0.885, 0.860]
    nonph_lr = [0.765, 0.794, 0.715, 0.669, 0.529, 0.674, 0.731]
    nonph_lo = [0.697, 0.705, 0.646, 0.543, 0.429, 0.542, 0.653]
    nonph_hi = [0.833, 0.886, 0.789, 0.795, 0.631, 0.805, 0.810]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(feature_sets))
    w = 0.4
    ax.bar(x - w/2, full_lr, w, label="full cohort (label↔protocol coupled)",
           color="tab:gray", alpha=0.6)
    ax.bar(x + w/2, nonph_lr, w, label="within-nonPH (HONEST)",
           color="tab:red", alpha=0.85,
           yerr=[np.array(nonph_lr) - np.array(nonph_lo),
                 np.array(nonph_hi) - np.array(nonph_lr)], capsize=3)
    ax.axhline(0.5, ls="--", color="black", alpha=0.5, label="random chance")
    ax.axhline(0.7, ls=":", color="tab:green", alpha=0.5, label="acceptable bound (≤ 0.7)")
    ax.set_xticks(x)
    ax.set_xticklabels(feature_sets, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Logistic Regression protocol AUC (5-fold CV)")
    ax.set_title("Protocol decodability: full cohort vs within-nonPH (label-leakage stripped)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fp = OUT_DIR / "fig2_protocol_decoder_bars.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"  {fp.name}")


# ---------- Fig 3: paired DeLong forest ----------
def fig3():
    rows = [
        ("arm_c − arm_b (full cohort)", 0.039, 0.028, 0.052, "blue"),
        ("arm_c − arm_b (contrast-only)", 0.006, -0.004, 0.018, "blue"),
        ("arm_b full − arm_b contrast-only", 0.049, 0.016, 0.084, "purple"),
        ("arm_c full − arm_c contrast-only", 0.082, 0.055, 0.109, "purple"),
        ("arm_c − arm_a (contrast-only, paired DeLong, n=189)", 0.025, -0.039, 0.089, "tab:red"),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ys = np.arange(len(rows))[::-1]
    for y, (name, mu, lo, hi, c) in zip(ys, rows):
        ax.errorbar(mu, y, xerr=[[mu - lo], [hi - mu]], fmt="o", color=c, capsize=4,
                    markersize=9, linewidth=1.6)
        ax.text(0.13, y, f"{mu:+.3f} [{lo:+.3f}, {hi:+.3f}]", va="center", fontsize=9,
                color=c)
    ax.axvline(0, ls="--", color="black", alpha=0.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlim(-0.07, 0.20)
    ax.set_xlabel("Δ AUC")
    ax.set_title("Paired Δ-AUC forest plot — significance vanishes under protocol balancing")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fp = OUT_DIR / "fig3_paired_delong_forest.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"  {fp.name}")


# ---------- Fig 4: HiPaS direction test ----------
def fig4():
    sl = pd.read_csv(ROOT / "outputs" / "skeleton_length.csv")
    proto = pd.read_csv(ROOT / "data" / "case_protocol.csv")
    df = proto.merge(sl, on="case_id")
    df["SL_artery_per_L"] = df["SL_artery_mm"] / df["lung_vol_mL"] * 1000
    df["SL_vein_per_L"] = df["SL_vein_mm"] / df["lung_vol_mL"] * 1000
    df = df.dropna(subset=["SL_artery_per_L", "SL_vein_per_L"])
    contrast = df[df["protocol"] == "contrast"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    rng = np.random.default_rng(20260424)

    # Panel A: Artery SL/L vs PH/nonPH (T1)
    ax = axes[0]
    for grp, color in [(0, "tab:blue"), (1, "tab:red")]:
        vals = contrast[contrast["label"] == grp]["SL_artery_per_L"].values
        x = grp + rng.uniform(-0.18, 0.18, len(vals))
        ax.scatter(x, vals, alpha=0.55, s=22, color=color)
        ax.hlines(np.median(vals), grp - 0.25, grp + 0.25, color="black", linewidth=2.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["nonPH (n=27)", "PH (n=163)"])
    ax.set_ylabel("Artery skeleton length per L lung (mm)")
    ax.set_title("T1: PH vs nonPH artery abundance (contrast only)\nδ=+0.354, p=0.003 — DIRECTION OPPOSITE TO HiPaS")
    ax.grid(True, alpha=0.3)

    # Panel B: Vein SL/L vs LAA-910 (T2)
    ax = axes[1]
    v2 = pd.read_csv(ROOT / "outputs" / "lung_features_v2.csv")[["case_id", "paren_LAA_910_frac"]]
    df2 = df.merge(v2, on="case_id")
    contrast2 = df2[df2["protocol"] == "contrast"].dropna(subset=["paren_LAA_910_frac"])
    ax.scatter(contrast2["paren_LAA_910_frac"], contrast2["SL_vein_per_L"],
               c=contrast2["label"], cmap="coolwarm", alpha=0.7, s=24)
    # Spearman line
    from scipy.stats import linregress
    if len(contrast2) > 5:
        m, b, r, p, _ = linregress(contrast2["paren_LAA_910_frac"], contrast2["SL_vein_per_L"])
        xs = np.linspace(contrast2["paren_LAA_910_frac"].min(),
                         contrast2["paren_LAA_910_frac"].max(), 50)
        ax.plot(xs, m * xs + b, "--", color="black", alpha=0.6, label=f"r={r:.2f}")
    ax.set_xlabel("Parenchyma LAA-910 fraction (emphysema severity)")
    ax.set_ylabel("Vein skeleton length per L lung (mm)")
    ax.set_title("T2: COPD severity vs vein abundance (contrast)\nρ=−0.65, p<10⁻³³ — MATCHES HiPaS")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fp = OUT_DIR / "fig4_hipas_directions.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"  {fp.name}")


# ---------- Fig 5: cache coverage stratified bar ----------
def fig5():
    audit = pd.read_csv(ROOT / "outputs" / "r6" / "missing_cache_audit.csv")
    audit["bucket"] = audit.apply(
        lambda r: f"label={r['label']}\n{r['protocol']}", axis=1)
    counts = audit.groupby(["bucket", "in_cache_v2_tri_flat"]).size().unstack(fill_value=0)
    counts.columns = ["missing", "in_cache"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    counts.plot(kind="bar", stacked=True, ax=ax,
                color=["tab:red", "tab:green"], width=0.6)
    for i, (b, row) in enumerate(counts.iterrows()):
        total = row.sum()
        miss = row["missing"]
        if miss > 0:
            ax.annotate(f"{miss}/{total}\n({100*miss/total:.0f}% miss)",
                        xy=(i, total + 1), ha="center", fontsize=9, color="tab:red")
    ax.set_xticklabels(counts.index, rotation=0)
    ax.set_ylabel("Number of cases")
    ax.set_title("Cache_v2_tri_flat coverage — missingness is label-correlated\n"
                 "(31/39 missing = plain-scan nonPH = vessel-segmentation placeholders)")
    ax.legend(["Missing (39)", "In cache (243)"])
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fp = OUT_DIR / "fig5_cache_coverage.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"  {fp.name}")


# ---------- Fig 6: classifier heatmap ----------
def fig6():
    # rows = feature sets, cols = endpoints
    feats = ["whole_lung_v1", "paren_only", "paren+spatial",
             "vessel_lung_integ", "GCN_input_aggregates"]
    cols = ["protocol\nfull cohort", "protocol\nwithin-nonPH",
            "disease\nfull cohort", "disease\ncontrast-only"]
    mat = np.array([
        [1.000, 0.765, 0.879, 0.677],   # whole_lung_v1
        [0.857, 0.794, 0.870, 0.860],   # paren_only
        [0.866, 0.731, 0.879, 0.855],   # paren+spatial
        [0.945, 0.674, 0.861, 0.774],   # vessel_lung_integ
        [0.936, 0.853, np.nan, 0.858],  # GCN_input_aggregates
    ])
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(mat, cmap="RdYlGn_r", vmin=0.45, vmax=1.0, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=10, color="black")
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=11,
                        color="white" if v > 0.78 or v < 0.6 else "black",
                        fontweight="bold")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=9)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=10)
    ax.set_title("Feature-set × endpoint AUC matrix (LR, 5-fold CV)\n"
                 "Within-nonPH protocol AUC ≈ honest leakage; disease contrast = honest performance")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="AUC")
    fig.tight_layout()
    fp = OUT_DIR / "fig6_classifier_heatmap.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"  {fp.name}")


def main():
    print("Building figures…")
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    print(f"All figures in {OUT_DIR}")


if __name__ == "__main__":
    main()
