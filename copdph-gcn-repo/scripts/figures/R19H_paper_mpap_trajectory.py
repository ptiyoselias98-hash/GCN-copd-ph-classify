"""R19.H — Paper-quality mPAP 5-stage evolution trajectory figure.

Combines R18.B (R17 morph) + R19.F (R19.D verification on legacy-only) +
R16.B (lung_features_v2) into a single 6-panel figure showing the strongest
endotype features across mPAP stages, with stage-wise mean ± 95% CI.

Output: outputs/figures/fig_r19_paper_mpap_trajectory.png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent.parent
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
MORPH = ROOT / "outputs" / "r17" / "per_structure_morphometrics.csv"
LUNG = ROOT / "outputs" / "lung_features_v2.csv"
LABELS = ROOT / "data" / "labels_extended_382.csv"
PROTO = ROOT / "data" / "case_protocol_extended.csv"
SEG_FAILS = ROOT / "outputs" / "r13" / "seg_failures_real.json"
MPAP = ROOT / "data" / "mpap_lookup_gold.json"


def main():
    morph = pd.read_csv(MORPH); lung = pd.read_csv(LUNG)
    lab = pd.read_csv(LABELS); pro = pd.read_csv(PROTO)
    df = lab.merge(pro[["case_id", "protocol"]], on="case_id") \
        .merge(morph, on="case_id", how="left", suffixes=("", "_dup1")) \
        .merge(lung, on="case_id", how="left", suffixes=("", "_dup2"))
    fails = set()
    if SEG_FAILS.exists():
        sf = json.loads(SEG_FAILS.read_text(encoding="utf-8"))
        fails = {r["case_id"] for r in sf.get("real_fails", []) + sf.get("lung_anomaly", [])}
    df = df[~df["case_id"].isin(fails)].copy()
    mpap = json.loads(MPAP.read_text(encoding="utf-8"))
    df["mpap"] = df["case_id"].map(mpap)
    df.loc[df["protocol"].str.lower() == "plain_scan", "mpap"] = 5.0
    df.loc[(df["label"] == 0) & (df["protocol"].str.lower() == "contrast"), "mpap"] = 15.0
    df["stage"] = -1
    df.loc[df["protocol"].str.lower() == "plain_scan", "stage"] = 0
    df.loc[(df["label"] == 0) & (df["protocol"].str.lower() == "contrast"), "stage"] = 1
    df.loc[(df["label"] == 1) & (df["mpap"] < 25), "stage"] = 2
    df.loc[(df["label"] == 1) & (df["mpap"] >= 25) & (df["mpap"] < 35), "stage"] = 3
    df.loc[(df["label"] == 1) & (df["mpap"] >= 35), "stage"] = 4
    df = df[df["stage"] >= 0].copy()

    # 6 panels for the highest-effect features
    feats = [
        ("artery_tort_p10", "Artery tortuosity p10", "#ef4444"),
        ("artery_len_p25",  "Artery edge length p25 (mm)", "#dc2626"),
        ("artery_len_p50",  "Artery edge length p50 (mm)", "#b91c1c"),
        ("vein_len_p25",    "Vein edge length p25 (mm)", "#3b82f6"),
        ("paren_std_HU",    "Parenchyma HU std (heterogeneity)", "#10b981"),
        ("lung_vol_mL",     "Lung volume (mL)", "#8b5cf6"),
    ]
    stage_labels = ["S0\nplain\nnonPH", "S1\ncontrast\nnonPH",
                    "S2\nPH\n<25", "S3\nPH\n25-35", "S4\nPH\n≥35"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (col, title, color) in zip(axes.flat, feats):
        if col not in df.columns: ax.set_visible(False); continue
        sub = df.dropna(subset=[col])
        rho, p = spearmanr(sub["stage"], sub[col])
        # Stage-wise mean ± 95% CI
        x_stages = []; y_means = []; y_ses = []; y_n = []
        for s in range(5):
            x = sub.loc[sub["stage"] == s, col].dropna().values
            if len(x) >= 3:
                x_stages.append(s); y_means.append(x.mean())
                y_ses.append(1.96 * x.std(ddof=1) / np.sqrt(len(x)))
                y_n.append(len(x))
            else:
                x_stages.append(s); y_means.append(np.nan); y_ses.append(0); y_n.append(0)
        # Plot mean line + CI band
        x_stages_a = np.array(x_stages, float)
        y_means_a = np.array(y_means)
        y_ses_a = np.array(y_ses)
        ax.errorbar(x_stages_a, y_means_a, yerr=y_ses_a, fmt="o-",
                    capsize=5, c=color, lw=2, markersize=8)
        # Per-stage scatter (jittered)
        rng = np.random.default_rng(42)
        for s in range(5):
            x = sub.loc[sub["stage"] == s, col].dropna().values
            jitter = rng.uniform(-0.12, 0.12, size=len(x))
            ax.scatter(s + jitter, x, c=color, alpha=0.15, s=12, zorder=1)
        # Annotate n per stage
        for s, n in zip(x_stages, y_n):
            if n > 0:
                ax.text(s, ax.get_ylim()[0], f"n={n}", ha="center", va="bottom",
                        fontsize=7, color="grey")
        ax.set_xticks(range(5))
        ax.set_xticklabels(stage_labels, fontsize=8)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(f"ρ={rho:+.3f}, p={p:.2g}", fontsize=10)
        ax.grid(alpha=0.3)

    fig.suptitle("R18.B/R19.F — Cross-sectional mPAP-staged endotype trajectory (legacy 282-cohort, n=261)\n"
                 "Stage 0 = plain-scan nonPH (mPAP default 0-10); Stage 4 = PH severe (real mPAP ≥35)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    out_path = OUT / "fig_r19_paper_mpap_trajectory.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
