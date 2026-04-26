"""R24.D — Structure ordering along inferred severity axis (NOT 'leadership').

Bootstrap-1000 structure-onset position along R24.A pseudotime, where each
structure's mean-z-score first exits ±τ SD. Sensitivity τ ∈ {0.25, 0.5, 0.75}.
Cross-check with R24.B mPAP changepoints.

Output:
  outputs/r24/r24d_structure_onset.csv
  outputs/figures/fig_r24d_structure_ordering.png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent
COHORT = ROOT / "outputs" / "r24" / "cohort_locked_table.csv"
MORPH = ROOT / "outputs" / "r20" / "morph_unified301.csv"
PSEUDO = ROOT / "outputs" / "r24" / "r24a_pseudotime_within_contrast.csv"
OUT = ROOT / "outputs" / "r24"
FIG = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

ARTIFACTS = {"airway_n_terminals", "airway_term_per_node",
             "artery_lap_eig0", "artery_n_terminals", "artery_term_per_node",
             "vein_lap_eig0", "vein_n_terminals", "vein_term_per_node"}


def structure_onset(pt, struct_z, tau):
    """Find pseudotime position where struct mean z-score first exits ±tau."""
    order = np.argsort(pt)
    pt_s = pt[order]; z_s = struct_z[order]
    # Sliding-window mean z (window=10)
    w = 10
    if len(z_s) < w: return float("nan")
    means = np.array([z_s[max(0, i-w//2):i+w//2+1].mean() for i in range(len(z_s))])
    # First i where |means[i]| > tau
    exits = np.where(np.abs(means) > tau)[0]
    if len(exits) == 0: return float("nan")
    return float(pt_s[exits[0]])


def main():
    cohort = pd.read_csv(COHORT)
    morph = pd.read_csv(MORPH).drop(columns=["label"], errors="ignore")
    pseudo = pd.read_csv(PSEUDO)
    df = pseudo.merge(morph, on="case_id", how="inner")
    print(f"R24.D within-contrast n={len(df)}")

    # Group features by structure
    groups = {
        "artery": [c for c in df.columns if c.startswith("artery_") and c not in ARTIFACTS],
        "vein": [c for c in df.columns if c.startswith("vein_") and c not in ARTIFACTS],
        "airway": [c for c in df.columns if c.startswith("airway_") and c not in ARTIFACTS],
    }
    # paren_ from lung features absent in morph_unified301 — skip parenchyma here
    print({k: len(v) for k, v in groups.items()})

    pt = df["pseudotime"].values
    rng = np.random.default_rng(42)
    taus = [0.25, 0.5, 0.75]
    rows = []
    for tau in taus:
        for struct, feats in groups.items():
            if not feats: continue
            X = df[feats].fillna(0).values.astype(float)
            X_z = RobustScaler().fit_transform(X)
            mean_z = X_z.mean(axis=1)  # mean across features per case
            point = structure_onset(pt, mean_z, tau)
            # Bootstrap
            boots = []
            for _ in range(1000):
                idx = rng.integers(0, len(pt), size=len(pt))
                bp = structure_onset(pt[idx], mean_z[idx], tau)
                if not np.isnan(bp): boots.append(bp)
            boots = np.array(boots) if boots else np.array([point])
            rows.append({
                "tau": tau, "structure": struct,
                "onset_pseudotime": point,
                "boot_lo": float(np.percentile(boots, 2.5)),
                "boot_hi": float(np.percentile(boots, 97.5)),
                "n_features_in_group": len(feats),
            })
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "r24d_structure_onset.csv", index=False)

    # Figure: ladder plot — structure onset vs tau
    fig, ax = plt.subplots(figsize=(10, 6))
    structures = ["artery", "vein", "airway"]
    colors = {"artery": "#ef4444", "vein": "#3b82f6", "airway": "#10b981"}
    width = 0.18
    x_pos = np.arange(len(taus))
    for i, struct in enumerate(structures):
        sub = res[res["structure"] == struct]
        offsets = (i - 1) * width
        for j, tau in enumerate(taus):
            r = sub[sub["tau"] == tau]
            if len(r):
                r = r.iloc[0]
                ax.errorbar(x_pos[j] + offsets, r["onset_pseudotime"],
                             yerr=[[r["onset_pseudotime"] - r["boot_lo"]],
                                   [r["boot_hi"] - r["onset_pseudotime"]]],
                             fmt="o", c=colors[struct], capsize=5, lw=2,
                             markersize=10,
                             label=struct if j == 0 else None)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"τ={t}" for t in taus])
    ax.set_xlabel("z-score exit threshold (sensitivity sweep)")
    ax.set_ylabel("Pseudotime position of first onset\n(bootstrap-1000 95% CI)")
    ax.set_title("R24.D — Structure ordering along inferred severity axis\n"
                 "Q2 辅助 vs 主导 / which structure changes earliest along pseudotime\n"
                 "Cross-sectional ordering — NOT temporal precedence")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_fig = FIG / "fig_r24d_structure_ordering.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(res.to_string())
    print(f"saved {out_fig}")


if __name__ == "__main__":
    main()
