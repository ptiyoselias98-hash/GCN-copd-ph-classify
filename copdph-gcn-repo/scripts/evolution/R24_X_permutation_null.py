"""R24.X — Stratified permutation null falsification.

1000 permutations of mPAP labels within (protocol × fold) strata; recompute
the real statistic on permuted mPAP. Real-stat must exceed 99th percentile
of null.

Targets:
  - R24.A pseudotime ρ_mPAP (within-contrast)
  - R24.G OOF severity ρ_mPAP (SSL d=32)

Output:
  outputs/r24/r24x_permutation_null.csv
  outputs/figures/fig_r24x_permutation_null.png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent
COHORT = ROOT / "outputs" / "r24" / "cohort_locked_table.csv"
PSEUDO = ROOT / "outputs" / "r24" / "r24a_pseudotime_within_contrast.csv"
SSL = ROOT / "outputs" / "r24" / "r24g_ssl_d32_oof_severity.csv"
OUT = ROOT / "outputs" / "r24"
FIG = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)


def stratified_permute_mpap(mpap, fold, rng):
    """Permute mPAP within each fold (since cohort is within-contrast, no protocol stratum)."""
    out = mpap.copy()
    for f in np.unique(fold):
        mask = (fold == f) & ~np.isnan(out)
        if mask.sum() <= 1: continue
        idx = np.where(mask)[0]
        perm = rng.permutation(idx)
        out[idx] = mpap[perm]
    return out


def one_perm(score, mpap, fold, seed):
    rng = np.random.default_rng(seed)
    perm_mpap = stratified_permute_mpap(mpap, fold, rng)
    valid = ~np.isnan(perm_mpap) & ~np.isnan(score)
    if valid.sum() < 5: return float("nan")
    rho, _ = spearmanr(score[valid], perm_mpap[valid])
    return float(rho)


def main():
    cohort = pd.read_csv(COHORT)
    pseudo = pd.read_csv(PSEUDO)
    ssl = pd.read_csv(SSL)
    print("loaded cohort + pseudotime + SSL severity")

    # Merge fold info into both
    pt_df = pseudo.merge(cohort[["case_id", "fold_id"]], on="case_id", how="inner")
    ssl_df = ssl.merge(cohort[["case_id", "fold_id"]], on="case_id", how="inner")

    targets = {
        "R24.A_pseudotime": (pt_df["pseudotime"].values,
                              pt_df["measured_mpap"].values,
                              pt_df["fold_id"].values),
        "R24.G_ssl_severity": (ssl_df["severity_pct"].values,
                                ssl_df["measured_mpap"].values,
                                ssl_df["fold_id"].values),
    }

    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for i, (name, (score, mpap, fold)) in enumerate(targets.items()):
        valid = ~np.isnan(mpap) & ~np.isnan(score)
        if valid.sum() < 5: continue
        real_rho, _ = spearmanr(score[valid], mpap[valid])
        # 1000 permutations
        null_rhos = Parallel(n_jobs=4, backend="loky")(
            delayed(one_perm)(score, mpap, fold, s) for s in range(1000)
        )
        null_rhos = np.array([v for v in null_rhos if not np.isnan(v)])
        # Use absolute value for two-sided gate
        real_abs = abs(real_rho)
        null_abs = np.abs(null_rhos)
        pct_99 = float(np.percentile(null_abs, 99))
        gate_pass = bool(real_abs > pct_99)
        rows.append({
            "target": name, "real_rho": float(real_rho),
            "real_abs_rho": float(real_abs),
            "null_99_pct_abs": pct_99,
            "n_perms": len(null_rhos),
            "gate_real_above_99_null": gate_pass,
        })
        # Hist
        ax = axes[i]
        ax.hist(null_rhos, bins=40, color="#94a3b8", alpha=0.7, edgecolor="black")
        ax.axvline(real_rho, color="red", lw=2, label=f"real ρ = {real_rho:+.3f}")
        ax.axvline(pct_99, color="orange", linestyle="--", lw=1.5,
                    label=f"99th pct |null| = {pct_99:.3f}")
        ax.axvline(-pct_99, color="orange", linestyle="--", lw=1.5)
        ax.set_xlabel(f"Spearman ρ ({name} score vs permuted mPAP)")
        ax.set_ylabel("Frequency")
        gate_str = "PASS" if gate_pass else "FAIL"
        ax.set_title(f"{name}\nGate {gate_str}: |real|={real_abs:.3f} vs 99-pct null={pct_99:.3f}")
        ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle("R24.X — Stratified permutation null falsification (1000 perms within fold)\n"
                 "Tests if real statistic could arise from random fold-stratified mPAP shuffling",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out_fig = FIG / "fig_r24x_permutation_null.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close()

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "r24x_permutation_null.csv", index=False)
    print(res.to_string())
    print(f"saved {out_fig}")


if __name__ == "__main__":
    main()
