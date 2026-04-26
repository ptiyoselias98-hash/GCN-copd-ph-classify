"""R24.F — Evolution feature panel: rank by |Spearman with measured-mPAP| with bootstrap optimism CI.

Closes Q4(a): systematic feature panel ranked by mPAP-correlation. Honest "same-sample
association ranking" labelling per round-2 codex feedback.

Output:
  outputs/r24/r24f_feature_rank.csv (feature, rho, p, p_holm, boot_lo, boot_hi, holm_pass)
  outputs/figures/fig_r24f_feature_panel.png  (forest plot top-20)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent
COHORT = ROOT / "outputs" / "r24" / "cohort_locked_table.csv"
MORPH = ROOT / "outputs" / "r20" / "morph_unified301.csv"
OUT = ROOT / "outputs" / "r24"
FIG = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

ARTIFACTS = {"airway_n_terminals", "airway_term_per_node",
             "artery_lap_eig0", "artery_n_terminals", "artery_term_per_node",
             "vein_lap_eig0", "vein_n_terminals", "vein_term_per_node"}


def holm(pvals, alpha=0.05):
    p = np.asarray(pvals, dtype=float); m = len(p)
    order = np.argsort(p); sp = p[order]
    adj = np.minimum(np.maximum.accumulate(sp * (m - np.arange(m))), 1.0)
    out = np.empty(m); out[order] = adj
    return out


def per_feature(x, y_feat, n_boot=500, seed=42):
    valid = ~(np.isnan(x) | np.isnan(y_feat))
    x_v = x[valid]; y_v = y_feat[valid]
    if len(x_v) < 30: return None
    rho, p = spearmanr(x_v, y_v)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x_v), size=len(x_v))
        try:
            b = spearmanr(x_v[idx], y_v[idx])[0]
            if not np.isnan(b): boots.append(b)
        except Exception:
            continue
    boots = np.array(boots) if boots else np.array([rho])
    return {"rho": float(rho), "p": float(p),
            "boot_lo": float(np.percentile(boots, 2.5)),
            "boot_hi": float(np.percentile(boots, 97.5)),
            "n": int(len(x_v))}


def main():
    cohort = pd.read_csv(COHORT)
    morph = pd.read_csv(MORPH)
    df = cohort.merge(morph, on="case_id", how="inner")
    df = df[df["measured_mpap_flag"]].copy()
    print(f"R24.F input: n={len(df)} measured-mPAP cases")

    feat_cols = [c for c in morph.columns
                 if c not in ("case_id", "label", "source_cache")
                 and c not in ARTIFACTS]
    x = df["measured_mpap"].values
    rows = Parallel(n_jobs=4, backend="loky")(
        delayed(per_feature)(x, df[f].values, 500) for f in feat_cols
    )
    res = pd.DataFrame([{"feature": f, **(r if r else {})} for f, r in zip(feat_cols, rows)])
    res = res.dropna(subset=["rho"]).copy()
    res["abs_rho"] = res["rho"].abs()
    res["p_holm"] = holm(res["p"].values)
    res["holm_significant"] = res["p_holm"] < 0.05
    res = res.sort_values("abs_rho", ascending=False).reset_index(drop=True)
    csv_path = OUT / "r24f_feature_rank.csv"
    res.to_csv(csv_path, index=False)
    print(f"Holm-significant: {int(res['holm_significant'].sum())}/{len(res)}")

    # Forest plot top-20
    top = res.head(20).iloc[::-1]  # reverse for ascending plot
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(top))
    colors = ["#ef4444" if v else "#94a3b8" for v in top["holm_significant"]]
    for i, (_, r) in enumerate(top.iterrows()):
        ax.errorbar(r["rho"], i, xerr=[[r["rho"] - r["boot_lo"]], [r["boot_hi"] - r["rho"]]],
                    fmt="o", c=colors[i], capsize=4, lw=2, markersize=8)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["feature"].values, fontsize=8)
    ax.set_xlabel("Spearman ρ with measured mPAP (n=102)\nbootstrap-500 95% CI; same-sample association ranking — interpret with optimism awareness")
    ax.set_title(f"R24.F — Evolution feature panel (top-20 by |ρ_mPAP|)\n"
                 f"Q4(a) systematic quantitative panel; Holm-sig (red) {int(res['holm_significant'].sum())}/{len(res)}",
                 fontsize=11)
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    out_fig = FIG / "fig_r24f_feature_panel.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {csv_path} + {out_fig}")
    print(f"top-5: {res.head(5)[['feature','rho','p_holm','holm_significant']].to_string()}")


if __name__ == "__main__":
    main()
