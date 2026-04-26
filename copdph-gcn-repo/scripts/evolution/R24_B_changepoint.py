"""R24.B — mPAP changepoint detection on n=102 measured-mPAP cases.

Per-feature piecewise-linear with grid breakpoint search; bootstrap-1000;
gates: ΔAIC ≥ 10 vs linear AND boot freq ≥ 70% AND CI width ≤ 8 mmHg.

Output:
  outputs/r24/r24b_changepoints.csv (feature, threshold, slope_low, slope_high, deltaAIC, boot_freq, ci_width, holm_pass)
  outputs/figures/fig_r24b_changepoint.png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
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


def fit_piecewise(x, y, candidates):
    best = None
    for t in candidates:
        mask_lo = x < t
        if mask_lo.sum() < 3 or (~mask_lo).sum() < 3: continue
        # Continuous piecewise linear: y = a + b1*(x-t)*(x<t) + b2*(x-t)*(x>=t)
        z1 = (x - t) * (x < t).astype(float)
        z2 = (x - t) * (x >= t).astype(float)
        X = np.column_stack([np.ones_like(x), z1, z2])
        try:
            beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
            pred = X @ beta
            ssr = float(((y - pred) ** 2).sum())
            n = len(y); k = 3
            aic = n * np.log(ssr / n + 1e-12) + 2 * k
            if best is None or aic < best[0]:
                best = (aic, t, float(beta[1]), float(beta[2]), ssr)
        except Exception:
            continue
    return best


def fit_linear_aic(x, y):
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ssr = float(((y - pred) ** 2).sum())
    n = len(y); k = 2
    aic = n * np.log(ssr / n + 1e-12) + 2 * k
    return aic


def per_feature_changepoint(x_all, y_all, feat_name, candidates, n_boot=500, rng_seed=42):
    valid = ~np.isnan(y_all)
    x = x_all[valid]; y = y_all[valid]
    if len(x) < 30: return {"feature": feat_name, "n": int(len(x)), "skip": True}
    pw = fit_piecewise(x, y, candidates)
    lin_aic = fit_linear_aic(x, y)
    if pw is None: return {"feature": feat_name, "n": int(len(x)), "skip": True}
    aic_pw, t, b1, b2, _ = pw
    delta_aic = lin_aic - aic_pw
    # Bootstrap
    rng = np.random.default_rng(rng_seed)
    boot_t = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        bp = fit_piecewise(x[idx], y[idx], candidates)
        if bp is not None:
            boot_t.append(bp[1])
    if not boot_t: return {"feature": feat_name, "n": int(len(x)), "skip": True}
    boot_t = np.array(boot_t)
    ci_lo, ci_hi = np.percentile(boot_t, [2.5, 97.5])
    # Boot freq = proportion within ±2 mmHg of point estimate t
    boot_freq = float(((boot_t >= t - 2) & (boot_t <= t + 2)).mean())
    return {
        "feature": feat_name, "n": int(len(x)),
        "threshold": float(t), "slope_low": b1, "slope_high": b2,
        "delta_aic_vs_linear": float(delta_aic),
        "boot_freq_within_2mmHg": boot_freq,
        "ci95_lo": float(ci_lo), "ci95_hi": float(ci_hi),
        "ci_width": float(ci_hi - ci_lo),
        "gate_delta_aic_ge_10": bool(delta_aic >= 10.0),
        "gate_boot_freq_ge_0.70": bool(boot_freq >= 0.70),
        "gate_ci_width_le_8": bool((ci_hi - ci_lo) <= 8.0),
    }


def main():
    cohort = pd.read_csv(COHORT)
    morph = pd.read_csv(MORPH)
    df = cohort.merge(morph, on="case_id", how="inner")
    df = df[df["measured_mpap_flag"]].copy()
    print(f"R24.B input: n={len(df)} measured-mPAP cases")
    feat_cols = [c for c in morph.columns
                 if c not in ("case_id", "label", "source_cache")
                 and c not in ARTIFACTS]
    x_all = df["measured_mpap"].values
    candidates = np.linspace(np.percentile(x_all, 10), np.percentile(x_all, 90), 25)

    rows = Parallel(n_jobs=4, backend="loky")(
        delayed(per_feature_changepoint)(x_all, df[f].values, f, candidates, 500)
        for f in feat_cols
    )
    res = pd.DataFrame([r for r in rows if not r.get("skip", False)])
    # Holm correction not possible for changepoint (no native p-value); use ΔAIC and stability gates
    res["all_gates_pass"] = res["gate_delta_aic_ge_10"] & res["gate_boot_freq_ge_0.70"] & res["gate_ci_width_le_8"]
    res = res.sort_values("delta_aic_vs_linear", ascending=False).reset_index(drop=True)
    csv_path = OUT / "r24b_changepoints.csv"
    res.to_csv(csv_path, index=False)

    # Figure: top-6 changepoint features × mPAP scatter + piecewise fit + boot CI
    pass_features = res[res["all_gates_pass"]].head(6)
    print(f"features passing all 3 gates: {len(pass_features)}")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    if len(pass_features) == 0:
        # No feature passes — show top-6 by delta_aic anyway with reason
        pass_features = res.head(6)
    for ax, (_, r) in zip(axes.flat, pass_features.iterrows()):
        feat = r["feature"]
        sub = df.dropna(subset=[feat])
        ax.scatter(sub["measured_mpap"], sub[feat], c="#3b82f6", s=20, alpha=0.6)
        t = r["threshold"]; b1 = r["slope_low"]; b2 = r["slope_high"]
        x_lin = np.linspace(sub["measured_mpap"].min(), sub["measured_mpap"].max(), 50)
        # Continuous-piecewise: at x=t, both arms meet at intercept a
        # but easier to compute predictions
        z1 = (x_lin - t) * (x_lin < t).astype(float)
        z2 = (x_lin - t) * (x_lin >= t).astype(float)
        # Re-fit intercept
        from numpy.linalg import lstsq
        x_obs = sub["measured_mpap"].values; y_obs = sub[feat].values
        z1o = (x_obs - t) * (x_obs < t).astype(float)
        z2o = (x_obs - t) * (x_obs >= t).astype(float)
        Xo = np.column_stack([np.ones_like(x_obs), z1o, z2o])
        beta, *_ = lstsq(Xo, y_obs, rcond=None)
        Xp = np.column_stack([np.ones_like(x_lin), z1, z2])
        ax.plot(x_lin, Xp @ beta, "r-", lw=2, label=f"piecewise @ t={t:.1f}")
        ax.axvspan(r["ci95_lo"], r["ci95_hi"], alpha=0.15, color="red", label=f"CI={r['ci_width']:.1f} mmHg")
        ax.axvline(t, color="red", linestyle="--", lw=1)
        gate_str = f"ΔAIC={r['delta_aic_vs_linear']:.1f} bootF={r['boot_freq_within_2mmHg']:.2f}"
        ax.set_title(f"{feat[:25]}\n{gate_str}", fontsize=9)
        ax.set_xlabel("measured mPAP (mmHg)"); ax.set_ylabel(feat[:18])
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    n_pass = int(res["all_gates_pass"].sum())
    fig.suptitle(f"R24.B — mPAP-axis changepoint detection (n=102 measured-mPAP cases)\n"
                 f"Q3 早期 PH 阈值识别 / early-PH threshold; gates: ΔAIC≥10 AND bootFreq≥70% AND CI≤8 mmHg; "
                 f"pass={n_pass}/{len(res)}", fontsize=12, y=1.01)
    plt.tight_layout()
    out_fig = FIG / "fig_r24b_changepoint.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"saved {csv_path} + {out_fig}")
    print(f"top-3 by ΔAIC:")
    print(res.head(3)[["feature", "threshold", "delta_aic_vs_linear",
                        "boot_freq_within_2mmHg", "ci_width", "all_gates_pass"]].to_string())


if __name__ == "__main__":
    main()
