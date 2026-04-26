"""C1_signature_severity — Phase C1: signature vs disease/mPAP severity.

Combines 4 tasks per spec (parallel via joblib):
  T1 within-contrast PH-vs-nonPH MWU + Cohen's d + Holm/FDR
  T2 mPAP Spearman + permutation null + mPAP bins (<20, 20-25, 25-35, ≥35)
  T3 borderline n=12 (mPAP 18-22) descriptive deep-dive
  T4 pruning curve N(d) log-log slope alpha (artery/vein) + sensitivity
       across multiple bin schemes

Cohort: B1 graph_signatures_patient_level.csv (n=290 × 172 features); use
within-contrast n=190 as PRIMARY for T1/T4; n=102 measured-mPAP for T2;
n=12 borderline for T3 (descriptive only per codex pass-1 guidance).

Output:
  outputs/supplementary/C1_signature_severity/signature_group_stats.csv
  outputs/supplementary/C1_signature_severity/mpap_correlation_table.csv
  outputs/supplementary/C1_signature_severity/borderline_deepdive.csv
  outputs/supplementary/C1_signature_severity/pruning_curve_results.json
  outputs/supplementary/C1_signature_severity/mpap_bin_trends.png
  outputs/supplementary/C1_signature_severity/top_signature_forest_plot.png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent
SIG = ROOT / "outputs" / "supplementary" / "B1_graph_signature" / "graph_signatures_patient_level.csv"
DICT = ROOT / "outputs" / "supplementary" / "B1_graph_signature" / "graph_signature_dictionary.json"
OUT = ROOT / "outputs" / "supplementary" / "C1_signature_severity"
OUT.mkdir(parents=True, exist_ok=True)

META_COLS = {"case_id", "label", "protocol", "is_contrast_only_subset",
             "measured_mpap", "measured_mpap_flag", "fold_id",
             "C1_all_available", "C2_within_contrast_only",
             "C3_borderline_mPAP_18_22", "C4_clear_low_high",
             "C5_early_COPD_no_PH_proxy"}


def cohen_d(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if len(a) < 2 or len(b) < 2: return float("nan")
    pooled = np.sqrt(((len(a)-1)*a.std(ddof=1)**2 + (len(b)-1)*b.std(ddof=1)**2)
                     / (len(a)+len(b)-2))
    if pooled == 0: return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def holm_bonferroni(pvals, alpha=0.05):
    p = np.asarray(pvals, dtype=float); m = len(p)
    order = np.argsort(p); sp = p[order]
    adj = np.minimum(np.maximum.accumulate(sp * (m - np.arange(m))), 1.0)
    out = np.empty(m); out[order] = adj
    return out < alpha, out


def fdr_bh(pvals, alpha=0.05):
    p = np.asarray(pvals, dtype=float)
    m_total = len(p)
    finite = np.isfinite(p)
    p_finite = p[finite]
    m = len(p_finite)
    out = np.full(m_total, np.nan)
    if m == 0:
        return out < alpha, out
    order = np.argsort(p_finite); sp = p_finite[order]
    adj = sp * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    finite_idx = np.where(finite)[0]
    out_finite = np.empty(m); out_finite[order] = adj
    out[finite_idx] = out_finite
    rejected = np.zeros(m_total, dtype=bool)
    rejected[finite_idx] = out_finite < alpha
    return rejected, out


# ================= T1 within-contrast PH vs nonPH =================
def t1_one_feature(name, ph_arr, nonph_arr):
    valid_p = ph_arr[~np.isnan(ph_arr)]
    valid_n = nonph_arr[~np.isnan(nonph_arr)]
    if len(valid_p) < 3 or len(valid_n) < 3:
        return {"feature": name, "skip": True}
    try:
        u, p = mannwhitneyu(valid_p, valid_n, alternative="two-sided")
    except Exception:
        return {"feature": name, "skip": True}
    d = cohen_d(valid_p, valid_n)
    # Bootstrap CI on Cohen's d
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(500):
        ip = rng.integers(0, len(valid_p), size=len(valid_p))
        ino = rng.integers(0, len(valid_n), size=len(valid_n))
        boots.append(cohen_d(valid_p[ip], valid_n[ino]))
    boots = np.array([b for b in boots if not np.isnan(b)])
    if len(boots) > 100:
        ci_lo, ci_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    else:
        ci_lo, ci_hi = float("nan"), float("nan")
    return {
        "feature": name, "n_PH": int(len(valid_p)), "n_nonPH": int(len(valid_n)),
        "PH_mean": float(valid_p.mean()), "nonPH_mean": float(valid_n.mean()),
        "cohen_d": float(d), "boot_d_ci_lo": ci_lo, "boot_d_ci_hi": ci_hi,
        "mwu_u": float(u), "p_raw": float(p),
    }


def task_t1(df, feat_cols):
    sub = df[df["is_contrast_only_subset"]].copy()
    print(f"T1 within-contrast n={len(sub)} (PH={int((sub.label==1).sum())}, "
          f"nonPH={int((sub.label==0).sum())})")
    rows = Parallel(n_jobs=4, backend="loky")(
        delayed(t1_one_feature)(f,
                                 sub.loc[sub.label==1, f].values.astype(float),
                                 sub.loc[sub.label==0, f].values.astype(float))
        for f in feat_cols
    )
    res = pd.DataFrame([r for r in rows if not r.get("skip", False)])
    if len(res) == 0: return res
    holm_sig, holm_p = holm_bonferroni(res["p_raw"].values)
    fdr_sig, fdr_p = fdr_bh(res["p_raw"].values)
    res["p_holm"] = holm_p; res["holm_sig"] = holm_sig
    res["p_fdr"] = fdr_p; res["fdr_sig"] = fdr_sig
    res["abs_d"] = res["cohen_d"].abs()
    res = res.sort_values("abs_d", ascending=False).reset_index(drop=True)
    return res


# ================= T2 mPAP Spearman =================
def t2_one_feature(name, mpap, vals, n_perm=500):
    valid = ~(np.isnan(mpap) | np.isnan(vals))
    x = mpap[valid]; y = vals[valid]
    if len(x) < 30: return {"feature": name, "skip": True}
    try:
        rho, p = spearmanr(x, y)
    except Exception:
        return {"feature": name, "skip": True}
    rng = np.random.default_rng(42)
    null_rhos = []
    for _ in range(n_perm):
        perm = rng.permutation(y)
        try:
            r, _ = spearmanr(x, perm); null_rhos.append(abs(r))
        except Exception: continue
    null_rhos = np.array(null_rhos)
    perm_p = float((null_rhos >= abs(rho)).mean()) if len(null_rhos) > 0 else float("nan")
    return {"feature": name, "n": int(len(x)), "spearman_rho": float(rho),
            "p_raw": float(p), "perm_p": perm_p}


def task_t2(df, feat_cols):
    sub = df[df["measured_mpap_flag"] & df["is_contrast_only_subset"]].copy()
    print(f"T2 measured-mPAP within-contrast n={len(sub)}")
    mpap_arr = sub["measured_mpap"].values.astype(float)
    rows = Parallel(n_jobs=4, backend="loky")(
        delayed(t2_one_feature)(f, mpap_arr, sub[f].values.astype(float))
        for f in feat_cols
    )
    res = pd.DataFrame([r for r in rows if not r.get("skip", False)])
    if len(res) == 0: return res
    holm_sig, holm_p = holm_bonferroni(res["p_raw"].values)
    fdr_sig, fdr_p = fdr_bh(res["p_raw"].values)
    res["p_holm"] = holm_p; res["holm_sig"] = holm_sig
    res["p_fdr"] = fdr_p; res["fdr_sig"] = fdr_sig
    res["abs_rho"] = res["spearman_rho"].abs()
    res = res.sort_values("abs_rho", ascending=False).reset_index(drop=True)
    return res


# ================= T3 borderline n=12 deep-dive =================
def task_t3(df, feat_cols):
    sub = df[df["C3_borderline_mPAP_18_22"]].copy()
    print(f"T3 borderline n={len(sub)} (PH={int((sub.label==1).sum())}, "
          f"nonPH={int((sub.label==0).sum())}) — DESCRIPTIVE ONLY (codex pass-1)")
    rows = []
    for f in feat_cols:
        vals = sub[f].values.astype(float)
        valid = ~np.isnan(vals)
        if valid.sum() < 5: continue
        rows.append({
            "feature": f,
            "n": int(valid.sum()),
            "borderline_mean": float(vals[valid].mean()),
            "borderline_std": float(vals[valid].std()),
            "borderline_median": float(np.median(vals[valid])),
        })
    return pd.DataFrame(rows).sort_values("feature")


# ================= T4 pruning curve =================
def task_t4(df, feat_cols):
    """Fit log-log N(d) ~ d^-alpha for artery + vein diameter percentiles within-contrast."""
    sub = df[df["is_contrast_only_subset"]].copy()
    out = {"n_within_contrast": int(len(sub)), "structures": {}}
    for struct in ("artery", "vein"):
        # Use diameter percentile features as log-binned diameter sample
        diam_cols = [f"{struct}_diam_p10", f"{struct}_diam_p25",
                     f"{struct}_diam_p50", f"{struct}_diam_p75",
                     f"{struct}_diam_p90"]
        diam_cols = [c for c in diam_cols if c in sub.columns]
        if len(diam_cols) < 3:
            out["structures"][struct] = {"skip": True, "reason": "insufficient diameter percentiles"}
            continue
        # Per-patient: count of nodes at each percentile (inverse CDF interpretation:
        # at p10 we have approximately 90% of edges with d>=p10; treat each percentile as a bin)
        # Sensitivity: 3 binning schemes
        results_per_scheme = {}
        for scheme_name, percs in [("p10_p25_p50_p75_p90", [10, 25, 50, 75, 90]),
                                     ("p10_p50_p90", [10, 50, 90]),
                                     ("p25_p75", [25, 75])]:
            cols_used = [f"{struct}_diam_p{p}" for p in percs if f"{struct}_diam_p{p}" in sub.columns]
            if len(cols_used) < 2: continue
            # Per-patient log-log slope on (mean diameter at each bin) vs (1 - perc/100 = surviving fraction proxy)
            slopes = []
            for _, row in sub.iterrows():
                diams = np.array([row[c] for c in cols_used], dtype=float)
                if not np.isfinite(diams).all() or (diams <= 0).any(): continue
                surviving = np.array([1.0 - int(c.split("_p")[-1]) / 100.0 for c in cols_used])
                # avoid 0
                surviving = np.clip(surviving, 0.01, 1.0)
                try:
                    log_d = np.log(diams); log_s = np.log(surviving)
                    A = np.vstack([log_d, np.ones_like(log_d)]).T
                    slope, _ = np.linalg.lstsq(A, log_s, rcond=None)[0]
                    # N(d) ~ d^-alpha → log N = -alpha * log d + c → alpha = -slope
                    alpha = -float(slope)
                    slopes.append(alpha)
                except Exception:
                    continue
            if slopes:
                results_per_scheme[scheme_name] = {
                    "n_patients": len(slopes),
                    "alpha_mean": float(np.mean(slopes)),
                    "alpha_std": float(np.std(slopes)),
                    "alpha_median": float(np.median(slopes)),
                }
        out["structures"][struct] = results_per_scheme
    return out


def main():
    df = pd.read_csv(SIG)
    feat_cols = [c for c in df.columns if c not in META_COLS
                 and pd.api.types.is_numeric_dtype(df[c])]
    print(f"loaded B1 panel: n={len(df)}, features={len(feat_cols)}")

    # Run T1, T2, T3, T4 (T1/T2 use joblib internally; T3/T4 are fast)
    print("\n=== T1 within-contrast PH-vs-nonPH ===")
    t1 = task_t1(df, feat_cols)
    if len(t1):
        t1.to_csv(OUT / "signature_group_stats.csv", index=False)
        print(f"T1 top-5 by |d|:")
        print(t1.head(5)[["feature", "cohen_d", "boot_d_ci_lo", "boot_d_ci_hi",
                          "p_raw", "p_holm", "p_fdr"]].to_string())

    print("\n=== T2 measured-mPAP Spearman with permutation null ===")
    t2 = task_t2(df, feat_cols)
    if len(t2):
        t2.to_csv(OUT / "mpap_correlation_table.csv", index=False)
        print(f"T2 top-5 by |ρ|:")
        print(t2.head(5)[["feature", "spearman_rho", "p_raw", "perm_p",
                          "p_holm", "p_fdr"]].to_string())

    print("\n=== T3 borderline n=12 descriptive ===")
    t3 = task_t3(df, feat_cols)
    if len(t3):
        t3.to_csv(OUT / "borderline_deepdive.csv", index=False)

    print("\n=== T4 pruning curve alpha ===")
    t4 = task_t4(df, feat_cols)
    (OUT / "pruning_curve_results.json").write_text(
        json.dumps(t4, indent=2), encoding="utf-8")
    print(json.dumps(t4, indent=2))

    # ===== Figure 1: mPAP-bin trends for top T2 features =====
    if len(t2) >= 1:
        sub_mp = df[df["measured_mpap_flag"] & df["is_contrast_only_subset"]].copy()
        bins = [(0, 20), (20, 25), (25, 35), (35, 100)]
        bin_labels = ["<20", "20-25", "25-35", "≥35"]
        top6 = t2.head(6)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, (_, r) in zip(axes.flat, top6.iterrows()):
            f = r["feature"]
            means = []; lo = []; hi = []; ns = []
            for blo, bhi in bins:
                m = (sub_mp["measured_mpap"] >= blo) & (sub_mp["measured_mpap"] < bhi)
                vals = sub_mp.loc[m, f].dropna().values
                if len(vals) >= 3:
                    means.append(float(vals.mean()))
                    sem = float(vals.std() / np.sqrt(len(vals)))
                    lo.append(means[-1] - 1.96 * sem)
                    hi.append(means[-1] + 1.96 * sem)
                    ns.append(len(vals))
                else:
                    means.append(np.nan); lo.append(np.nan); hi.append(np.nan); ns.append(len(vals))
            xs = list(range(4))
            ax.errorbar(xs, means, yerr=[[m - l for m, l in zip(means, lo)],
                                          [h - m for m, h in zip(means, hi)]],
                         fmt="o-", c="#3b82f6", lw=2, markersize=8)
            ax.set_xticks(xs); ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(bin_labels, ns)],
                                                    fontsize=8)
            ax.set_xlabel("mPAP bin (mmHg)")
            ax.set_ylabel(f[:25])
            ax.set_title(f"{f[:30]}\nρ={r['spearman_rho']:+.3f} perm-p={r['perm_p']:.3g}", fontsize=9)
            ax.grid(alpha=0.3)
        fig.suptitle("C1 — Top-6 mPAP-correlated features by bin (within-contrast n_resolved=102)\n"
                     "permutation null p, Holm/FDR shown in mpap_correlation_table.csv",
                     fontsize=12, y=1.01)
        plt.tight_layout()
        plt.savefig(OUT / "mpap_bin_trends.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ===== Figure 2: forest plot top-20 by Cohen's d =====
    if len(t1) >= 1:
        top20 = t1.head(20).iloc[::-1]
        fig, ax = plt.subplots(figsize=(10, 8))
        for i, (_, r) in enumerate(top20.iterrows()):
            color = "#ef4444" if r["holm_sig"] else ("#f59e0b" if r["fdr_sig"] else "#94a3b8")
            ax.errorbar(r["cohen_d"], i,
                         xerr=[[r["cohen_d"] - r["boot_d_ci_lo"]],
                               [r["boot_d_ci_hi"] - r["cohen_d"]]],
                         fmt="o", c=color, capsize=4, lw=2, markersize=8)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels(top20["feature"].values, fontsize=8)
        ax.set_xlabel("Cohen's d (PH − nonPH within-contrast)\nRed = Holm-sig α=0.05; Orange = FDR-sig; Grey = NS")
        ax.set_title(f"C1 — Top-20 within-contrast PH-vs-nonPH features\n"
                     f"n_PH={int((df[df.is_contrast_only_subset].label==1).sum())}, "
                     f"n_nonPH={int((df[df.is_contrast_only_subset].label==0).sum())}",
                     fontsize=11)
        ax.grid(alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(OUT / "top_signature_forest_plot.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"\nsaved all C1 outputs to {OUT}")


if __name__ == "__main__":
    main()
