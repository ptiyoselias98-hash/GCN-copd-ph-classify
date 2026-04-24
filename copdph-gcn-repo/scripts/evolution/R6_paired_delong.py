"""R6.1 — Paired DeLong arm_c (vessel + lung globals) − arm_a (vessel only) on contrast-only.

PRIMARY ENDPOINT for the paper. Both arms trained with identical config
except arm_c adds 13 lung scalar globals; both use gcn_only mode and
the same 189 case set (no radiomics filter).

Reuses fast_delong from R5_delong.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from R5_delong import fast_delong, load_arm  # noqa: E402

ROOT = Path(__file__).parent.parent.parent
ARM_A = ROOT / "outputs" / "r5" / "arm_a_contrast_only_probs.json"
ARM_C = ROOT / "outputs" / "r5" / "arm_c_contrast_only_probs.json"
OUT_MD = ROOT / "outputs" / "r6" / "R6_paired_delong_primary.md"
OUT_JSON = ROOT / "outputs" / "r6" / "R6_paired_delong_primary.json"
OUT_MD.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    yt_a, ys_a, mode_a = load_arm(ARM_A, mode="gcn_only")
    yt_c, ys_c, mode_c = load_arm(ARM_C, mode="gcn_only")
    assert len(yt_a) == len(yt_c) == 189, f"{len(yt_a)} vs {len(yt_c)}"
    assert (yt_a == yt_c).all(), "label vector mismatch — case order differs"

    auc_a, auc_c, var_a, var_c, cov, z, p = fast_delong(ys_c, ys_a, yt_a)
    diff = auc_c - auc_a
    se = np.sqrt(var_a + var_c - 2 * cov) if (var_a + var_c - 2 * cov) > 0 else 0.0
    ci_lo, ci_hi = diff - 1.96 * se, diff + 1.96 * se

    # Bootstrap CI on Δ
    rng = np.random.default_rng(20260424)
    boot = np.empty(5000)
    from sklearn.metrics import roc_auc_score
    for i in range(5000):
        idx = rng.integers(0, len(yt_a), len(yt_a))
        try:
            boot[i] = roc_auc_score(yt_a[idx], ys_c[idx]) - roc_auc_score(yt_a[idx], ys_a[idx])
        except ValueError:
            boot[i] = np.nan
    boot = boot[~np.isnan(boot)]
    bs_lo, bs_hi = np.percentile(boot, [2.5, 97.5])

    out = {
        "primary_endpoint": "arm_c − arm_a contrast-only gcn_only AUC (paired DeLong)",
        "case_set": "contrast-only 189 (163 PH + 26 nonPH), shared between arms",
        "n_cases": int(len(yt_a)),
        "AUC_arm_a_vessel_only": auc_a,
        "AUC_arm_c_vessel_plus_lung": auc_c,
        "delta_AUC": diff,
        "delong_z": z,
        "delong_p_two_sided": p,
        "delong_ci95": [float(ci_lo), float(ci_hi)],
        "bootstrap_ci95": [float(bs_lo), float(bs_hi)],
        "covariance_arms": cov,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    sig_marker = "EXCLUDES 0 — significant" if (ci_lo > 0 or ci_hi < 0) else "INCLUDES 0 — NOT significant"
    lines = [
        "# R6.1 — Paired DeLong on arm_c − arm_a contrast-only (PRIMARY ENDPOINT)",
        "",
        f"Same 189 contrast-only cases (163 PH + 26 nonPH), gcn_only mode, ensembled across 5 folds × 3 repeats.",
        "",
        f"- AUC arm_a (vessel-only): **{auc_a:.4f}**",
        f"- AUC arm_c (vessel + 13 lung scalar globals): **{auc_c:.4f}**",
        f"- Δ AUC (arm_c − arm_a): **{diff:+.4f}**",
        f"- DeLong 95% CI on Δ: **[{ci_lo:+.4f}, {ci_hi:+.4f}]**  ({sig_marker})",
        f"- DeLong z = {z:.3f}, **p two-sided = {p:.4g}**",
        f"- Bootstrap 95% CI on Δ (n=5000): **[{bs_lo:+.4f}, {bs_hi:+.4f}]**",
        "",
        "## Interpretation",
        "",
        "- This is the W6 case-level paired confirmatory test the Round-4/5 reviewers required.",
        "- Single pre-specified endpoint, no multiplicity correction needed.",
        f"- arm_c lung-feature contribution under protocol balancing: {('NOT statistically significant' if (ci_lo <= 0 <= ci_hi) else 'STATISTICALLY SIGNIFICANT')}.",
        "- Conclusion is consistent with the Round-2 §13.5 retraction of the lung-feature claim.",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
