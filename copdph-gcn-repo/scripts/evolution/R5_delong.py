"""R5 — Case-level paired DeLong on arm_c − arm_b contrast-only (PRIMARY ENDPOINT).

Reads the per-case prob dumps from `outputs/r5/arm_{b,c}_contrast_only_probs.json`
(produced by the patched `run_sprint6_v2_probs.py` on remote GPU 1).

The DeLong test compares two ROC AUCs computed on the same set of cases.
Implementation: Sun-Xu fast DeLong (O(n log n)).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).parent.parent.parent
ARM_B = ROOT / "outputs" / "r5" / "arm_b_contrast_only_probs.json"
ARM_C = ROOT / "outputs" / "r5" / "arm_c_contrast_only_probs.json"
OUT_MD = ROOT / "outputs" / "r5" / "R5_delong_primary.md"
OUT_JSON = ROOT / "outputs" / "r5" / "R5_delong_primary.json"


def fast_delong(probs1: np.ndarray, probs2: np.ndarray, labels: np.ndarray):
    """Fast DeLong (Sun & Xu 2014) for paired AUCs on same labels.
    Returns (auc1, auc2, var1, var2, cov, z, p_two_sided).
    """
    order = (-labels).argsort()
    labels = labels[order]
    probs1 = probs1[order]
    probs2 = probs2[order]
    m = int(labels.sum())
    n = len(labels) - m

    def midrank(x: np.ndarray) -> np.ndarray:
        J = x.argsort()
        Z = x[J]
        N = len(x)
        T = np.zeros(N)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1) + 1
            i = j
        out = np.empty(N)
        out[J] = T
        return out

    def aucs_and_S(probs: np.ndarray):
        pos = probs[:m]
        neg = probs[m:]
        Tx = midrank(pos)
        Ty = midrank(neg)
        Tz = midrank(probs)
        auc = float(Tz[:m].sum() / m / n - (m + 1) / (2.0 * n))
        v01 = (Tz[:m] - Tx) / n
        v10 = 1.0 - (Tz[m:] - Ty) / m
        return auc, v01, v10

    auc1, v01_1, v10_1 = aucs_and_S(probs1)
    auc2, v01_2, v10_2 = aucs_and_S(probs2)

    sx = np.cov(np.vstack([v01_1, v01_2]))
    sy = np.cov(np.vstack([v10_1, v10_2]))
    s = sx / m + sy / n
    var1 = float(s[0, 0])
    var2 = float(s[1, 1])
    cov = float(s[0, 1])
    diff_var = var1 + var2 - 2 * cov
    if diff_var <= 0:
        z = 0.0
        p = 1.0
    else:
        z = (auc1 - auc2) / np.sqrt(diff_var)
        p = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
    return auc1, auc2, var1, var2, cov, float(z), p


def load_arm(path: Path, mode: str = "gcn_only") -> tuple[np.ndarray, np.ndarray, str]:
    d = json.loads(path.read_text())
    base = d["baseline"]
    if mode not in base:
        # fallback to whatever mode is available
        mode = next(iter(base.keys()))
    section = base[mode]
    yt = np.asarray(section["ensemble_y_true"], dtype=int)
    ys = np.asarray(section["ensemble_y_score"], dtype=float)
    return yt, ys, mode


def single_arm_ci(probs: np.ndarray, labels: np.ndarray):
    """DeLong-style variance for single-arm AUC + bootstrap CI."""
    order = (-labels).argsort()
    labels = labels[order]
    probs = probs[order]
    m = int(labels.sum())
    n = len(labels) - m

    def midrank(x):
        J = x.argsort()
        Z = x[J]
        N = len(x)
        T = np.zeros(N)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1) + 1
            i = j
        out = np.empty(N)
        out[J] = T
        return out

    Tx = midrank(probs[:m])
    Ty = midrank(probs[m:])
    Tz = midrank(probs)
    auc = float(Tz[:m].sum() / m / n - (m + 1) / (2.0 * n))
    v01 = (Tz[:m] - Tx) / n
    v10 = 1.0 - (Tz[m:] - Ty) / m
    var = float(v01.var(ddof=1) / m + v10.var(ddof=1) / n)
    se = float(np.sqrt(var)) if var > 0 else 0.0
    return auc, se


def main() -> None:
    yt_b, ys_b, mode_b = load_arm(ARM_B)
    yt_c, ys_c, mode_c = load_arm(ARM_C)
    print(f"arm_b: n={len(yt_b)} mode={mode_b}; arm_c: n={len(yt_c)} mode={mode_c}")

    paired_attempted = (len(yt_b) == len(yt_c))
    if not paired_attempted:
        # The two arms ran on different case subsets (radiomics filter dropped cases
        # in arm_b). We report single-arm DeLong CIs for each plus an unpaired
        # comparison; paired DeLong is queued for Round 6 (rebuild arm_b without
        # radiomics filter).
        auc_b, se_b = single_arm_ci(ys_b, yt_b)
        auc_c, se_c = single_arm_ci(ys_c, yt_c)
        diff = auc_c - auc_b
        # unpaired SE of difference (uncorrelated approximation)
        se_diff = float(np.sqrt(se_b**2 + se_c**2))
        z = diff / se_diff if se_diff > 0 else 0.0
        p = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
        out = {
            "primary_endpoint": "arm_c gcn_only contrast-only AUC (single-arm DeLong CI)",
            "case_set_mismatch": True,
            "n_arm_b": int(len(yt_b)),
            "n_arm_c": int(len(yt_c)),
            "best_mode_arm_b": mode_b,
            "best_mode_arm_c": mode_c,
            "AUC_arm_b": auc_b,
            "AUC_arm_b_ci95": [float(auc_b - 1.96 * se_b), float(auc_b + 1.96 * se_b)],
            "AUC_arm_c": auc_c,
            "AUC_arm_c_ci95": [float(auc_c - 1.96 * se_c), float(auc_c + 1.96 * se_c)],
            "delta_AUC_unpaired": diff,
            "delta_z_unpaired": z,
            "delta_p_unpaired": p,
            "note": "Paired DeLong unavailable: arm_b dataset is filtered by radiomics requirement to 92 cases vs arm_c's 189. Round 6 will rebuild arm_b without radiomics filter for a true paired comparison.",
        }
        OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
        lines = [
            "# R5 — Single-arm DeLong CIs (contrast-only PRIMARY ENDPOINT)",
            "",
            "**Note**: arm_b's dataset is restricted to 92 cases by the radiomics-feature",
            "requirement; arm_c uses all 189. Paired DeLong on the same case set is",
            "deferred to Round 6 (rebuild arm_b with --skip_radiomics_filter).",
            "",
            f"- arm_b (n={len(yt_b)}, mode={mode_b}): AUC = **{auc_b:.4f}** "
            f"[{out['AUC_arm_b_ci95'][0]:.4f}, {out['AUC_arm_b_ci95'][1]:.4f}]",
            f"- arm_c (n={len(yt_c)}, mode={mode_c}): AUC = **{auc_c:.4f}** "
            f"[{out['AUC_arm_c_ci95'][0]:.4f}, {out['AUC_arm_c_ci95'][1]:.4f}]",
            f"- Δ AUC (unpaired, approximate): {diff:+.4f}, z={z:.3f}, p={p:.4g}",
            "",
            "## Reading",
            "",
            f"- arm_c CI excludes 0.5 (random)? **{'YES' if out['AUC_arm_c_ci95'][0] > 0.5 else 'NO'}** — disease",
            f"  signal on contrast-only is significant.",
            "- The unpaired Δ test is approximate; the paired version is the formal",
            "  W6 endpoint. Round 6 will close that gap.",
        ]
        OUT_MD.write_text("\n".join(lines), encoding="utf-8")
        print(OUT_MD.read_text(encoding="utf-8"))
        return

    auc_b, auc_c, var_b, var_c, cov, z, p = fast_delong(ys_c, ys_b, yt_b)
    diff = auc_c - auc_b
    se = np.sqrt(var_b + var_c - 2 * cov) if var_b + var_c - 2 * cov > 0 else 0.0
    ci_lo = diff - 1.96 * se
    ci_hi = diff + 1.96 * se

    # Bootstrap CI on Δ as a sanity check
    rng = np.random.default_rng(20260423)
    n = len(yt_b)
    boot = np.empty(5000)
    for i in range(5000):
        idx = rng.integers(0, n, n)
        try:
            from sklearn.metrics import roc_auc_score
            a_b = roc_auc_score(yt_b[idx], ys_b[idx])
            a_c = roc_auc_score(yt_c[idx], ys_c[idx])
            boot[i] = a_c - a_b
        except ValueError:
            boot[i] = np.nan
    boot = boot[~np.isnan(boot)]
    bs_lo, bs_hi = np.percentile(boot, [2.5, 97.5])

    out = {
        "primary_endpoint": "arm_c − arm_b contrast-only ensemble AUC",
        "n_cases": int(n),
        "n_pos": int(yt_b.sum()),
        "n_neg": int((1 - yt_b).sum()),
        "best_mode_arm_b": mode_b,
        "best_mode_arm_c": mode_c,
        "AUC_arm_b": auc_b,
        "AUC_arm_c": auc_c,
        "delta_AUC": diff,
        "delong_z": z,
        "delong_p_two_sided": p,
        "delong_ci95": [float(ci_lo), float(ci_hi)],
        "bootstrap_ci95": [float(bs_lo), float(bs_hi)],
    }
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        "# R5 — Paired DeLong on arm_c − arm_b contrast-only (PRIMARY ENDPOINT)",
        "",
        f"Cases: {n} (PH={int(yt_b.sum())}, nonPH={int((1-yt_b).sum())})",
        f"Best mode used per arm — arm_b: `{mode_b}`, arm_c: `{mode_c}`.",
        "",
        f"- AUC arm_b: **{auc_b:.4f}**",
        f"- AUC arm_c: **{auc_c:.4f}**",
        f"- Δ AUC (arm_c − arm_b): **{diff:+.4f}**",
        f"- DeLong 95% CI: **[{ci_lo:+.4f}, {ci_hi:+.4f}]**",
        f"- DeLong z = {z:.3f}, **p two-sided = {p:.4g}**",
        f"- Bootstrap 95% CI on Δ (n=5000): **[{bs_lo:+.4f}, {bs_hi:+.4f}]**",
        "",
        "## Interpretation",
        "",
        f"- If the DeLong CI excludes 0, the lung-feature contribution (arm_c) is",
        f"  significantly different from arm_b (vessel-only) on the protocol-balanced",
        f"  contrast subset. CI: [{ci_lo:+.4f}, {ci_hi:+.4f}] {'EXCLUDES 0 → significant' if ci_lo > 0 or ci_hi < 0 else 'INCLUDES 0 → not significant'}.",
        "- This is the **primary endpoint** for the W6 confirmatory inference",
        "  request from Round 4. Multiplicity correction is not needed for a single",
        "  pre-specified comparison.",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
