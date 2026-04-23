"""Fold-level bootstrap CIs + paired Wilcoxon on arm AUC differences (W6, first pass).

True DeLong requires per-case probabilities which the current sprint6 result
JSONs do not persist. This script uses the 15 fold AUCs (5-fold × 3 repeats)
that ARE saved in each `sprint6_results.json → baseline.gcn_only.folds`
and reports:
  (a) percentile bootstrap CI on mean fold-AUC per arm,
  (b) paired Wilcoxon + paired bootstrap CI on the Δ between arms over the
      15 shared folds (arm_b vs arm_c, full vs contrast-only).

Outputs `outputs/_ci_fold_level.md` with a reviewer-facing summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "outputs"
OUT_MD = OUT_DIR / "_ci_fold_level.md"
OUT_JSON = OUT_DIR / "_ci_fold_level.json"

ARMS = {
    "arm_b_full": "sprint6_arm_b_triflat_v2/sprint6_results.json",
    "arm_c_full": "sprint6_arm_c_quad_v2/sprint6_results.json",
    "arm_b_contrast_only": "sprint6_arm_b_contrast_only_v2/sprint6_results.json",
    "arm_c_contrast_only": "sprint6_arm_c_contrast_only_v2/sprint6_results.json",
}

RNG = np.random.default_rng(20260423)
N_BOOT = 10_000


def load_fold_aucs(path: Path) -> np.ndarray:
    d = json.loads(path.read_text())
    return np.asarray(
        [f["AUC"] for f in d["baseline"]["gcn_only"]["folds"]], dtype=float
    )


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float, float]:
    n = len(x)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        means[b] = x[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def paired_bootstrap_delta_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT
) -> tuple[float, float, float]:
    assert len(x) == len(y)
    diffs = x - y
    n = len(diffs)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        boot[b] = diffs[idx].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def main() -> None:
    data = {k: load_fold_aucs(OUT_DIR / v) for k, v in ARMS.items()}

    rows_single = []
    for name, x in data.items():
        m, lo, hi = bootstrap_mean_ci(x)
        rows_single.append((name, len(x), m, lo, hi, float(x.std(ddof=1))))

    # Paired comparisons: arm_c − arm_b, matched by fold index.
    pairs = [
        ("arm_c_full − arm_b_full", data["arm_c_full"], data["arm_b_full"]),
        (
            "arm_c_contrast_only − arm_b_contrast_only",
            data["arm_c_contrast_only"],
            data["arm_b_contrast_only"],
        ),
        (
            "arm_b_full − arm_b_contrast_only",
            data["arm_b_full"],
            data["arm_b_contrast_only"],
        ),
        (
            "arm_c_full − arm_c_contrast_only",
            data["arm_c_full"],
            data["arm_c_contrast_only"],
        ),
    ]
    rows_paired = []
    for name, a, b in pairs:
        d_mean, d_lo, d_hi = paired_bootstrap_delta_ci(a, b)
        try:
            w = stats.wilcoxon(a, b, zero_method="pratt")
            w_stat, w_p = float(w.statistic), float(w.pvalue)
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        t = stats.ttest_rel(a, b)
        rows_paired.append(
            {
                "pair": name,
                "n_folds": len(a),
                "delta_mean": d_mean,
                "delta_ci_lo": d_lo,
                "delta_ci_hi": d_hi,
                "wilcoxon_stat": w_stat,
                "wilcoxon_p": w_p,
                "paired_t_stat": float(t.statistic),
                "paired_t_p": float(t.pvalue),
            }
        )

    # Markdown summary
    lines = [
        "# Fold-level bootstrap CIs + paired tests (W6, first pass)",
        "",
        "Per-arm 95% percentile bootstrap CIs on mean fold-AUC",
        "(n_boot=10000, 15 fold AUCs = 5-fold × 3 repeats):",
        "",
        "| Arm | n | mean AUC | 95% CI | SD |",
        "|---|---|---|---|---|",
    ]
    for name, n, m, lo, hi, sd in rows_single:
        lines.append(f"| {name} | {n} | {m:.3f} | [{lo:.3f}, {hi:.3f}] | {sd:.3f} |")
    lines += [
        "",
        "Paired deltas (matched by fold index, 15 paired folds):",
        "",
        "| Pair | Δ mean | 95% CI | Wilcoxon W | p | paired t | p |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows_paired:
        lines.append(
            f"| {r['pair']} | {r['delta_mean']:+.3f} | "
            f"[{r['delta_ci_lo']:+.3f}, {r['delta_ci_hi']:+.3f}] | "
            f"{r['wilcoxon_stat']:.2f} | {r['wilcoxon_p']:.4f} | "
            f"{r['paired_t_stat']:.2f} | {r['paired_t_p']:.4f} |"
        )
    lines += [
        "",
        "**Caveat**: these are fold-level tests, not case-level DeLong. Proper DeLong on",
        "paired AUCs requires per-case predicted probabilities, which are not persisted",
        "in the current `sprint6_results.json`. A small rerun writing val-set probs",
        "per fold is scheduled for Round 2 to replace these intervals with DeLong.",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    OUT_JSON.write_text(
        json.dumps(
            {
                "single": [
                    {
                        "arm": n,
                        "n_folds": nf,
                        "mean": m,
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "sd": sd,
                    }
                    for (n, nf, m, lo, hi, sd) in rows_single
                ],
                "paired": rows_paired,
                "n_boot": N_BOOT,
                "seed": 20260423,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(OUT_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
