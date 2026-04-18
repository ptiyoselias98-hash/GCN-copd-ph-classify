"""Group statistics on ALL CT/vessel features (PH vs non-PH).

Reads the radiomics CSV (patient_id, label, ... all CT features), runs
Mann-Whitney U per feature, applies Benjamini-Hochberg FDR correction,
and writes:
  - outputs/ct_feature_group_stats.csv   (sorted by p-value)
  - outputs/ct_feature_group_stats.md    (human-readable top table)
  - outputs/ct_feature_boxplots_top.png  (boxplots for top-K significant)
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("ct-stats")

RADIOMICS_CSV = Path(__file__).resolve().parent.parent / "data" / "copd_ph_radiomics.csv"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
TOP_K = 16  # how many top-significant features to plot


def _df_to_md(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = "\n".join("| " + " | ".join(str(v) for v in row) + " |"
                     for row in df.itertuples(index=False, name=None))
    return "\n".join([header, sep, body])


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Effect size: fraction(a>b) - fraction(a<b), range [-1, 1]."""
    a = np.asarray(a); b = np.asarray(b)
    if a.size == 0 or b.size == 0:
        return np.nan
    gt = np.sum(a[:, None] > b[None, :])
    lt = np.sum(a[:, None] < b[None, :])
    return (gt - lt) / (a.size * b.size)


def main() -> None:
    if not RADIOMICS_CSV.exists():
        raise FileNotFoundError(RADIOMICS_CSV)
    df = pd.read_csv(RADIOMICS_CSV)
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    logger.info("loaded %d rows × %d cols", *df.shape)

    if "label" not in df.columns:
        raise ValueError("CSV missing 'label' column")
    labels = df["label"].astype(int).values
    logger.info("label counts: PH=%d, non-PH=%d", int((labels == 1).sum()),
                int((labels == 0).sum()))

    feat_cols = [c for c in df.columns if c not in ("patient_id", "label")]
    numeric_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])]
    logger.info("numeric features: %d", len(numeric_cols))

    rows = []
    for col in numeric_cols:
        s = df[col]
        g0 = s[labels == 0].dropna().values.astype(float)
        g1 = s[labels == 1].dropna().values.astype(float)
        if g0.size < 3 or g1.size < 3 or np.nanstd(np.r_[g0, g1]) < 1e-12:
            continue
        try:
            _, p = stats.mannwhitneyu(g0, g1, alternative="two-sided")
        except ValueError:
            continue
        rows.append({
            "feature": col,
            "n_nonPH": g0.size,
            "n_PH": g1.size,
            "median_nonPH": float(np.median(g0)),
            "median_PH": float(np.median(g1)),
            "mean_nonPH": float(np.mean(g0)),
            "mean_PH": float(np.mean(g1)),
            "delta_median": float(np.median(g1) - np.median(g0)),
            "cliffs_delta": cliffs_delta(g1, g0),
            "p_mannwhitney": float(p),
        })

    res = pd.DataFrame(rows)
    res["p_BH_FDR"] = bh_fdr(res["p_mannwhitney"].values)
    res["sig_0.05"] = res["p_BH_FDR"] < 0.05
    res = res.sort_values("p_mannwhitney").reset_index(drop=True)
    logger.info("tested %d features — %d significant at FDR<0.05",
                len(res), int(res["sig_0.05"].sum()))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "ct_feature_group_stats.csv"
    res.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info("wrote %s", csv_path)

    md_path = OUT_DIR / "ct_feature_group_stats.md"
    top = res.head(30).copy()
    for c in ("median_nonPH", "median_PH", "delta_median", "cliffs_delta"):
        top[c] = top[c].map(lambda x: f"{x:.3g}")
    top["p_mannwhitney"] = top["p_mannwhitney"].map(lambda x: f"{x:.2e}")
    top["p_BH_FDR"] = top["p_BH_FDR"].map(lambda x: f"{x:.2e}")
    lines = [
        f"# CT feature group statistics — PH vs non-PH\n",
        f"Source: `{RADIOMICS_CSV.name}` (n_PH={int((labels==1).sum())}, "
        f"n_nonPH={int((labels==0).sum())}).\n",
        f"Tested {len(res)} numeric features; "
        f"**{int(res['sig_0.05'].sum())} significant** at BH-FDR<0.05.\n",
        "## Top 30 features by raw p-value\n",
        _df_to_md(top[["feature", "median_nonPH", "median_PH", "delta_median",
                       "cliffs_delta", "p_mannwhitney", "p_BH_FDR"]]),
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", md_path)

    plot_top_k(df, labels, res, k=TOP_K)


def plot_top_k(df: pd.DataFrame, labels: np.ndarray, res: pd.DataFrame,
               k: int = 16) -> None:
    top = res.head(k)
    ncol = 4
    nrow = int(np.ceil(len(top) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow))
    axes = np.atleast_2d(axes).ravel()
    for ax, (_, rec) in zip(axes, top.iterrows()):
        col = rec["feature"]
        g0 = df.loc[labels == 0, col].dropna().values.astype(float)
        g1 = df.loc[labels == 1, col].dropna().values.astype(float)
        bp = ax.boxplot([g0, g1], labels=["non-PH", "PH"], widths=0.55,
                        patch_artist=True, showmeans=True)
        for patch, c in zip(bp["boxes"], ["#4C9AFF", "#E5484D"]):
            patch.set_facecolor(c); patch.set_alpha(0.55)
        rng = np.random.default_rng(0)
        for i, grp in enumerate([g0, g1], start=1):
            jitter = rng.normal(0, 0.04, size=len(grp))
            ax.scatter(np.full_like(grp, i, dtype=float) + jitter, grp, s=8,
                       alpha=0.6, color=["#0A3D91", "#8B1A1D"][i - 1])
        title = f"{col}\np={rec['p_mannwhitney']:.2e}  FDR={rec['p_BH_FDR']:.2e}"
        ax.set_title(title, fontsize=9)
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    for ax in axes[len(top):]:
        ax.axis("off")
    fig.suptitle(f"Top {len(top)} CT features — PH vs non-PH (Mann-Whitney)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = OUT_DIR / "ct_feature_boxplots_top.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
