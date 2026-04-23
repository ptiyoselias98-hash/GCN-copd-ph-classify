"""E2 — parenchyma phenotype cluster on protocol-robust v2 features.

Input: `outputs/lung_features_v2.csv` (paren_* + spatial_* features).
Input: `data/case_protocol.csv` (label + protocol).

Pipeline:
  1. Pick 9 parenchyma-only features (paren HU + LAA + apical/mid/basal LAA
     + apical-basal gradient).
  2. Z-score globally, drop rows with any NaN.
  3. UMAP (n_neighbors=15, min_dist=0.1, random_state=20260423).
  4. GMM k ∈ {2, 3, 4, 5}; pick k by BIC.
  5. Per-cluster: size, label counts, PH proportion + Wilson 95% CI,
     protocol mix, top-3 centroid features.
  6. Two figures: (a) UMAP coloured by label, (b) UMAP coloured by protocol.

Output:
  - outputs/evolution/E2_paren_cluster.json
  - outputs/evolution/E2_paren_cluster.md
  - outputs/evolution/E2_paren_umap_label.png
  - outputs/evolution/E2_paren_umap_protocol.png
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

try:
    import umap  # type: ignore
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False

from sklearn.decomposition import PCA

ROOT = Path(__file__).parent.parent.parent
V2 = ROOT / "outputs" / "lung_features_v2.csv"
PROTO = ROOT / "data" / "case_protocol.csv"
OUT_DIR = ROOT / "outputs" / "evolution"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATS = [
    "paren_mean_HU", "paren_std_HU",
    "paren_LAA_950_frac", "paren_LAA_910_frac", "paren_LAA_856_frac",
    "apical_LAA_950_frac", "middle_LAA_950_frac", "basal_LAA_950_frac",
    "apical_basal_LAA950_gradient",
]

SEED = 20260423


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / denom, (centre + half) / denom


def main() -> None:
    v2 = pd.read_csv(V2)
    proto = pd.read_csv(PROTO)
    df = proto.merge(v2, on="case_id", how="inner").dropna(subset=FEATS).reset_index(drop=True)
    print(f"cases after NaN drop: {len(df)}")

    X = StandardScaler().fit_transform(df[FEATS].to_numpy())

    # BIC sweep
    bic_log = {}
    best_k, best_bic, best_model = None, np.inf, None
    for k in (2, 3, 4, 5):
        m = GaussianMixture(n_components=k, covariance_type="full", random_state=SEED, n_init=5)
        m.fit(X)
        bic = float(m.bic(X))
        bic_log[k] = bic
        if bic < best_bic:
            best_bic, best_k, best_model = bic, k, m
    assert best_model is not None
    print(f"GMM BIC: {bic_log}  → k={best_k}")

    cluster = best_model.predict(X)
    df["cluster"] = cluster

    # 2D embedding for visualization (UMAP if available, else PCA)
    if _HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=SEED)
        emb = reducer.fit_transform(X)
        emb_label = "UMAP"
    else:
        reducer = PCA(n_components=2, random_state=SEED)
        emb = reducer.fit_transform(X)
        emb_label = "PCA"

    # Per-cluster summary
    clusters_out = []
    for k in range(best_k):
        idx = cluster == k
        sub = df[idx]
        n = int(idx.sum())
        n_ph = int((sub["label"] == 1).sum())
        p = n_ph / n if n else 0.0
        lo, hi = wilson(p, n)
        proto_mix = sub["protocol"].value_counts().to_dict()
        centroid = best_model.means_[k]
        top3 = sorted(
            ((FEATS[i], float(centroid[i])) for i in range(len(FEATS))),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:3]
        clusters_out.append(
            {
                "cluster": k,
                "size": n,
                "n_ph": n_ph,
                "n_nonph": n - n_ph,
                "ph_proportion": p,
                "ph_ci95": [lo, hi],
                "protocol_mix": proto_mix,
                "top3_z_features": top3,
            }
        )

    # Plots
    fig, ax = plt.subplots(figsize=(6, 5))
    for lab, c, marker in ((0, "tab:blue", "o"), (1, "tab:red", "^")):
        m = df["label"] == lab
        ax.scatter(emb[m, 0], emb[m, 1], c=c, label=f"label={lab}", s=18, alpha=0.75, edgecolor="k", linewidth=0.3, marker=marker)
    ax.set_xlabel(f"{emb_label}-1")
    ax.set_ylabel(f"{emb_label}-2")
    ax.set_title(f"E2 parenchyma {emb_label} — label (n={len(df)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "E2_paren_umap_label.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {"contrast": "tab:orange", "plain_scan": "tab:green"}
    for proto_name, col in colors.items():
        m = df["protocol"] == proto_name
        ax.scatter(emb[m, 0], emb[m, 1], c=col, label=proto_name, s=18, alpha=0.75, edgecolor="k", linewidth=0.3)
    ax.set_xlabel(f"{emb_label}-1")
    ax.set_ylabel(f"{emb_label}-2")
    ax.set_title(f"E2 parenchyma {emb_label} — protocol (n={len(df)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "E2_paren_umap_protocol.png", dpi=150)
    plt.close(fig)

    # Markdown summary
    lines = [
        "# E2 — Parenchyma phenotype cluster (protocol-robust v2 features)",
        "",
        f"Cases: {len(df)} (label=0: {(df['label']==0).sum()}, label=1: {(df['label']==1).sum()})",
        f"Features (n={len(FEATS)}): {', '.join(FEATS)}",
        f"Best GMM k by BIC: **k={best_k}**  (BIC: {bic_log})",
        "",
        "## Cluster composition",
        "",
        "| Cluster | Size | PH count | PH proportion | 95% Wilson CI | Protocol mix | Top-3 |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in clusters_out:
        t3 = ", ".join(f"{n}={v:+.2f}" for (n, v) in c["top3_z_features"])
        mix = ", ".join(f"{k}={v}" for k, v in c["protocol_mix"].items())
        lines.append(
            f"| {c['cluster']} | {c['size']} | {c['n_ph']} | "
            f"{c['ph_proportion']:.2%} | "
            f"[{c['ph_ci95'][0]:.2f}, {c['ph_ci95'][1]:.2f}] | "
            f"{mix} | {t3} |"
        )
    overall_ph_frac = float((df["label"] == 1).mean())
    lines += [
        "",
        f"**Baseline PH proportion across all clustered cases: {overall_ph_frac:.2%}.**",
        "A cluster is 'PH-enriched' if its proportion CI excludes the baseline.",
        "",
        "## Protocol-stratified check",
        "",
        "PH proportion within `contrast` cases only (should be ~163/189 ≈ 86% if protocol is balanced):",
        "",
        "| Cluster | n_contrast | PH_contrast | PH% (contrast) |",
        "|---|---|---|---|",
    ]
    for c in clusters_out:
        sub = df[df["cluster"] == c["cluster"]]
        sub_c = sub[sub["protocol"] == "contrast"]
        nc = len(sub_c)
        nphc = int((sub_c["label"] == 1).sum())
        pct = (nphc / nc) if nc else float("nan")
        lines.append(f"| {c['cluster']} | {nc} | {nphc} | {pct:.2%} |")
    lines += [
        "",
        "## Figures",
        "",
        "![UMAP by label](E2_paren_umap_label.png)",
        "",
        "![UMAP by protocol](E2_paren_umap_protocol.png)",
        "",
        "## Interpretation",
        "",
        "A protocol-robust cluster should show (a) a non-trivially PH-enriched cluster with 95% CI above",
        "the baseline PH rate, and (b) balanced protocol mix within that cluster. If instead the PH-enriched",
        "clusters are dominated by `contrast` cases, this means the parenchyma features still leak protocol —",
        "in which case the vessel-graph contribution (E1) becomes the primary driver of any disease claim.",
    ]
    (OUT_DIR / "E2_paren_cluster.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "E2_paren_cluster.json").write_text(
        json.dumps(
            {
                "n_cases": len(df),
                "features": FEATS,
                "best_k": best_k,
                "bic": bic_log,
                "clusters": clusters_out,
                "overall_ph_proportion": overall_ph_frac,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print((OUT_DIR / "E2_paren_cluster.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
