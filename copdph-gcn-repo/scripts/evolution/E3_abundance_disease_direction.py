"""E3 — HiPaS-aligned disease-direction test on local vessel abundance.

Chu et al. Nature Comm 2025 (HiPaS, n=11,784, lung-volume-controlled) reports:
  - PAH → lower pulmonary artery abundance (skeleton length, branch count)
  - COPD → lower pulmonary vein abundance

We test this on our contrast-only 189-case subset using `artery_vol_mL` and
`vein_vol_mL` (volume is the crudest abundance proxy but is the one metric
we already have locally without skeletonization).

Tests:
  1. PH (163) vs nonPH (26) on `artery_vol_mL / lung_vol_mL` in the contrast subset
     — expect PH < nonPH (Mann-Whitney one-sided, Cliff's delta).
  2. COPD severity (paren_LAA_910_frac) vs `vein_vol_mL / lung_vol_mL`
     in the full 282 cohort, stratified by protocol — expect negative
     correlation (Spearman) that survives within both protocols.
  3. Report two-sided p-values too for multiplicity honesty.

Output: outputs/evolution/E3_abundance_directions.{md,json}.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent.parent.parent
V2 = ROOT / "outputs" / "lung_features_v2.csv"
PROTO = ROOT / "data" / "case_protocol.csv"
OUT_DIR = ROOT / "outputs" / "evolution"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("nan")
    xx = x.reshape(-1, 1)
    yy = y.reshape(1, -1)
    g = (xx > yy).sum() - (xx < yy).sum()
    return float(g / (nx * ny))


def main() -> None:
    v2 = pd.read_csv(V2)
    proto = pd.read_csv(PROTO)
    df = proto.merge(v2, on="case_id", how="inner")

    # normalize vessel volumes by lung volume
    df["artery_frac"] = df["artery_vol_mL"] / df["lung_vol_mL"]
    df["vein_frac"] = df["vein_vol_mL"] / df["lung_vol_mL"]
    df["airway_frac"] = df["airway_vol_mL"] / df["lung_vol_mL"]

    # drop placeholders + NaN rows
    valid = df[(df["artery_placeholder"] == 0) & (df["vein_placeholder"] == 0)].copy()
    valid = valid.dropna(subset=["artery_frac", "vein_frac", "lung_vol_mL", "paren_LAA_910_frac"])
    print(f"valid (no placeholder + non-NaN): {len(valid)} / 282")

    results: dict = {"tests": []}

    # === Test 1: PH vs nonPH on artery_frac (contrast-only) ===
    contrast = valid[valid["protocol"] == "contrast"]
    ph = contrast[contrast["label"] == 1]["artery_frac"].to_numpy()
    nph = contrast[contrast["label"] == 0]["artery_frac"].to_numpy()
    if len(ph) and len(nph):
        u, p_two = stats.mannwhitneyu(ph, nph, alternative="two-sided")
        _, p_less = stats.mannwhitneyu(ph, nph, alternative="less")
        delta = cliffs_delta(ph, nph)
        median_ph = float(np.median(ph))
        median_nph = float(np.median(nph))
        hipas_predicted = "PH < nonPH (reduced artery abundance)"
        matches = median_ph < median_nph
        results["tests"].append(
            {
                "name": "T1_artery_frac_PH_vs_nonPH_contrast",
                "hipas_prediction": hipas_predicted,
                "n_ph": int(len(ph)),
                "n_nonph": int(len(nph)),
                "median_ph": median_ph,
                "median_nph": median_nph,
                "U": float(u),
                "p_two_sided": float(p_two),
                "p_one_sided_less": float(p_less),
                "cliffs_delta": delta,
                "direction_matches_hipas": matches,
            }
        )

    # === Test 2: COPD severity (LAA_910) vs vein_frac, stratified by protocol ===
    for proto_name in ("contrast", "plain_scan"):
        sub = valid[valid["protocol"] == proto_name]
        if len(sub) < 10:
            continue
        rho, p = stats.spearmanr(sub["paren_LAA_910_frac"], sub["vein_frac"])
        results["tests"].append(
            {
                "name": f"T2_vein_frac_vs_LAA910_{proto_name}",
                "hipas_prediction": "Spearman(LAA_910, vein_frac) < 0 (more emphysema → less vein)",
                "n": int(len(sub)),
                "spearman_rho": float(rho),
                "spearman_p": float(p),
                "direction_matches_hipas": bool(rho < 0),
            }
        )

    # === Test 3: same but with artery_frac vs LAA_910 (sanity; HiPaS predicts PAH-specific artery loss, not COPD-specific) ===
    for proto_name in ("contrast", "plain_scan"):
        sub = valid[valid["protocol"] == proto_name]
        if len(sub) < 10:
            continue
        rho, p = stats.spearmanr(sub["paren_LAA_910_frac"], sub["artery_frac"])
        results["tests"].append(
            {
                "name": f"T3_artery_frac_vs_LAA910_{proto_name}",
                "hipas_prediction": "weaker effect than vein (COPD-specific)",
                "n": int(len(sub)),
                "spearman_rho": float(rho),
                "spearman_p": float(p),
            }
        )

    # === Test 4: PH vs nonPH on vein_frac (secondary; HiPaS says COPD drives vein, PAH drives artery) ===
    ph_v = contrast[contrast["label"] == 1]["vein_frac"].to_numpy()
    nph_v = contrast[contrast["label"] == 0]["vein_frac"].to_numpy()
    if len(ph_v) and len(nph_v):
        u, p_two = stats.mannwhitneyu(ph_v, nph_v, alternative="two-sided")
        delta = cliffs_delta(ph_v, nph_v)
        results["tests"].append(
            {
                "name": "T4_vein_frac_PH_vs_nonPH_contrast",
                "hipas_prediction": "weaker than T1 (PH is artery-specific)",
                "n_ph": int(len(ph_v)),
                "n_nonph": int(len(nph_v)),
                "median_ph": float(np.median(ph_v)),
                "median_nph": float(np.median(nph_v)),
                "p_two_sided": float(p_two),
                "cliffs_delta": delta,
            }
        )

    # Markdown
    lines = [
        "# E3 — HiPaS-aligned disease-direction test (local vessel abundance)",
        "",
        "HiPaS (Chu et al., Nature Comm 2025, n=11,784) reports (lung-volume controlled):",
        "",
        "- **PAH → lower pulmonary artery abundance** (skeleton length, branch count)",
        "- **COPD → lower pulmonary vein abundance**",
        "",
        "We test these directions on our cohort using `artery_vol_mL / lung_vol_mL`",
        "and `vein_vol_mL / lung_vol_mL` as abundance proxies, restricted to cases",
        "without placeholder segmentations.",
        "",
    ]
    for t in results["tests"]:
        lines.append(f"## {t['name']}")
        lines.append("")
        for k, v in t.items():
            if k == "name":
                continue
            if isinstance(v, float):
                lines.append(f"- **{k}**: {v:.4f}")
            elif isinstance(v, bool):
                lines.append(f"- **{k}**: {v}")
            else:
                lines.append(f"- **{k}**: {v}")
        lines.append("")
    # Summary
    t1 = next((t for t in results["tests"] if t["name"].startswith("T1")), None)
    t2c = next((t for t in results["tests"] if t["name"].startswith("T2") and "contrast" in t["name"]), None)
    summary = []
    if t1:
        summary.append(
            f"T1 (PAH → ↓artery on contrast-only): median PH {t1['median_ph']:.4f} "
            f"vs nonPH {t1['median_nph']:.4f}, two-sided p={t1['p_two_sided']:.4g}, "
            f"Cliff's δ={t1['cliffs_delta']:+.3f}. "
            f"Direction matches HiPaS: **{t1['direction_matches_hipas']}**."
        )
    if t2c:
        summary.append(
            f"T2 (COPD → ↓vein on contrast): Spearman ρ={t2c['spearman_rho']:+.3f}, "
            f"p={t2c['spearman_p']:.4g}. "
            f"Direction matches HiPaS: **{t2c['direction_matches_hipas']}**."
        )
    lines += ["## Summary", "", *[f"- {s}" for s in summary]]
    lines += [
        "",
        "**Reading**: we use volume fractions as a crude proxy for HiPaS's skeleton-length",
        "and branch-count metrics. True skeleton-based abundance requires the remote kimimaro",
        "graphs (queued). Even with this proxy, directional agreement with HiPaS serves as",
        "a literature-aligned falsification for our cache quality.",
    ]
    (OUT_DIR / "E3_abundance_directions.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "E3_abundance_directions.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print((OUT_DIR / "E3_abundance_directions.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
