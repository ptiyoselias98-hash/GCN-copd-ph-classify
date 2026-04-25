"""R16.A — Independent segmentation QC on the 100 new plain-scan cases.

R15 reviewer flag: Simple_AV_seg trained on CTPA, applied to plain-scan;
domain-transfer QC missing. This script flags outliers in lung/artery/vein
mask metrics that suggest segmentation failure WITHOUT independent
ground truth (we don't have hand-labels). Falls back on:

  - Lung volume out of plausible adult human range (1.5-8.5 L)
  - Lung HU mean out of plausible parenchyma range (-1000 to -500)
  - Vessel/lung ratio out of plausible vasculature range (0.5%-25%)
  - Artery vol / vein vol ratio out of plausible bipartite range (0.3-3.0)
  - Pure-parenchyma N voxels < 100k (suspicious for full chest CT)

Outputs: outputs/r16/seg_qc_new100.{json,md}
        outputs/r16/seg_qc_new100_flagged.csv  (flagged cases for spot QC)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
LUNG = ROOT / "outputs" / "r15" / "lung_features_new100.csv"
OUT = ROOT / "outputs" / "r16"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    if not LUNG.exists():
        raise SystemExit(f"missing {LUNG}")
    df = pd.read_csv(LUNG)
    print(f"loaded {len(df)} new100 cases")
    rules = [
        ("lung_vol_implausible_low", df["lung_vol_mL"] < 1500),
        ("lung_vol_implausible_high", df["lung_vol_mL"] > 8500),
        ("paren_HU_implausible_high", df["paren_mean_HU"] > -500),
        ("paren_HU_implausible_low", df["paren_mean_HU"] < -1000),
        ("vessel_lung_ratio_too_high", df["vessel_airway_over_lung"] > 0.25),
        ("vessel_lung_ratio_too_low", df["vessel_airway_over_lung"] < 0.005),
        ("paren_too_few_voxels", df["paren_n_voxels"] < 100000),
    ]
    if "artery_vol_mL" in df.columns and "vein_vol_mL" in df.columns:
        ratio = df["artery_vol_mL"] / df["vein_vol_mL"].replace(0, np.nan)
        rules += [
            ("av_ratio_extreme_high", ratio > 5.0),
            ("av_ratio_extreme_low", ratio < 0.2),
        ]

    flag_matrix = np.zeros((len(df), len(rules)), bool)
    rule_names = []
    for j, (name, mask) in enumerate(rules):
        rule_names.append(name)
        flag_matrix[:, j] = mask.fillna(False).values
    n_flags_per_case = flag_matrix.sum(axis=1)
    df["n_qc_flags"] = n_flags_per_case
    flagged = df[n_flags_per_case > 0][["case_id", "n_qc_flags",
                                        "lung_vol_mL", "paren_mean_HU",
                                        "vessel_airway_over_lung",
                                        "paren_n_voxels"]].copy()
    if "artery_vol_mL" in df.columns:
        flagged["artery_vol_mL"] = df.loc[flagged.index, "artery_vol_mL"]
    if "vein_vol_mL" in df.columns:
        flagged["vein_vol_mL"] = df.loc[flagged.index, "vein_vol_mL"]
    flagged = flagged.sort_values("n_qc_flags", ascending=False)
    flagged.to_csv(OUT / "seg_qc_new100_flagged.csv", index=False)

    # Marginal stats
    out = {
        "n_total": int(len(df)),
        "n_flagged": int((n_flags_per_case > 0).sum()),
        "n_severe_flagged": int((n_flags_per_case >= 3).sum()),
        "rule_hits": {name: int(flag_matrix[:, j].sum())
                      for j, name in enumerate(rule_names)},
        "lung_vol_mL_summary": {
            "min": float(df["lung_vol_mL"].min()),
            "p5": float(np.percentile(df["lung_vol_mL"], 5)),
            "median": float(df["lung_vol_mL"].median()),
            "p95": float(np.percentile(df["lung_vol_mL"], 95)),
            "max": float(df["lung_vol_mL"].max()),
        },
        "paren_mean_HU_summary": {
            "min": float(df["paren_mean_HU"].min()),
            "p5": float(np.percentile(df["paren_mean_HU"], 5)),
            "median": float(df["paren_mean_HU"].median()),
            "p95": float(np.percentile(df["paren_mean_HU"], 95)),
            "max": float(df["paren_mean_HU"].max()),
        },
        "vessel_lung_ratio_summary": {
            "min": float(df["vessel_airway_over_lung"].min()),
            "p5": float(np.percentile(df["vessel_airway_over_lung"], 5)),
            "median": float(df["vessel_airway_over_lung"].median()),
            "p95": float(np.percentile(df["vessel_airway_over_lung"], 95)),
            "max": float(df["vessel_airway_over_lung"].max()),
        },
    }
    (OUT / "seg_qc_new100.json").write_text(json.dumps(out, indent=2),
                                                encoding="utf-8")

    md = ["# R16.A — Independent segmentation QC on 100 new plain-scan cases",
          "",
          "Addresses R15 reviewer flag: Simple_AV_seg trained on CTPA → plain-scan",
          "domain-transfer concern. No hand-label ground truth available; this is",
          "a marginal-distribution sanity check using physiological priors.",
          "",
          f"**Total cases**: {out['n_total']}",
          f"**Cases with ≥1 QC flag**: {out['n_flagged']} ({out['n_flagged']/out['n_total']:.0%})",
          f"**Cases with ≥3 QC flags (severe)**: {out['n_severe_flagged']}",
          "",
          "## Per-rule hit counts",
          "",
          "| QC rule | n flagged |",
          "|---|---|"]
    for name, n in out["rule_hits"].items():
        md.append(f"| {name} | {n} |")
    md += ["",
           "## Distribution summaries (physiological priors in parens)",
           "",
           "| metric | min | p5 | median | p95 | max | plausible range |",
           "|---|---|---|---|---|---|---|",
           f"| lung_vol_mL | {out['lung_vol_mL_summary']['min']:.0f} | "
           f"{out['lung_vol_mL_summary']['p5']:.0f} | "
           f"{out['lung_vol_mL_summary']['median']:.0f} | "
           f"{out['lung_vol_mL_summary']['p95']:.0f} | "
           f"{out['lung_vol_mL_summary']['max']:.0f} | (1500-8500) |",
           f"| paren_mean_HU | {out['paren_mean_HU_summary']['min']:.0f} | "
           f"{out['paren_mean_HU_summary']['p5']:.0f} | "
           f"{out['paren_mean_HU_summary']['median']:.0f} | "
           f"{out['paren_mean_HU_summary']['p95']:.0f} | "
           f"{out['paren_mean_HU_summary']['max']:.0f} | (-1000 to -500) |",
           f"| vessel/lung | {out['vessel_lung_ratio_summary']['min']:.4f} | "
           f"{out['vessel_lung_ratio_summary']['p5']:.4f} | "
           f"{out['vessel_lung_ratio_summary']['median']:.4f} | "
           f"{out['vessel_lung_ratio_summary']['p95']:.4f} | "
           f"{out['vessel_lung_ratio_summary']['max']:.4f} | (0.005-0.25) |",
           "",
           "Flagged cases listed in `seg_qc_new100_flagged.csv` (sorted by n_qc_flags).",
           ""]
    (OUT / "seg_qc_new100.md").write_text("\n".join(md), encoding="utf-8")
    print(f"flagged={out['n_flagged']} severe={out['n_severe_flagged']} "
          f"of {out['n_total']}")


if __name__ == "__main__":
    main()
