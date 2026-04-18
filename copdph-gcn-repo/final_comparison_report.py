"""Aggregate sprint2 + sprint3 + sprint5 results into a single comparison table.

Output:
  outputs/final_comparison.xlsx
  outputs/final_comparison.png  (grouped bar: AUC / Spec / F1)

Picks `enhanced/hybrid` rows where available (most diagnostic arm).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTROOT = Path(r"C:\Users\cheng\Desktop\图卷积-肺小血管演化规律探索\outputs")

SOURCES = [
    # (label, json_path_relative_to_OUTROOT, feat_set, mode)
    ("Sprint2 baseline/gcn_only",     "sprint2_v2/sprint2_results.json",     "baseline", "gcn_only"),
    ("Sprint2 enhanced/hybrid",       "sprint2_v2/sprint2_results.json",     "enhanced", "hybrid"),
    ("Sprint3 focal_local4 enh/hyb",  "sprint3_focal_local4/sprint3_results.json", "enhanced", "hybrid"),
    ("Sprint3 wce_local4 enh/rad",    "sprint3_wce_local4/sprint3_results.json",    "enhanced", "radiomics_only"),
    ("Sprint5 full enh/hyb",          "sprint5_full/sprint5_results.json",    "enhanced", "hybrid"),
    ("Sprint5 ndrop_only enh/hyb",    "sprint5_ndrop_only/sprint5_results.json", "enhanced", "hybrid"),
    ("Sprint5 mpap_only enh/hyb",     "sprint5_mpap_only/sprint5_results.json",  "enhanced", "hybrid"),
]

METRICS = ["AUC", "Accuracy", "Precision", "Sensitivity", "F1", "Specificity"]


def load_row(label, json_rel, fs, mode):
    p = OUTROOT / json_rel
    if not p.exists():
        return {"method": label, **{k: np.nan for k in METRICS}, "pooled_AUC": np.nan}
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if fs not in data or mode not in data.get(fs, {}):
        return {"method": label, **{k: np.nan for k in METRICS}, "pooled_AUC": np.nan}
    r = data[fs][mode]
    mean = r.get("mean", {})
    pooled = float(r.get("pooled_AUC", np.nan))
    row = {"method": label, "pooled_AUC": pooled}
    for k in METRICS:
        row[k] = float(mean.get(k, np.nan))
    return row


def main() -> int:
    rows = [load_row(*s) for s in SOURCES]
    df = pd.DataFrame(rows)
    out_xlsx = OUTROOT / "final_comparison.xlsx"
    df.to_excel(out_xlsx, index=False)
    print(f"saved {out_xlsx}")
    print(df.to_string(index=False))

    # Plot: AUC, Spec, F1 grouped bar
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(df))
    w = 0.25
    for i, k in enumerate(["AUC", "Specificity", "F1"]):
        ax.bar(x + (i - 1) * w, df[k].values, w, label=k)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" ", "\n") for m in df["method"]],
                       rotation=0, fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.axhline(0.8, color="gray", linestyle="--", alpha=0.4)
    ax.legend()
    ax.set_title("Final Comparison — fold-mean metrics")
    plt.tight_layout()
    out_png = OUTROOT / "final_comparison.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"saved {out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
