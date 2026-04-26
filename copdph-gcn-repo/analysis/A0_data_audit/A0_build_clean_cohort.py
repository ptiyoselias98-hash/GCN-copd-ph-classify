"""A0_build_clean_cohort — Phase A: define 5 cohorts (C1..C5) for downstream phases.

Loads existing cohort_locked_table (R24.0) + mPAP gold + protocol + label,
defines 5 analysis cohorts per the user's Phase A specification:
  C1_all_available
  C2_within_contrast_only
  C3_borderline_mPAP_18_22
  C4_clear_mPAP_low_high (mPAP<20 vs ≥35)
  C5_early_COPD_no_PH (proxy: nonPH plain-scan or contrast-nonPH; no early-COPD label in this cohort)

Output: outputs/supplementary/A0_data_audit/clean_cohort_table.csv with cohort flags
        + protocol_summary.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
COHORT = ROOT / "outputs" / "r24" / "cohort_locked_table.csv"
OUT = ROOT / "outputs" / "supplementary" / "A0_data_audit"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(COHORT)
    print(f"loaded R24.0 locked table: n={len(df)}")
    # cohort flags
    df["C1_all_available"] = True
    df["C2_within_contrast_only"] = df["is_contrast_only_subset"].astype(bool)
    df["C3_borderline_mPAP_18_22"] = (
        df["measured_mpap_flag"]
        & (df["measured_mpap"] >= 18)
        & (df["measured_mpap"] <= 22)
    )
    df["C4_clear_low_high"] = (
        df["measured_mpap_flag"]
        & ((df["measured_mpap"] < 20) | (df["measured_mpap"] >= 35))
    )
    # C5 proxy: nonPH cases as "early COPD no PH"
    df["C5_early_COPD_no_PH_proxy"] = (df["label"] == 0)

    cohort_flag_cols = [c for c in df.columns if c.startswith("C") and c[1].isdigit()]
    summary = {}
    for c in cohort_flag_cols:
        sub = df[df[c]].copy()
        if len(sub) == 0:
            summary[c] = {"n": 0}
            continue
        mpap_arr = sub.loc[sub["measured_mpap_flag"], "measured_mpap"].astype(float).values
        summary[c] = {
            "n": int(len(sub)),
            "n_PH": int((sub["label"] == 1).sum()),
            "n_nonPH": int((sub["label"] == 0).sum()),
            "n_measured_mpap": int(sub["measured_mpap_flag"].sum()),
            "mpap_p25_p50_p75": (
                [float(np.percentile(mpap_arr, p)) for p in (25, 50, 75)]
                if len(mpap_arr) else None
            ),
            "mpap_min_max": (
                [float(mpap_arr.min()), float(mpap_arr.max())]
                if len(mpap_arr) else None
            ),
            "protocol_distribution": sub["protocol"].value_counts().to_dict(),
        }
    out_path = OUT / "clean_cohort_table.csv"
    df.to_csv(out_path, index=False)
    (OUT / "protocol_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
