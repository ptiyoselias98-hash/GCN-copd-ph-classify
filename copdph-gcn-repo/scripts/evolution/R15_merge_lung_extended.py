"""R15.F — Merge legacy + new100 lung-feature CSVs into one extended panel.

Schema-aligns columns (legacy lung_features_v2.csv has airway, the new
extractor sets airway_vol_mL=0.0 since Simple_AV_seg doesn't produce
airway masks). Emits outputs/r15/lung_features_extended.csv with an
`is_new_ingestion` flag.

For the 22 case_ids present in BOTH legacy and new (refilled placeholders):
prefer the NEW extraction (real masks vs prior placeholder).
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
LEGACY = ROOT / "outputs" / "lung_features_v2.csv"
NEW = ROOT / "outputs" / "r15" / "lung_features_new100.csv"
OUT = ROOT / "outputs" / "r15" / "lung_features_extended.csv"


def main():
    leg = pd.read_csv(LEGACY)
    new = pd.read_csv(NEW)
    print(f"legacy rows: {len(leg)}, new rows: {len(new)}")
    print(f"legacy cols: {len(leg.columns)}, new cols: {len(new.columns)}")
    overlap = set(leg["case_id"]) & set(new["case_id"])
    print(f"overlap (refilled placeholders): {len(overlap)}")

    leg["is_new_ingestion"] = 0
    new["is_new_ingestion"] = 1
    # Drop legacy rows that are in new (prefer real masks)
    leg_keep = leg[~leg["case_id"].isin(overlap)].copy()
    print(f"legacy rows kept: {len(leg_keep)}")

    # Align columns: union, missing → NaN
    all_cols = sorted(set(leg.columns) | set(new.columns))
    leg_keep = leg_keep.reindex(columns=all_cols)
    new = new.reindex(columns=all_cols)
    ext = pd.concat([leg_keep, new], ignore_index=True)
    ext.to_csv(OUT, index=False)
    print(f"saved {len(ext)} rows × {len(ext.columns)} cols → {OUT}")
    # Audit by group
    audit = ext[["case_id", "is_new_ingestion"]].copy()
    audit["is_ph"] = audit["case_id"].str.startswith("ph_")
    audit["is_nonph"] = audit["case_id"].str.startswith("nonph_")
    grp = audit.groupby(["is_new_ingestion", "is_ph"]).size()
    print("breakdown:")
    print(grp.to_string())


if __name__ == "__main__":
    main()
