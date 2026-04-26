"""R25.A — Build extended 147D feature universe (morph + lung + TDA).

Closes Q2 lung+airway auxiliary by adding parenchyma + TDA. 212-case intersection.
Output: outputs/r24/extended_features_212.csv
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent.parent
MORPH = ROOT / "outputs" / "r20" / "morph_unified301.csv"
LUNG = ROOT / "outputs" / "lung_features_v2.csv"
TDA = ROOT / "outputs" / "r17" / "per_structure_tda.csv"
COHORT = ROOT / "outputs" / "r24" / "cohort_locked_table.csv"
OUT = ROOT / "outputs" / "r24"

ARTIFACTS = {"airway_n_terminals", "airway_term_per_node",
             "artery_lap_eig0", "artery_n_terminals", "artery_term_per_node",
             "vein_lap_eig0", "vein_n_terminals", "vein_term_per_node"}


def main():
    m = pd.read_csv(MORPH).drop(columns=["label", "source_cache"], errors="ignore")
    lung = pd.read_csv(LUNG).drop(columns=["label"], errors="ignore")
    tda = pd.read_csv(TDA).drop(columns=["label"], errors="ignore")

    # Drop artifacts
    m = m[[c for c in m.columns if c not in ARTIFACTS]]
    print(f"morph: {m.shape}, lung: {lung.shape}, tda: {tda.shape}")

    # Intersection on case_id
    df = m.merge(lung, on="case_id", how="inner").merge(tda, on="case_id", how="inner")
    print(f"intersection: {df.shape}")

    # Merge in cohort lock fields
    coh = pd.read_csv(COHORT)
    df = df.merge(coh[["case_id", "label", "protocol", "is_contrast_only_subset",
                         "measured_mpap", "measured_mpap_flag", "fold_id"]],
                  on="case_id", how="inner")
    feat_cols = [c for c in df.columns if c not in
                 ("case_id", "label", "protocol", "is_contrast_only_subset",
                  "measured_mpap", "measured_mpap_flag", "fold_id")]
    print(f"after cohort merge: {df.shape}, feature_count={len(feat_cols)}")
    print(f"  PH={int((df.label==1).sum())} nonPH={int((df.label==0).sum())}")
    print(f"  within-contrast: {int(df.is_contrast_only_subset.sum())}")
    print(f"  measured-mPAP: {int(df.measured_mpap_flag.sum())}")

    out_path = OUT / "extended_features_212.csv"
    df.to_csv(out_path, index=False)
    print(f"saved {out_path}")

    # Diagnostic: feature breakdown
    morph_n = sum(c.startswith(("artery_","vein_","airway_")) and not c.endswith("_x") and not c.endswith("_y") and "_pers" not in c for c in feat_cols)
    tda_n = sum("_pers" in c for c in feat_cols)
    lung_n = sum(c.startswith(("paren_", "lung_", "apical")) for c in feat_cols)
    print(f"breakdown: morph={morph_n}, tda={tda_n}, lung={lung_n}")


if __name__ == "__main__":
    main()
