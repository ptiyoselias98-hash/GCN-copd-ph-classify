"""B1_extract_graph_signatures — Phase B1: assemble patient-level signature panel.

Hybrid data sources per A0 critical finding:
  artery + vein  → outputs/r20/morph_unified301.csv (Simple_AV_seg unified, 290 cases)
  airway         → outputs/r17/per_structure_morphometrics.csv (legacy R17 HiPaS, 282 cases × 44 airway feats)
  lung parenchyma→ outputs/lung_features_v2.csv (282 cases × 51 feats)
  TDA H0/H1      → outputs/r17/per_structure_tda.csv (282 cases × 18 feats; airway TDA all-zero, drop)

Output:
  outputs/supplementary/B1_graph_signature/graph_signatures_patient_level.csv
  outputs/supplementary/B1_graph_signature/graph_signature_dictionary.json
  outputs/supplementary/B1_graph_signature/signature_missingness.csv

Hard blacklist: R17 numerical artifacts (n_terminals=0, lap_eig0~1e-18, term_per_node trivially-zero):
  airway_n_terminals, airway_term_per_node,
  artery_lap_eig0, artery_n_terminals, artery_term_per_node,
  vein_lap_eig0, vein_n_terminals, vein_term_per_node
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
COHORT = ROOT / "outputs" / "supplementary" / "A0_data_audit" / "clean_cohort_table.csv"
MORPH_UNIFIED = ROOT / "outputs" / "r20" / "morph_unified301.csv"
MORPH_LEGACY_R17 = ROOT / "outputs" / "r17" / "per_structure_morphometrics.csv"
LUNG = ROOT / "outputs" / "lung_features_v2.csv"
TDA = ROOT / "outputs" / "r17" / "per_structure_tda.csv"
OUT = ROOT / "outputs" / "supplementary" / "B1_graph_signature"
OUT.mkdir(parents=True, exist_ok=True)

ARTIFACT_BLACKLIST = {
    "airway_n_terminals", "airway_term_per_node",
    "artery_lap_eig0", "artery_n_terminals", "artery_term_per_node",
    "vein_lap_eig0", "vein_n_terminals", "vein_term_per_node",
    # placeholder columns from R20.G builder (always 0 / NaN)
    "artery_placeholder", "vein_placeholder", "airway_placeholder",
    "airway_n_terminals_legR17", "airway_term_per_node_legR17",
}


def categorize_feature(col: str) -> str:
    """Bucket signatures per Phase B1 spec categories 1-8.

    Note: lowercase the input to make string match case-insensitive (was a bug
    where uppercase HU/LAA never matched and dumped lung features into "0_other").
    """
    raw = col
    c = col.lower()
    if "_pers" in c:
        return "8_TDA_persistence"
    if c.startswith("paren_") or c.startswith("lung_") or "apical" in c or "_hu" in c or "laa" in c:
        return "7_lung_parenchyma"
    if "_diam" in c or "_len" in c or "_tort" in c or "_curv" in c:
        return "3_diameter_length_distribution"
    if ("_n_nodes" in c or "_n_edges" in c or "_total_len" in c or "_total_vol" in c
        or "_vol_ml" in c or "_n_voxels" in c or c == "vessel_airway_over_lung"
        or c == "vessel_airway_vol_ml"):
        return "1_basic_graph_size"
    if any(s in c for s in ("_degree", "_n_branches", "_branch_per_node",
                              "_n_components", "_max_tree_depth", "_path_length",
                              "_skew", "_kurt", "_sd", "_p10", "_p25", "_p50", "_p75", "_p90", "_mean",
                              "_tortuosity_proxy", "_max_degree", "_mean_degree",
                              "_lap_eig")):
        return "2_branching_topology"
    return "0_other"


def main():
    coh = pd.read_csv(COHORT)
    print(f"cohort A0: n={len(coh)}")

    # Source 1: artery + vein from unified-301 (drop airway placeholder + blacklist)
    m_uni = pd.read_csv(MORPH_UNIFIED).drop(columns=["label", "source_cache"], errors="ignore")
    art_vein_cols = [c for c in m_uni.columns if c.startswith(("artery_", "vein_"))
                     and c not in ARTIFACT_BLACKLIST
                     and "placeholder" not in c.lower()]
    df_av = m_uni[["case_id"] + art_vein_cols].copy()
    print(f"  unified-301 artery+vein: {len(art_vein_cols)} cols")

    # Source 2: airway from legacy R17 (real graph)
    m_leg = pd.read_csv(MORPH_LEGACY_R17).drop(columns=["label"], errors="ignore")
    airway_cols = [c for c in m_leg.columns if c.startswith("airway_")
                   and c not in ARTIFACT_BLACKLIST]
    df_aw = m_leg[["case_id"] + airway_cols].copy()
    df_aw = df_aw.rename(columns={c: c + "_legR17" for c in airway_cols})
    airway_cols_renamed = [c + "_legR17" for c in airway_cols]
    print(f"  legacy R17 airway: {len(airway_cols)} cols (suffixed _legR17)")

    # Source 3: lung
    lung = pd.read_csv(LUNG).drop(columns=["label"], errors="ignore")
    # Drop string-typed path-like columns + placeholder columns
    lung_cols = [c for c in lung.columns if c != "case_id"
                 and pd.api.types.is_numeric_dtype(lung[c])
                 and "placeholder" not in c.lower()]
    df_lung = lung[["case_id"] + lung_cols].copy()
    print(f"  lung_features_v2: {len(lung_cols)} cols")

    # Source 4: TDA (drop airway H0/H1 = all zero per A0 audit)
    tda = pd.read_csv(TDA).drop(columns=["label"], errors="ignore")
    tda_cols = [c for c in tda.columns if c != "case_id"
                and not c.startswith("airway_pers")
                and pd.api.types.is_numeric_dtype(tda[c])]
    df_tda = tda[["case_id"] + tda_cols].copy()
    print(f"  TDA non-airway: {len(tda_cols)} cols")

    # Merge on case_id (intersection)
    df = (coh[["case_id", "label", "protocol", "is_contrast_only_subset",
                "measured_mpap", "measured_mpap_flag", "fold_id",
                "C1_all_available", "C2_within_contrast_only",
                "C3_borderline_mPAP_18_22", "C4_clear_low_high",
                "C5_early_COPD_no_PH_proxy"]]
          .merge(df_av, on="case_id", how="left")
          .merge(df_aw, on="case_id", how="left")
          .merge(df_lung, on="case_id", how="left")
          .merge(df_tda, on="case_id", how="left"))
    print(f"merged: {df.shape}")

    feature_cols = (art_vein_cols + airway_cols_renamed + lung_cols + tda_cols)
    print(f"total features: {len(feature_cols)}")

    # Missingness audit
    miss_rows = []
    for c in feature_cols:
        n_total = len(df)
        n_missing = int(df[c].isna().sum())
        n_finite = int(df[c].dropna().shape[0])
        sd = float(df[c].std()) if n_finite >= 2 else 0.0
        unique = int(df[c].nunique(dropna=True))
        miss_rows.append({
            "feature": c,
            "category": categorize_feature(c),
            "n_total": n_total,
            "n_missing": n_missing,
            "missing_frac": n_missing / n_total if n_total else 0.0,
            "n_finite": n_finite,
            "std": sd,
            "n_unique": unique,
            "near_zero_sd": bool(sd < 1e-6),
            "constant": bool(unique <= 1),
        })
    miss_df = pd.DataFrame(miss_rows)
    miss_df.to_csv(OUT / "signature_missingness.csv", index=False)

    # Drop near-zero-SD / constant after blacklist
    drop_cols = miss_df[(miss_df["near_zero_sd"]) | (miss_df["constant"])]["feature"].tolist()
    feature_cols_clean = [c for c in feature_cols if c not in drop_cols]
    print(f"after near-zero-SD/constant drop: {len(feature_cols_clean)} cols (dropped {len(drop_cols)})")

    # Save patient-level signature CSV
    keep_cols = ["case_id", "label", "protocol", "is_contrast_only_subset",
                  "measured_mpap", "measured_mpap_flag", "fold_id",
                  "C1_all_available", "C2_within_contrast_only",
                  "C3_borderline_mPAP_18_22", "C4_clear_low_high",
                  "C5_early_COPD_no_PH_proxy"] + feature_cols_clean
    df_sig = df[keep_cols].copy()
    df_sig.to_csv(OUT / "graph_signatures_patient_level.csv", index=False)

    # Dictionary: feature → source + category + brief description
    # Order: TDA persistence first (matches "_pers" before artery_/vein_ prefix bug fix)
    dictionary = {}
    for c in feature_cols_clean:
        if "_pers" in c:
            src = "R17.5_TDA_gudhi"
        elif c.endswith("_legR17"):
            src = "legacy_R17_HiPaS_style_pipeline"
        elif c.startswith(("artery_", "vein_")):
            src = "unified_301_Simple_AV_seg"
        elif (c.startswith(("paren_", "lung_")) or "_hu" in c.lower() or "laa" in c.lower()
              or "apical" in c.lower()):
            src = "lung_features_v2"
        elif c in lung_cols:
            # catch-all for lung_features_v2 columns that don't match prefix
            # (e.g. basal_n_voxels, middle_n_voxels, whole_n_voxels, vessel_airway_*)
            src = "lung_features_v2"
        else:
            src = "unknown"
        dictionary[c] = {
            "source": src,
            "category": categorize_feature(c),
            "blacklisted_R17_artifact": False,
        }
    (OUT / "graph_signature_dictionary.json").write_text(
        json.dumps({"n_features_clean": len(feature_cols_clean),
                    "n_blacklisted_artifacts": len(ARTIFACT_BLACKLIST),
                    "n_dropped_near_zero_or_constant": len(drop_cols),
                    "category_breakdown": pd.Series(
                        [v["category"] for v in dictionary.values()]
                    ).value_counts().to_dict(),
                    "source_breakdown": pd.Series(
                        [v["source"] for v in dictionary.values()]
                    ).value_counts().to_dict(),
                    "feature_dictionary": dictionary},
                   indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"saved: {OUT}/graph_signatures_patient_level.csv")
    print(f"saved: {OUT}/graph_signature_dictionary.json")
    print(f"saved: {OUT}/signature_missingness.csv")
    print(f"\nfinal panel: n={len(df_sig)} cases × {len(feature_cols_clean)} features")
    print("category breakdown:")
    print(pd.Series([v["category"] for v in dictionary.values()]).value_counts())


if __name__ == "__main__":
    main()
