"""01_build_clean_feature_table — Phase 1 of clean AUC boost.

Merge into one within-contrast feature table:
  A. SSL severity (R25.C ensemble OOF severity-percentile from outputs/r24/r25c_ensemble_oof.csv)
  B. Lung heterogeneity (lung_features_v2)
  C. Vascular topology (artery/vein len/tort/percentile from morph_unified301)
  D. Pruning / distribution (proxies via R17 morph + lung volume ratios)
  E. A/V imbalance (artery_vein_*_ratio computed from per-structure morph)
  F. TDA (R17.5 H0/H1 persistence — exclude airway pers all-zero)
  G. Benchmark (PA/Ao if present — currently absent in repo, listed for completeness)

Drop features with >30% missing. Save TWO tables: all_cohort + within_contrast.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
COHORT = ROOT / "outputs" / "supplementary" / "A0_data_audit" / "clean_cohort_table.csv"
SSL = ROOT / "outputs" / "r24" / "r25c_ensemble_oof.csv"
MORPH_UNIFIED = ROOT / "outputs" / "r20" / "morph_unified301.csv"
MORPH_LEGACY = ROOT / "outputs" / "r17" / "per_structure_morphometrics.csv"
LUNG = ROOT / "outputs" / "lung_features_v2.csv"
TDA = ROOT / "outputs" / "r17" / "per_structure_tda.csv"
OUT = ROOT / "outputs" / "clean_auc_boost"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    coh = pd.read_csv(COHORT)
    print(f"A0 locked cohort: n={len(coh)}")

    # === Feature group A: SSL severity (R25.C ensemble) ===
    ssl_cols = []
    if SSL.exists():
        ssl = pd.read_csv(SSL)[["case_id", "severity_pct_ensemble"]].rename(
            columns={"severity_pct_ensemble": "ssl_severity_score"}
        )
        # Per-seed scores for std
        ssl_full = pd.read_csv(SSL)
        seed_cols = [c for c in ssl_full.columns if c.startswith("sev_seed_")]
        if seed_cols:
            ssl["ssl_seed_mean"] = ssl_full[seed_cols].mean(axis=1)
            ssl["ssl_seed_std"] = ssl_full[seed_cols].std(axis=1)
        ssl_cols = [c for c in ssl.columns if c != "case_id"]
        print(f"  A. SSL severity: {len(ssl_cols)} cols ({len(ssl)} cases) — R25.C ensemble (within-contrast only)")
    else:
        ssl = pd.DataFrame({"case_id": []})

    # === Feature group B: lung heterogeneity ===
    lung_cols_keep = []
    if LUNG.exists():
        lung_raw = pd.read_csv(LUNG).drop(columns=["label"], errors="ignore")
        # numeric only
        lung_cols_keep = [c for c in lung_raw.columns if c != "case_id"
                          and pd.api.types.is_numeric_dtype(lung_raw[c])
                          and "placeholder" not in c.lower()]
        lung = lung_raw[["case_id"] + lung_cols_keep].copy()
        print(f"  B. Lung heterogeneity: {len(lung_cols_keep)} cols ({len(lung)} cases)")
    else:
        lung = pd.DataFrame({"case_id": []})

    # === Feature group C: vascular topology (artery/vein len + tort percentiles) ===
    BLACKLIST = {"airway_n_terminals", "airway_term_per_node",
                 "artery_lap_eig0", "artery_n_terminals", "artery_term_per_node",
                 "vein_lap_eig0", "vein_n_terminals", "vein_term_per_node",
                 "artery_placeholder", "vein_placeholder", "airway_placeholder"}
    morph = pd.read_csv(MORPH_UNIFIED).drop(
        columns=["label", "source_cache"], errors="ignore"
    )
    av_cols = [c for c in morph.columns
               if c.startswith(("artery_", "vein_"))
               and c not in BLACKLIST
               and "placeholder" not in c.lower()]
    morph_av = morph[["case_id"] + av_cols].copy()
    print(f"  C. Vascular topology (artery+vein): {len(av_cols)} cols ({len(morph_av)} cases) — unified Simple_AV_seg")

    # === Feature group D: pruning / distribution (computed proxies + per-patient α) ===
    # Audit point 7 fix: include per-patient pruning_alpha from C1 T4
    pruning_cols = []   # only legacy-R17 derived cols; alpha cols tracked separately
    alpha_cols = []
    legacy_pruning_extra = pd.DataFrame({"case_id": []})
    ALPHA_FILE = ROOT / "outputs" / "supplementary" / "C1_signature_severity" / "pruning_alpha_per_patient.csv"
    if ALPHA_FILE.exists():
        alpha_df = pd.read_csv(ALPHA_FILE)
        legacy_pruning_extra = alpha_df.copy()
        alpha_cols = ["artery_pruning_alpha", "vein_pruning_alpha"]
        print(f"  D-pre. Per-patient pruning α (R17 5-bin scheme): {len(alpha_df)} cases × 2 cols")
    if MORPH_LEGACY.exists():
        legacy = pd.read_csv(MORPH_LEGACY).drop(columns=["label"], errors="ignore")
        # peripheral / central proxies — use diam_p10/p90 ratios
        for struct in ("artery", "vein"):
            d10 = f"{struct}_diam_p10"
            d90 = f"{struct}_diam_p90"
            if d10 in legacy.columns and d90 in legacy.columns:
                with np.errstate(divide="ignore", invalid="ignore"):
                    legacy[f"{struct}_diam_peripheral_central_ratio_legR17"] = (
                        legacy[d10] / legacy[d90].replace(0, np.nan))
                pruning_cols.append(f"{struct}_diam_peripheral_central_ratio_legR17")
            l10 = f"{struct}_len_p10"; l90 = f"{struct}_len_p90"
            if l10 in legacy.columns and l90 in legacy.columns:
                with np.errstate(divide="ignore", invalid="ignore"):
                    legacy[f"{struct}_len_peripheral_central_ratio_legR17"] = (
                        legacy[l10] / legacy[l90].replace(0, np.nan))
                pruning_cols.append(f"{struct}_len_peripheral_central_ratio_legR17")
        legacy_pruning = legacy[["case_id"] + pruning_cols].copy()
        print(f"  D. Pruning ratios (computed): {len(pruning_cols)} cols")
    else:
        legacy_pruning = pd.DataFrame({"case_id": []})

    # === Feature group E: A/V imbalance ratios ===
    av_imbalance_cols = []
    if "artery_n_nodes" in morph.columns and "vein_n_nodes" in morph.columns:
        morph["av_node_ratio"] = morph["artery_n_nodes"] / morph["vein_n_nodes"].replace(0, np.nan)
        av_imbalance_cols.append("av_node_ratio")
    if "artery_total_len_mm" in morph.columns and "vein_total_len_mm" in morph.columns:
        morph["av_length_ratio"] = morph["artery_total_len_mm"] / morph["vein_total_len_mm"].replace(0, np.nan)
        av_imbalance_cols.append("av_length_ratio")
    if "artery_n_branches" in morph.columns and "vein_n_branches" in morph.columns:
        morph["av_branches_ratio"] = morph["artery_n_branches"] / morph["vein_n_branches"].replace(0, np.nan)
        av_imbalance_cols.append("av_branches_ratio")
    if "artery_tort_p50" in morph.columns and "vein_tort_p50" in morph.columns:
        morph["av_tortuosity_ratio"] = morph["artery_tort_p50"] / morph["vein_tort_p50"].replace(0, np.nan)
        av_imbalance_cols.append("av_tortuosity_ratio")
    morph_imbalance = morph[["case_id"] + av_imbalance_cols].copy()
    print(f"  E. A/V imbalance: {len(av_imbalance_cols)} cols (computed)")

    # === Feature group F: TDA (drop airway pers) ===
    tda_cols = []
    if TDA.exists():
        tda_raw = pd.read_csv(TDA).drop(columns=["label"], errors="ignore")
        tda_cols = [c for c in tda_raw.columns if c != "case_id"
                    and not c.startswith("airway_pers")
                    and pd.api.types.is_numeric_dtype(tda_raw[c])]
        tda = tda_raw[["case_id"] + tda_cols].copy()
        print(f"  F. TDA (artery+vein only): {len(tda_cols)} cols")
    else:
        tda = pd.DataFrame({"case_id": []})

    # === Feature group G: benchmark (PA/Ao) — currently absent ===
    benchmark_cols = []
    print(f"  G. Benchmark (PA/Ao, PA_diameter): {len(benchmark_cols)} cols — NOT PRESENT in repo")

    # ===== Merge all =====
    df = (coh[["case_id", "label", "protocol", "is_contrast_only_subset",
                "measured_mpap", "measured_mpap_flag", "fold_id"]]
          .rename(columns={"measured_mpap": "mpap"})
          .merge(ssl, on="case_id", how="left")
          .merge(lung, on="case_id", how="left")
          .merge(morph_av, on="case_id", how="left")
          .merge(legacy_pruning, on="case_id", how="left")
          .merge(legacy_pruning_extra, on="case_id", how="left")
          .merge(morph_imbalance, on="case_id", how="left")
          .merge(tda, on="case_id", how="left"))
    print(f"\nmerged ALL n={len(df)} × {df.shape[1]} cols")

    feat_cols = (ssl_cols + lung_cols_keep + av_cols + pruning_cols + alpha_cols
                 + av_imbalance_cols + tda_cols)
    feat_cols = [c for c in feat_cols if c in df.columns]
    print(f"raw feature cols: {len(feat_cols)}")

    # Drop features with >30% missing — compute on WITHIN-CONTRAST primary cohort (per spec)
    df_within_for_qc = df[df["is_contrast_only_subset"]]
    missing = df_within_for_qc[feat_cols].isna().mean()
    keep = missing[missing <= 0.30].index.tolist()
    dropped = missing[missing > 0.30].index.tolist()
    print(f"dropped {len(dropped)} features with >30% missing in WITHIN-CONTRAST cohort: "
          f"{dropped[:10]}{' ...' if len(dropped)>10 else ''}")
    feat_cols = keep

    # Feature group map
    group_map = {
        "A_ssl_severity": [c for c in ssl_cols if c in feat_cols],
        "B_lung_heterogeneity": [c for c in lung_cols_keep if c in feat_cols],
        "C_vascular_topology": [c for c in av_cols if c in feat_cols],
        "D_pruning_distribution": [c for c in (pruning_cols + alpha_cols) if c in feat_cols],
        "E_av_imbalance": [c for c in av_imbalance_cols if c in feat_cols],
        "F_tda": [c for c in tda_cols if c in feat_cols],
        "G_benchmark": [c for c in benchmark_cols if c in feat_cols],
    }
    (OUT / "feature_group_map.json").write_text(
        json.dumps({k: {"features": v, "n": len(v)}
                    for k, v in group_map.items()}, indent=2),
        encoding="utf-8")
    print("\nfeature group breakdown:")
    for k, v in group_map.items(): print(f"  {k}: {len(v)}")

    keep_meta = ["case_id", "label", "mpap", "protocol",
                 "is_contrast_only_subset", "measured_mpap_flag", "fold_id"]
    df_all = df[keep_meta + feat_cols].copy()
    df_all.to_csv(OUT / "clean_feature_table_all.csv", index=False)
    df_within = df_all[df_all["is_contrast_only_subset"]].copy()
    df_within.to_csv(OUT / "clean_feature_table_within_contrast.csv", index=False)

    summary = {
        "n_all": int(len(df_all)),
        "n_within_contrast": int(len(df_within)),
        "n_PH_within": int((df_within["label"]==1).sum()),
        "n_nonPH_within": int((df_within["label"]==0).sum()),
        "n_measured_mpap_within": int(df_within["measured_mpap_flag"].sum()),
        "n_features_total": len(feat_cols),
        "feature_group_n": {k: len(v) for k, v in group_map.items()},
    }
    (OUT / "cohort_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
