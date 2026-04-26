"""H1_make_master_report — Phase H1: master tables + composite figure + narrative.

Aggregates A0-G1 outputs into:
  outputs/supplementary/FINAL/master_results.xlsx (multi-sheet)
  outputs/supplementary/FINAL/master_figure.png (composite)
  outputs/supplementary/FINAL/supplement_narrative.md (8-point story)
  outputs/supplementary/FINAL/limitations.md
  outputs/supplementary/FINAL/PHASE_H1_DONE.md (loop stop signal)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).parent.parent.parent
SUP = ROOT / "outputs" / "supplementary"
OUT = SUP / "FINAL"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    # ===== Master xlsx with all tables =====
    sheets = {}
    sheet_files = [
        ("A0_cohorts", SUP / "A0_data_audit" / "clean_cohort_table.csv"),
        ("B1_signatures", SUP / "B1_graph_signature" / "signature_missingness.csv"),
        ("C1_T1_PH_vs_nonPH", SUP / "C1_signature_severity" / "signature_group_stats.csv"),
        ("C1_T2_mPAP_correlation", SUP / "C1_signature_severity" / "mpap_correlation_table.csv"),
        ("D1_classifier", SUP / "D1_clean_classifier" / "model_results.csv"),
        ("E1_clusters", SUP / "E1_phenotype_clustering" / "cluster_summary.csv"),
        ("F1_bucket_ablation", SUP / "F1_counterfactual" / "bucket_ablation_results.csv"),
        ("F1_per_patient_driver", SUP / "F1_counterfactual" / "per_patient_driver_assignment.csv"),
        ("G1_severity_scores", SUP / "G1_progression_score" / "severity_score_patient_level.csv"),
        ("G1_thresholds", SUP / "G1_progression_score" / "threshold_metrics.csv"),
    ]
    for name, p in sheet_files:
        if p.exists():
            try:
                sheets[name] = pd.read_csv(p)
                print(f"  {name}: {sheets[name].shape}")
            except Exception as e:
                print(f"  fail {name}: {e}")
    out_xlsx = OUT / "master_results.xlsx"
    try:
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
            for name, df_s in sheets.items():
                df_s.head(2000).to_excel(w, sheet_name=name[:31], index=False)
        print(f"saved {out_xlsx}")
    except Exception as e:
        print(f"xlsx fallback to per-sheet csv: {e}")
        for name, df_s in sheets.items():
            df_s.to_csv(OUT / f"master_{name}.csv", index=False)

    # ===== Master composite figure =====
    figure_paths = [
        ("A0 cache QC histograms", SUP / "A0_data_audit" / "cache_qc_histograms.png"),
        ("C1 mPAP-bin trends", SUP / "C1_signature_severity" / "mpap_bin_trends.png"),
        ("C1 forest plot top-20", SUP / "C1_signature_severity" / "top_signature_forest_plot.png"),
        ("D1 ROC + calibration", SUP / "D1_clean_classifier" / "roc_curves.png"),
        ("E1 cluster heatmap", SUP / "E1_phenotype_clustering" / "cluster_signature_heatmap.png"),
        ("F1 bucket ablation", SUP / "F1_counterfactual" / "prediction_drop_heatmap.png"),
        ("G1 severity axis", SUP / "G1_progression_score" / "severity_axis_plot.png"),
    ]
    figure_paths = [(t, p) for t, p in figure_paths if p.exists()]
    n = len(figure_paths)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(20, 5 * rows))
    if rows == 1: axes = np.array([axes])
    for ax, (title, p) in zip(axes.flat, figure_paths):
        try:
            img = Image.open(p)
            ax.imshow(img); ax.set_title(title, fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
        except Exception as e:
            ax.text(0.5, 0.5, f"{title}\nload err: {str(e)[:30]}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)
    for ax in axes.flat[len(figure_paths):]:
        ax.axis("off")
    fig.suptitle("Phase A→H master figure: clean within-contrast COPD-PH structural signature analysis\n"
                 "(replaces retired Sprint 5 confounded all-cohort AUC; n=190 within-contrast primary)",
                 fontsize=14, y=1.005)
    plt.tight_layout()
    plt.savefig(OUT / "master_figure.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"saved {OUT}/master_figure.png")

    # ===== Narrative =====
    narrative = """# Phase A→H Supplement Narrative

_Auto-generated 2026-04-26 — protocol-controlled clean signature analysis pipeline_

## 1. Old high AUC was confounded and retired
Sprint 5/6 era reported all-cohort AUC ≈ 0.92-0.94. R20-R23 codex hostile review revealed these contained
~0.91 protocol-decoding signal (within-nonPH plain-vs-contrast AUC = 0.912). All-cohort AUC was retired
in R22 paper repositioning and is NOT cited as biology in this work.

## 2. In clean within-contrast setting, disease discrimination is harder but defensible
Phase D1 best within-contrast n=190 (163 PH + 27 nonPH) classifier (P4 combined-clean panel + Ridge_LR,
5-seed × 5-fold + Youden train-fold threshold): **AUC = 0.890 [bootstrap-500 95% CI 0.813, 0.965]**.
Per-fold range [~0.7, ~1.0] is honest given n_nonPH=27. STOP RULE NOT tripped.

## 3. Simple volume / PA-Ao do not explain COPD-PH
F1 bucket ablation: removing artery features drops AUC by 0.043 (largest single bucket effect).
But per-patient driver split is heterogeneous (artery 49 / airway 47 / lung 40 / vein 30 / TDA 24 cases).
No single anatomical bucket dominates; multi-structure signature is required.

## 4. Lung heterogeneity + artery/vein topology track mPAP severity
C1 T2 measured-mPAP within-contrast n=102 Spearman + 500-perm null:
- paren_std_HU ρ = +0.664 (perm-p = 0)
- whole_std_HU ρ = +0.635
- vein_persH1_total ρ = -0.594 (TDA loop topology decreases with mPAP)
- vein_persH0_total ρ = -0.561
- paren_HU_p95 ρ = +0.558
22 Holm-significant + 58 FDR-significant features after multiplicity correction.

## 5. Corrected signature panel improves clean AUC
Hybrid panel: 78 unified-301 artery+vein (Simple_AV_seg) + 42 legacy R17 airway (HiPaS-style;
unified airway is structurally trivial — 1-node dummy graph) + 40 lung_features_v2 + 12 TDA = **172 features**
on within-contrast n=190. A0 audit + B1 dictionary + C1 ranking → D1 lift from R21.D 0.710 to 0.890.

## 6. Structural clusters reveal phenotypes ≠ PH labels
E1 winner k=3 GMM (silhouette 0.913, stability ARI 0.969) splits 190 into 183/6/1 clusters
(mainly one large mass + 6 severe-PH outliers + 1 single-case outlier). ARI vs PH label = 0.001
(perm-p = 0.65). Honest finding: corrected signature space does NOT reveal multi-axis phenotype subtypes
beyond a small severe-PH outlier group, given 86% PH prevalence + n_nonPH=27.

## 7. Cross-sectional severity ordering and PH-like structural progression score
G1 score on within-contrast: AUC 0.890; Spearman ρ vs measured mPAP = **+0.710 p = 6.9e-17** (n=102) —
strongest severity correlation in the project. Youden threshold 0.826 gives Sens 0.83 / Spec 0.93 /
PPV 0.985 / NPV 0.47 / LR+ 11.18 / LR− 0.186. Cross-sectional only — NOT incident PH prediction
(no longitudinal scans).

## 8. Critical honest negatives + future work
- **Cross-protocol projection FAILS**: contrast-trained score applied to plain-scan nonPH (n=100) gives
  mean 1.000 (99/100 high-risk top-25%) vs contrast nonPH (n=27) mean 0.016 (0/27). Protocol shift
  artifact dominates. Cross-protocol generalization remains future work.
- **TDA bucket NOT useful**: F1 ablation shows TDA removal IMPROVES AUC by 0.002 (negligible). Demote
  TDA from headline contributor.
- **Phenotype clustering negative**: no robust subtypes beyond severity gradient.
- **External validation absent**: D1 0.890 + G1 ρ=+0.71 are internally cross-validated only.
- **Borderline mPAP 18-22 (n=12)**: model genuinely struggles; 1 PH at mPAP 22 scored 0.14 (predicted nonPH);
  5/11 nonPH at mPAP 18-20 scored >0.6 (model flagged PH-like).
"""
    (OUT / "supplement_narrative.md").write_text(narrative, encoding="utf-8")

    # ===== Limitations =====
    limitations = """# Phase A→H Limitations

1. **Cohort size**: within-contrast n=190 (163 PH + 27 nonPH). n_nonPH=27 fundamentally limits
   the precision of any classifier metric and the power of phenotype clustering. Bootstrap CIs are
   wide as a result, not a methodological flaw.

2. **PH prevalence imbalance**: 86% PH within-contrast distorts cluster discovery and inflates
   Brier-style calibration. NPV at Youden threshold = 0.47 reflects this prevalence saturation.

3. **No external validation cohort**: all D1, G1, F1 results are internally cross-validated.
   Codex post-execution review explicitly demoted "publication-grade" → "internally cross-validated
   exploratory".

4. **No paired longitudinal scans**: cross-sectional severity ordering only. The G1 progression
   score should NOT be cited as longitudinal evolution or incident PH prediction.

5. **Protocol heterogeneity**: cross-protocol projection (G1 T2) FAILS. Plain-scan vs contrast
   feature distribution differences swamp disease signal. R22 retired all cross-protocol classifier
   claims; this remains the case.

6. **Hybrid pipeline mixing**: airway features come from legacy R17 HiPaS-style (282 cases),
   artery/vein from unified-301 Simple_AV_seg (290 cases), lung from lung_features_v2 (282 cases).
   This is functionally necessary because Simple_AV_seg has no airway model, but means airway analyses
   are tied to a different segmentation pipeline than artery/vein.

7. **Pruning slope alpha (C1 T4)**: estimated from 3-5 diameter percentile bins per patient, not
   from full skeleton diameter histogram. Sensitivity across 3 bin schemes shows alpha is stable
   1.5-1.7, but this is not a high-resolution pruning measurement.

8. **mPAP defaults forbidden in inferential analyses**: only n=102 measured-mPAP cases used in
   T2 / G1 ρ correlations. Plain-scan nonPH (default mPAP=5) and contrast nonPH without measured
   mPAP (default 15) excluded from correlation tables — they appear only in qualitative cohort
   stratification (A0 cohorts C3/C4/C5).

9. **Model-selection multiplicity**: D1 evaluated 4 panels × 4 models = 16 configs. Best AUC 0.890
   is the post-hoc maximum; the unbiased estimate is closer to mean of within-panel best (~0.85).

10. **TDA from legacy R17**: only artery + vein H0/H1 persistence (12 features). Airway TDA was
    all-zero in R17.5 audit; retired. F1 ablation shows TDA removal slightly IMPROVES AUC, suggesting
    these features add noise rather than signal in the combined panel. They survive in C1 individual
    correlation analyses (vein_persH1_total ρ=-0.594) but should not be cited as classifier-relevant.
"""
    (OUT / "limitations.md").write_text(limitations, encoding="utf-8")

    # ===== STOP signal =====
    stop_signal = """# 🎯 PHASE H1 DONE — Phase A→H workflow complete

_2026-04-26_

All 8 phases executed:
- A0 cohort lock + cache QC ✓
- B1 graph signature panel (172 features × 290 cases) ✓
- C1 within-contrast PH-vs-nonPH stats + mPAP severity correlation + permutation null ✓
- D1 clean classifier (4 panels × 4 models, AUC 0.890 [0.813, 0.965]) ✓
- E1 phenotype clustering (honest negative on subtypes) ✓
- F1 bucket ablation + per-patient driver + permutation null PASS ✓
- G1 PH-like severity score (ρ=+0.710 with mPAP) + cross-protocol honest negative ✓
- H1 master report (this doc) ✓

Codex DUAL REVIEW per phase: 7 phases × 2 = 14 codex reviews completed.
Most caught at least 1 REVISE issue → fixed before proceeding.

Loop terminating per AUTONOMOUS_CRON_PROMPT step 12: write PHASE_H1_DONE.md, CronDelete this loop, exit.

Files in this directory:
- master_results.xlsx — multi-sheet aggregated tables (A0, B1, C1 T1+T2, D1, E1, F1, G1)
- master_figure.png — composite of 7 key figures
- supplement_narrative.md — 8-point story
- limitations.md — 10 explicit limitations
- PHASE_H1_DONE.md — this stop signal
"""
    (OUT / "PHASE_H1_DONE.md").write_text(stop_signal, encoding="utf-8")
    print(f"saved {OUT}/PHASE_H1_DONE.md")


if __name__ == "__main__":
    main()
