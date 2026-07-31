# COPD-PH ML Classification — Reproduction Results

**Date:** 2026-07-31  
**Data:** 100 patients with complete CT quantitative features (74 PH, 26 nonPH)  
**Source:** copd-ph患者113例0331.xlsx, Cols 250–424 (commercial CT radiomics)  
**Label leakage excluded:** mPAP, PVSP, PA, PVR, NT-ProBNP, 6MWD, PA_diameter, mRAP, mPAWP, CO, CI

---

## Data Input — Three Feature Sets

### CT_all (175 features, all from commercial CT software)

| Group | #Features | Description |
|-------|-----------|-------------|
| Mean Density (HU) | 42 | CT density of lung/artery/vein per lobe — reflects tissue destruction + vessel filling |
| Volume (ml) | 37 | Lung/lobe volume, vessel volume, airway volume, tissue/non-tissue partitioning |
| Vessel Tortuosity | 14 | Artery/vein tortuosity per lobe — captures vascular remodeling |
| Branch Count | 13 | Vessel branch count per structure per lobe — distal pruning signal |
| LAA (emphysema) | 12 | Low-attenuation area (LAA910, LAA950) as % and ml — emphysema quantification |
| BV10 (vessel <10mm) | 6 | Blood volume in mid/small vessels |
| Calcification | 6 | Agatston calcium score, calcification volume |
| Fractal Dimension | 4 | Vascular tree complexity |
| BV5 (small vessel <5mm) | 3 | Blood volume in distal small vessels — key PH marker |
| Density StdDev | 3 | CT density heterogeneity |
| Mass (g) | 3 | Lung mass |
| Pi10 (airway) | 1 | Airway wall thickness at internal perimeter 10mm |
| Other | 31 | Branch wall area %, diameter metrics, lobe-level counts |

### Clinical_safe (11 features, all non-diagnostic labs)

| Feature | Full Name | Normal Range | Unit | Rationale |
|---------|-----------|-------------|------|-----------|
| Age | Age | — | years | COPD prevalence ↑ with age |
| WBC | White Blood Cells | 4.0–10.0 | ×10⁹/L | Chronic inflammation in COPD |
| RBC | Red Blood Cells | 4.0–5.5 / 3.5–5.0 | ×10¹²/L | Hypoxia → compensatory polycythemia |
| HB | Hemoglobin | 120–160 / 110–150 | g/L | Hypoxia compensation |
| PLT | Platelets | 100–300 | ×10⁹/L | Endothelial injury → consumption |
| ALT | ALT | 0–40 | U/L | Right heart failure → hepatic congestion |
| AST | AST | 0–40 | U/L | Same; AST/ALT ratio ↑ in cardiac injury |
| Crea | Creatinine | 44–133 | μmol/L | Renal hypoperfusion in PH |
| UA | Uric Acid | 150–420 | μmol/L | Tissue hypoxia → purine metabolism ↑ |
| D-Dimer | D-Dimer | <500 | μg/L | Pulmonary microthrombosis |
| cTnT | hs-cTnT | <14 | ng/L | RV wall stress → myocardial injury |

17 additional clinical features dropped due to >70% missing data:
BMI, GOLD grade, smoking history, FVC%, FEV1%, FEV1FVC, DLCO_SB%, DLCO_VA%, TLC%, RV%, RVTLC, LVEF, TAPSE, RVTD, RVLD, RVWDd, Aortic diameter.

### Combined_safe = CT_all (175) + Clinical_safe (11) = 186 features

---

## Methods

- **5-fold stratified CV** (seed=42, sklearn `StratifiedKFold`)
- Models: LogisticRegression (L2, C=1), RandomForest (n=200, depth=8), XGBoost (n=200, depth=5), SVM (RBF), MLP (64→32)
- **No imputation leakage**: NaN imputed per-column with median, computed on training folds only
- **No dimensionality reduction**: all 175 CT features used directly (each is a physically meaningful measurement)
- Metrics per fold: AUC, Accuracy, Precision, Sensitivity (Recall), F1, Specificity — plus pooled AUC across all fold predictions

---

## Results — Main Table

**5-fold CV, mean ± std. Sorted by AUC ↓**

| # | Model | Features | AUC | Acc | Sens | Spec | F1 | Prec | Pooled AUC |
|---|-------|----------|-----|-----|------|------|-----|------|------------|
| 1 | **XGBoost** | Combined_safe | **0.961** ±.022 | 0.880 | 0.946 | 0.693 | 0.921 | 0.898 | 0.947 |
| 2 | LogisticRegression | CT_all | 0.948 ±.021 | 0.870 | 0.879 | **0.847** | 0.908 | 0.945 | 0.942 |
| 3 | XGBoost | CT_all | 0.939 ±.021 | 0.870 | 0.919 | 0.733 | 0.912 | 0.908 | 0.929 |
| 4 | LogisticRegression | Combined_safe | 0.930 ±.031 | 0.840 | 0.865 | 0.767 | 0.889 | 0.918 | 0.937 |
| 5 | RandomForest | Combined_safe | 0.893 ±.051 | 0.870 | 0.946 | 0.653 | 0.915 | 0.889 | 0.886 |
| 6 | SVM_RBF | CT_all | 0.886 ±.027 | 0.850 | 0.906 | 0.693 | 0.900 | 0.895 | 0.877 |
| 7 | RandomForest | CT_all | 0.871 ±.015 | 0.850 | 0.932 | 0.613 | 0.901 | 0.875 | 0.866 |
| 8 | XGBoost | Clinical_safe | 0.777 ±.102 | 0.760 | 0.812 | 0.613 | 0.825 | 0.858 | 0.779 |

**Best overall: XGBoost + Combined_safe — AUC 0.961, Sens 0.946**
**Best specificity: LogisticRegression + CT_all — AUC 0.948, Spec 0.847**

---

## Feature Importance — Top 10 (Permutation Importance, XGBoost)

| Rank | Feature | Category | ΔAUC |
|------|---------|----------|------|
| 1 | Left Artery (LUL) Volume | Volume | +0.0216 |
| 2 | Left Artery (LLL) Tortuosity | Vessel Tortuosity | +0.0131 |
| 3 | Left Artery (LLL) Branch Count | Branch Count | +0.0122 |
| 4 | Left Artery (RLL) Branch Count | Branch Count | +0.0102 |
| 5 | Left Artery (RLL) Volume | Volume | +0.0101 |
| 6 | Left Artery (LUL) Volume | Volume | +0.0099 |
| 7 | Right Lung Density StdDev | Density StdDev | +0.0066 |
| 8 | Total Branch Count | Other | +0.0049 |
| 9 | Left Vein (RML) Branch Count | Branch Count | +0.0044 |
| 10 | Left Vein (RML) Min Density | Mean Density | +0.0038 |

### Category-Level Importance

| Category | Total ΔAUC | Interpretation |
|----------|--------|---------------|
| **Volume** | **0.0462** | Pulmonary artery enlargement = PH hallmark |
| **Branch Count** | **0.0310** | Distal vessel pruning → fewer visible branches |
| **Vessel Tortuosity** | **0.0198** | Vascular remodeling → tortuous, twisted vessels |
| Mean Density | 0.0193 | Combined emphysema + vessel density signal |
| Density StdDev | 0.0093 | Regional heterogeneity |
| LAA (emphysema) | 0.0033 | Emphysema alone is weak — PH adds vascular signal |
| BV5 (<5mm) | 0.0012 | Redundant with branch count (collinear) |
| BV10 (<10mm) | 0.0011 | Same — information already captured by volume + branches |

**Core finding:** Vessel volume + branch count + tortuosity account for ~50% of total feature importance. This is the CT signature of PH — **enlarged proximal pulmonary arteries with distal pruning and increased tortuosity**.

---

## Caveats & Limitations

- **Sample imbalance**: 74 PH vs 26 nonPH (2.85:1). Specificity suffers across all models.
- **Protocol confound**: All data is contrast-enhanced CT. Non-contrast scans would shift the distribution.
- **Single-center**: All data from one hospital. External validation needed.
- **Missing clinical data**: PFT (lung function), echocardiography missing for >97% of patients. Including these would likely improve specificity.
- **Not for clinical use**: Internal cross-validation only. Prospective validation required.
- **Honest label-leakage audit performed**: mPAP/PVSP/PA/PVR/NT-ProBNP/6MWD excluded. Including them artificially inflates AUC to 1.0.

---

## File Inventory

```
reproduction/
├── README.md                        ← This file
├── ml_classify.py                   ← Main classification script (reproducible)
├── feature_importance.py            ← Feature importance analysis
├── step1_prepare_data.py            ← DICOM→NIfTI converter (not needed for ML)
├── step2_build_graphs.py            ← Graph builder (GCN mode, not needed for ML)
├── step3_train_gcn.py               ← GCN trainer (not needed for ML)
├── labels.csv                       ← Patient ID → label mapping
├── patient_manifest.json            ← Full patient manifest
├── outputs/
│   ├── ml_classification/
│   │   ├── ml_results.json          ← All 20 experiments, 6 metrics × folds
│   │   ├── ml_results_summary.xlsx  ← Excel summary table
│   │   ├── roc_curves.png           ← ROC curves (models × feature sets)
│   │   └── metrics_comparison.png   ← 6-metric bar chart comparison
│   └── feature_importance/
│       ├── feature_importance.json  ← Top 50 permutation importance + XGBoost gain
│       ├── feature_importance.xlsx  ← Excel export
│       ├── top25_permutation_importance.png  ← Top 25 feature bar chart
│       └── group_importance.png     ← Category-level importance chart
```

---

## Reproducing

```bash
cd reproduction
python ml_classify.py          # Run all 20 ML experiments (30 sec)
python feature_importance.py   # Feature importance analysis (10 sec)
```

Requirements: `scikit-learn`, `xgboost`, `pandas`, `openpyxl`, `matplotlib`

Excel source must be at:
`E:\桌面文件\资料集-基于大模型与多智能体的COPD-PH资料集\copd-ph患者113例0331.xlsx`
