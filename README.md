# COPD-PH Pulmonary Vessel GCN

Graph-neural-network classification of pulmonary hypertension (PH) from chest-CT
pulmonary vessel trees, fused with commercial radiomics features. The repo
covers the full Sprint 2 pipeline: vessel-tree graph construction, hybrid GCN
training, 5-fold cross-validation, and interpretability visualizations.

## Task

Binary classification — **COPD-PH vs COPD-nonPH** — from:
- **Graph modality**: per-patient pulmonary vessel tree. Nodes = vessel
  segments, edges = bifurcation connectivity. 12D baseline / 16D enhanced node
  features (geometry + CT density + commercial fractal dim / A-V density).
- **Radiomics modality**: 45D commercial CT radiomics vector per patient.

Three training modes are compared:

| mode | inputs |
|---|---|
| `radiomics_only` | 45D radiomics → MLP head |
| `gcn_only` | vessel graph → GCN + global pool |
| `hybrid` | GCN embedding ⊕ radiomics → MLP head |

## Results (5-fold CV, n = 96 matched patients)

See `outputs/sprint2_metrics.xlsx` and `outputs/sprint2_radar*.png`.

| feat_set | mode | AUC | ACC | F1 | Spec | Sens | Prec |
|---|---|---|---|---|---|---|---|
| baseline (12D) | radiomics_only | 0.854 | 0.831 | 0.887 | 0.700 | 0.868 | 0.910 |
| baseline (12D) | gcn_only | **0.926** | 0.784 | 0.833 | 0.620 | 0.823 | 0.892 |
| baseline (12D) | hybrid | 0.900 | 0.726 | 0.806 | 0.540 | 0.798 | 0.855 |
| enhanced (16D) | radiomics_only | 0.881 | 0.741 | 0.781 | 0.590 | 0.791 | 0.879 |
| enhanced (16D) | gcn_only | 0.854 | 0.776 | 0.834 | 0.750 | 0.777 | 0.921 |
| enhanced (16D) | hybrid | 0.887 | **0.821** | 0.865 | **0.820** | 0.811 | **0.950** |

The 4 enhancement features (fractal dim, artery/vein density, volume-calibrated
diameter) substantially lift specificity of the hybrid model (0.54 → 0.82).

## Visualizations

### 6-metric radar (5-fold CV, 0.9 reference ring)

![Sprint 2 radar — split](outputs/sprint2_radar.png)

![Sprint 2 radar — all 6 configs overlaid](outputs/sprint2_radar_combined.png)

### Group statistics — PH vs non-PH

Distributions of the 4 commercial vessel features (boxplots + Mann-Whitney p).

![Group stats](outputs/viz_group_stats.png)

### 3D pulmonary vessel graphs

One PH + one non-PH case, nodes coloured by branching degree.

![Vessel tree samples](outputs/viz_vessel_tree_samples.png)

### Per-node saliency (enhanced-hybrid GCN, fold 1)

Same two cases, nodes coloured by input-gradient saliency `|∂ p(PH) / ∂ x_node|`.

![Saliency trees](outputs/viz_saliency_trees.png)

### Per-node saliency across all 5 folds

For each fold k, we re-train the enhanced-hybrid GCN on its training set and
compute saliency on the first PH + non-PH case from its validation set.

| | val AUC |
|---|---|
| Fold 1 | 0.96 |
| Fold 2 | 1.00 |
| Fold 3 | 1.00 |
| Fold 4 | 0.78 |
| Fold 5 | 0.78 |

![Saliency — fold 1](outputs/viz_saliency_fold1.png)
![Saliency — fold 2](outputs/viz_saliency_fold2.png)
![Saliency — fold 3](outputs/viz_saliency_fold3.png)
![Saliency — fold 4](outputs/viz_saliency_fold4.png)
![Saliency — fold 5](outputs/viz_saliency_fold5.png)

Combined 5×2 grid: [`outputs/viz_saliency_all_folds.png`](outputs/viz_saliency_all_folds.png)

### Why do folds 4 & 5 underperform? — mPAP audit

Gold standard: COPD-PH iff mPAP > 20 mmHg (resting). For each fold's val set we
joined the case_id back to the patient excel, recovered mPAP, and asked: are
folds 4/5 dominated by borderline (mPAP ≈ 20) cases?

![Per-fold mPAP audit](outputs/fold_mpap_audit.png)

| Fold | val AUC | n_val (matched) | n_PH | n_nonPH | nonPH max mPAP | PH min mPAP | gap | borderline ∈[18,22] |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.96 | 21 | 15 | 6 | 20 | 22 | **+2.0** | 5 |
| 2 | 1.00 | 17 | 11 | 6 | 18 | 26 | +8.0 | 1 |
| 3 | 1.00 | 21 | 17 | 4 | 15 | 23 | +8.0 | 0 |
| 4 | 0.78 | 24 | 19 | 5 | 20 | 26 | +6.0 | 2 |
| 5 | 0.78 | 22 | 17 | 5 | 19 | 24 | +5.0 | 3 |

**Findings (audit script: `analyze_folds_mpap.py`):**

1. **Labels are clean** — case_id prefix ↔ excel `PH` column agree on every
   matched case (0 mismatches across 105 val cases).
2. **Borderline density does not explain the gap.** Fold 1 has the *most*
   borderline overlap (gap = +2 mmHg, 5 borderline cases) yet still hits
   AUC = 0.96. Folds 4/5 have moderate borderline load (2–3 cases) and
   reasonably wide separation gaps (+5/+6 mmHg).
3. **Class imbalance is the dominant likely driver.** Folds 4 & 5 have only
   5 nonPH cases each (~20% negatives) — a single misclassified negative
   pushes specificity down by 0.20 and AUC noticeably. Folds 2/3 had cleaner
   negative-class examples (max mPAP ≤ 18, well below threshold).
4. **Individual variation, not threshold ambiguity.** In fold 5 the lowest
   PH case is at mPAP = 24 (clearly above 20) and the highest non-PH is at
   19 — clinically separable. The remaining errors are vessel-graph features
   for *clinically* clear-cut PH cases that nonetheless look atypical to the
   GCN. See `outputs/viz_saliency_fold5.png` for which nodes drive the
   prediction.

So the answer to "is it threshold-borderline cases?" is **no** — it's a
combination of (i) very few negatives in folds 4/5 inflating the variance of
specificity-driven AUC, and (ii) a few PH cases whose pulmonary vessel graph
geometry is atypical. Per-case mPAP is in `outputs/fold_mpap_audit.xlsx`.

## Repo layout

```
.
├── config.yaml                 # pipeline + training hyperparameters
├── hybrid_gcn.py               # HybridGCN(radiomics_only | gcn_only | hybrid)
├── gcn_models.py               # GCN backbones
├── graph_builder.py            # vessel tree → PyG graph
├── quantification.py           # vessel-segment geometric descriptors
├── skeleton.py                 # CT mask → centerline skeleton
├── enhance_features.py         # 12D → 16D node-feature enhancement
├── extract_radiomics.py        # commercial radiomics loader
├── run_hybrid.py               # Sprint 1: 5-fold CV, 3 modes
├── run_sprint2.py              # Sprint 2: baseline vs enhanced × 3 modes
├── run_demo.py, main.py        # smoke + full pipeline entry points
├── visualize.py                # group stats + 3D trees + saliency
├── make_report.py              # radar chart + xlsx summary
├── utils/                      # shared pipeline utilities
└── outputs/
    ├── sprint2_results.json    # raw fold metrics
    ├── sprint2_metrics.xlsx    # mean ± std summary
    ├── sprint2_radar*.png      # 6-metric radar charts
    └── viz_*.png               # interpretability plots
```

## Training pipeline

1. **Preprocess** CT masks → 3D vessel skeleton → PyG graph (`skeleton.py`,
   `graph_builder.py`, `quantification.py`) → cache as `cache/<case_id>.pkl`.
2. **Enhance** (optional) — augment 12D node features with commercial
   fractal / density / volume terms (`enhance_features.py`).
3. **Train** 5-fold CV across `{radiomics_only, gcn_only, hybrid}` × baseline/
   enhanced (`run_sprint2.py`).
4. **Visualize** (`visualize.py`): group stats, 3D trees, saliency.
5. **Report** (`make_report.py`): radar chart + xlsx summary.

Example (remote):

```bash
conda activate pulmonary_bv5_py39
CUDA_VISIBLE_DEVICES=0 python run_sprint2.py \
    --labels  "/.../labels.csv" \
    --splits  "/.../folds" \
    --output_dir outputs/sprint2_enhanced --epochs 200

python visualize.py \
    --labels "/.../labels.csv" \
    --splits "/.../folds" \
    --output_dir outputs/viz --epochs 120

python make_report.py
```

## Data & privacy

No patient-level data (CSV / cache pkls / xlsx) is committed. The user must
supply:

- `data/copd_ph_radiomics.csv` — 45-col commercial radiomics
- `cache/<case_id>.pkl` — per-patient PyG graphs
- `labels.csv` + `splits/folds/fold_{1..5}/{train,val}.txt`

## Dependencies

Python 3.9, PyTorch 2.x, `torch_geometric`, pandas, numpy, matplotlib,
scikit-learn, scipy, openpyxl. See `pulmonary_bv5_py39` conda env on the
training server for exact versions.
