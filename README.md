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

## Visualizations — `outputs/`

- `sprint2_radar.png`, `sprint2_radar_combined.png` — 6-metric radar (ACC, AUC,
  F1, Specificity, Sensitivity, Precision), with a 0.9 reference ring.
- `viz_group_stats.png` — PH vs non-PH distributions of the 4 commercial
  vessel features (boxplots + Mann-Whitney p).
- `viz_vessel_tree_samples.png` — 3D vessel graphs of one PH + one non-PH case,
  nodes coloured by branching degree.
- `viz_saliency_trees.png` — same two cases, nodes coloured by input-gradient
  saliency from a fold-1 enhanced-hybrid GCN.

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
