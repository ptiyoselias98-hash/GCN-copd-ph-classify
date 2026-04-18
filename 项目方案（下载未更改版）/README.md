# Pulmonary Microvascular GCN

Graph convolution framework for analyzing pulmonary vascular remodeling patterns in COPD-PH progression.

## Overview

This project constructs graph representations of pulmonary vascular trees from CT scans and applies graph neural networks to discover evolution patterns between COPD (without pulmonary hypertension) and COPD-PH.

### Pipeline Stages

1. **Input**: CT volumes + segmentation masks (artery, vein, lung, airway) in NIfTI format
2. **Skeleton extraction**: 3D thinning → centerline → bifurcation/terminal point detection
3. **Graph construction**: Nodes at bifurcations, edges along vessel segments, features from CT + morphology
4. **Graph convolution**: GCN/GraphSAGE/GAT for graph-level and node-level representation learning
5. **Evolution analysis**: Classification (COPD vs COPD-PH), feature importance, embedding visualization

### Node Features (12 dimensions)

| Feature | Source | Clinical Meaning |
|---------|--------|-----------------|
| Diameter | Distance transform | Vessel caliber, narrowing in PH |
| Length | Centerline path | Segment extent |
| Tortuosity | Path/Euclidean | Vessel distortion, increases in PH |
| CT density | HU values | Blood volume proxy |
| Generation | Tree depth | Proximal vs distal location |
| Orientation | Direction vector (3D) | Spatial arrangement |
| Centroid | Spatial coords (3D) | Regional localization |
| Strahler order | Tree hierarchy | Branching importance |
| Degree | Graph connectivity | Junction complexity |

### Quantitative Outputs

- **Vascular**: BV5, BV10, pruning index, artery-vein ratio, diameter distribution
- **Parenchyma**: LAA%, mean lung density, density histogram
- **Airway**: Wall area %, wall thickness ratio, airway count

## Quick Start

### Demo (no real data needed)

```bash
cd pulmonary_gcn
python run_demo.py
```

This generates synthetic vascular trees with realistic COPD vs COPD-PH phenotypes and runs the full analysis pipeline.

### With Real Data

#### 1. Organize data

```
data/raw/
├── patient_001/
│   ├── ct.nii.gz
│   ├── artery.nii.gz
│   ├── vein.nii.gz
│   ├── lung.nii.gz
│   └── airway.nii.gz
├── patient_002/
│   └── ...
```

#### 2. Create labels file

```csv
patient_id,label
patient_001,0
patient_002,1
```

Where `0` = COPD (no PH), `1` = COPD-PH.

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

#### 4. Run

```bash
# Full training pipeline
python main.py --mode train --data_dir ./data/raw --labels ./data/labels.csv

# Feature extraction only
python main.py --mode features --data_dir ./data/raw --labels ./data/labels.csv

# 5-fold cross-validation
python main.py --mode cv --data_dir ./data/raw --labels ./data/labels.csv

# Choose model architecture
python main.py --mode train --model GAT --epochs 300
```

## Project Structure

```
pulmonary_gcn/
├── main.py                  # Main entry point (requires PyTorch + PyG)
├── run_demo.py              # Standalone demo (NumPy/sklearn only)
├── requirements.txt
├── configs/
│   └── config.yaml          # Full configuration
├── models/
│   └── gcn_models.py        # GCN, GraphSAGE, GAT architectures
├── utils/
│   ├── skeleton.py          # 3D skeletonization + topology parsing
│   ├── graph_builder.py     # Graph construction from skeleton
│   ├── quantification.py    # BV5, LAA%, WA% feature extraction
│   ├── pipeline.py          # NIfTI loading + full processing pipeline
│   └── training.py          # Training loop, evaluation, cross-validation
└── outputs/                 # Results, embeddings, saved models
```

## Key Design Decisions

### Why Graph Convolution?

Traditional vascular analysis uses global summary statistics (total vessel volume, mean diameter). Graph convolution preserves the **topological structure** — which branches are connected to which, where pruning occurs relative to bifurcations, how local features propagate through the tree. This captures:

- **Spatial pruning patterns**: GCN node embeddings encode each vessel's context within the tree, revealing whether pruning is uniform or concentrated in specific subtrees
- **Cross-scale interactions**: Message passing aggregates features across generations, linking proximal remodeling to distal loss
- **Patient-level fingerprints**: Graph pooling creates a single embedding per patient that encodes both topology and morphology

### Model Choice

- **GCN**: Simplest, good baseline. Fixed aggregation weights.
- **GraphSAGE** (recommended): Inductive — learns to aggregate neighbor features, generalizes better across patients with different tree sizes.
- **GAT**: Learnable attention weights per edge — can discover which neighboring vessels matter most. Higher capacity but needs more data.

### Handling Class Imbalance (27 vs 167)

- Weighted cross-entropy loss (inverse class frequency)
- Stratified train/val/test splits
- `class_weight='balanced'` for sklearn baselines
- AUC as primary metric (robust to imbalance)

## Expected Results with Real Data

With synthetic data, separation is perfect (AUC=1.0) because the generative model encodes strong phenotypic differences. With real CT data, expect:

- **AUC 0.75–0.90** for COPD vs COPD-PH classification
- **Key discriminative features**: BV5 ratio, branch count, tortuosity, LAA%
- **Evolution patterns**: progressive distal vessel pruning, increasing tortuosity, parenchymal destruction

## Citation

If you use this framework, please cite the relevant methodological papers:

- Kipf & Welling (2017) - Semi-supervised classification with GCN
- Hamilton et al. (2017) - Inductive representation learning (GraphSAGE)
- Veličković et al. (2018) - Graph attention networks
