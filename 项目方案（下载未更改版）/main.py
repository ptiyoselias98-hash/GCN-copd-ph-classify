#!/usr/bin/env python3
"""
Pulmonary Microvascular GCN — Main Entry Point

Usage:
  # Process data and train model:
  python main.py --mode train --data_dir ./data/raw --labels ./data/labels.csv

  # Run with synthetic data for testing:
  python main.py --mode demo

  # Extract features only (no training):
  python main.py --mode features --data_dir ./data/raw --labels ./data/labels.csv

  # Cross-validation:
  python main.py --mode cv --data_dir ./data/raw --labels ./data/labels.csv
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import pickle
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.skeleton import VesselSkeleton, compute_strahler_order
from utils.graph_builder import VascularGraphBuilder, normalize_graph_features
from utils.quantification import VascularQuantifier, ParenchymaQuantifier, AirwayQuantifier
from models.gcn_models import build_model


# ============================================================
# Synthetic data generator (for testing without real data)
# ============================================================

def generate_synthetic_vascular_tree(
    label: int = 0,
    noise: float = 0.1
) -> dict:
    """
    Generate a synthetic vascular tree graph for testing.

    COPD-PH (label=1) trees have:
      - Fewer branches (pruning)
      - Smaller distal diameters
      - Higher tortuosity
      - Lower BV5 ratio
    """
    from utils.graph_builder import VascularGraphBuilder

    rng = np.random.RandomState()

    # Base tree parameters
    if label == 0:  # COPD without PH
        n_generations = np.random.randint(6, 9)
        base_diameter = 8.0 + rng.randn() * 0.5
        pruning_prob = 0.1
        diameter_decay = 0.72
        tortuosity_base = 1.05
    else:  # COPD-PH
        n_generations = np.random.randint(4, 7)
        base_diameter = 7.0 + rng.randn() * 0.5
        pruning_prob = 0.35  # More pruning
        diameter_decay = 0.65  # Faster narrowing
        tortuosity_base = 1.15  # More tortuous

    branches = []
    branch_features = []

    # Generate tree by recursive branching
    def _grow(parent_pos, diameter, generation, direction):
        if generation > n_generations or diameter < 0.3:
            return

        # Random pruning
        if generation > 2 and rng.random() < pruning_prob:
            return

        length = diameter * (3 + rng.randn() * 0.5)
        tortuosity = tortuosity_base + rng.randn() * 0.05

        end_pos = parent_pos + direction * length + rng.randn(3) * noise

        # Create intermediate path
        n_pts = max(5, int(length))
        path = np.linspace(parent_pos, end_pos, n_pts)
        path += rng.randn(*path.shape) * noise * 0.3

        branches.append({
            'start': tuple(parent_pos.astype(int)),
            'end': tuple(end_pos.astype(int)),
            'path': path.astype(int),
            'length_voxels': n_pts
        })

        branch_features.append({
            'diameter': diameter,
            'length': length,
            'tortuosity': tortuosity,
            'mean_ct_density': -700 + rng.randn() * 50,
            'orientation': (direction / np.linalg.norm(direction)).tolist(),
            'centroid': ((parent_pos + end_pos) / 2).tolist(),
            'num_voxels': n_pts
        })

        # Branch into two children
        child_d = diameter * diameter_decay * (1 + rng.randn() * 0.05)
        angle = np.pi / 6 + rng.randn() * 0.1

        # Left child
        rot = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1]
        ])
        left_dir = rot @ direction
        _grow(end_pos, child_d, generation + 1, left_dir)

        # Right child
        rot_r = np.array([
            [np.cos(-angle), -np.sin(-angle), 0],
            [np.sin(-angle),  np.cos(-angle), 0],
            [0, 0, 1]
        ])
        right_dir = rot_r @ direction
        _grow(end_pos, child_d * (1 + rng.randn() * 0.03),
              generation + 1, right_dir)

    # Start from root
    root_pos = np.array([128.0, 128.0, 128.0])
    _grow(root_pos, base_diameter, 0, np.array([0, -1, 0]))

    # Build graph
    builder = VascularGraphBuilder(
        spatial_edge_threshold=15.0,
        add_spatial_edges=True
    )

    graph = builder.build_graph(branches, branch_features, label=label)

    # Synthetic quantitative features
    features = {
        'vascular': {
            'bv5': 0.012 - label * 0.004 + rng.randn() * 0.001,
            'bv10': 0.025 - label * 0.006 + rng.randn() * 0.002,
            'pruning_index': 0.2 + label * 0.25 + rng.randn() * 0.05,
            'mean_diameter': base_diameter * 0.6,
            'num_total_branches': len(branches),
            'artery_vein_ratio': 0.9 + rng.randn() * 0.1,
        },
        'parenchyma': {
            'laa_pct': 5.0 + label * 8.0 + rng.randn() * 2.0,
            'mean_lung_density': -850 + label * 20 + rng.randn() * 10,
        },
        'airway': {
            'wall_area_pct': 55 + label * 5 + rng.randn() * 3,
            'wall_thickness_ratio': 0.4 + label * 0.05 + rng.randn() * 0.02,
            'airway_count': 80 - label * 15 + int(rng.randn() * 5),
        }
    }

    return {
        'graph': graph,
        'features': features,
        'label': label,
        'patient_id': f"synthetic_{label}_{rng.randint(10000)}"
    }


def generate_synthetic_dataset(
    n_copd: int = 27,
    n_copd_ph: int = 167
) -> list:
    """Generate full synthetic dataset for testing."""
    print(f"Generating synthetic dataset: {n_copd} COPD + {n_copd_ph} COPD-PH...")

    dataset = []
    for i in range(n_copd):
        data = generate_synthetic_vascular_tree(label=0)
        data['patient_id'] = f"COPD_{i:03d}"
        dataset.append(data)

    for i in range(n_copd_ph):
        data = generate_synthetic_vascular_tree(label=1)
        data['patient_id'] = f"COPD_PH_{i:03d}"
        dataset.append(data)

    print(f"Generated {len(dataset)} samples")
    return dataset


# ============================================================
# Feature analysis and visualization
# ============================================================

def analyze_features(dataset: list, output_dir: str = './outputs'):
    """Analyze and compare features between groups."""
    os.makedirs(output_dir, exist_ok=True)

    copd = [d for d in dataset if d['label'] == 0]
    copd_ph = [d for d in dataset if d['label'] == 1]

    print(f"\n{'='*60}")
    print(f"Feature Analysis: COPD (n={len(copd)}) vs COPD-PH (n={len(copd_ph)})")
    print(f"{'='*60}")

    categories = ['vascular', 'parenchyma', 'airway']
    results = {}

    for cat in categories:
        print(f"\n--- {cat.upper()} FEATURES ---")
        cat_results = {}

        # Collect all feature keys
        all_keys = set()
        for d in dataset:
            if cat in d['features']:
                all_keys.update(d['features'][cat].keys())

        for key in sorted(all_keys):
            copd_vals = [d['features'][cat].get(key, 0) for d in copd
                         if isinstance(d['features'].get(cat, {}).get(key, 0),
                                       (int, float))]
            ph_vals = [d['features'][cat].get(key, 0) for d in copd_ph
                       if isinstance(d['features'].get(cat, {}).get(key, 0),
                                     (int, float))]

            if copd_vals and ph_vals:
                copd_mean = np.mean(copd_vals)
                ph_mean = np.mean(ph_vals)
                copd_std = np.std(copd_vals)
                ph_std = np.std(ph_vals)

                # Effect size (Cohen's d)
                pooled_std = np.sqrt((copd_std**2 + ph_std**2) / 2)
                cohen_d = abs(copd_mean - ph_mean) / max(pooled_std, 1e-6)

                print(f"  {key:30s}: COPD={copd_mean:.4f}±{copd_std:.4f} | "
                      f"PH={ph_mean:.4f}±{ph_std:.4f} | d={cohen_d:.2f}")

                cat_results[key] = {
                    'copd_mean': copd_mean, 'copd_std': copd_std,
                    'ph_mean': ph_mean, 'ph_std': ph_std,
                    'cohen_d': cohen_d
                }

        results[cat] = cat_results

    # Save results
    with open(os.path.join(output_dir, 'feature_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/feature_analysis.json")
    return results


def save_embeddings_for_viz(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_dir: str = './outputs'
):
    """Save embeddings for external t-SNE/UMAP visualization."""
    os.makedirs(output_dir, exist_ok=True)

    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=min(30, len(labels) - 1),
                     random_state=42)
        coords_2d = tsne.fit_transform(embeddings)

        np.savez(
            os.path.join(output_dir, 'embeddings.npz'),
            embeddings=embeddings,
            tsne_2d=coords_2d,
            labels=labels
        )
        print(f"Embeddings saved to {output_dir}/embeddings.npz")

        # Print cluster quality
        from sklearn.metrics import silhouette_score
        sil = silhouette_score(coords_2d, labels)
        print(f"t-SNE silhouette score: {sil:.3f}")

    except ImportError:
        np.savez(
            os.path.join(output_dir, 'embeddings.npz'),
            embeddings=embeddings,
            labels=labels
        )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Pulmonary Microvascular GCN Pipeline'
    )
    parser.add_argument('--mode', type=str, default='demo',
                        choices=['demo', 'train', 'features', 'cv'],
                        help='Running mode')
    parser.add_argument('--data_dir', type=str, default='./data/raw',
                        help='Root directory with patient data')
    parser.add_argument('--labels', type=str, default='./data/labels.csv',
                        help='CSV file with patient_id,label columns')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: cpu, cuda, auto')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--model', type=str, default='GraphSAGE',
                        choices=['GCN', 'GraphSAGE', 'GAT'])
    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")

    # Config
    config = {
        'skeleton': {'min_branch_length': 3, 'resample_spacing': [1, 1, 1]},
        'graph': {
            'node_features': 12,
            'spatial_edge_threshold': 15.0,
            'add_spatial_edges': True,
            'use_directed': False
        },
        'model': {
            'type': args.model,
            'hidden_channels': 64,
            'num_layers': 3,
            'dropout': 0.3,
            'pooling': 'mean',
            'heads': 4
        },
        'training': {
            'epochs': args.epochs,
            'lr': 0.001,
            'weight_decay': 0.0005,
            'batch_size': args.batch_size,
            'patience': 30,
            'seed': 42
        }
    }

    os.makedirs(args.output_dir, exist_ok=True)

    # ===========================
    # MODE: Demo (synthetic data)
    # ===========================
    if args.mode == 'demo':
        print("\n" + "="*60)
        print("DEMO MODE — Using synthetic vascular tree data")
        print("="*60 + "\n")

        dataset = generate_synthetic_dataset(n_copd=27, n_copd_ph=50)

        # Feature analysis
        analyze_features(dataset, args.output_dir)

        # Prepare graphs for training
        graphs = [d['graph'] for d in dataset]
        graphs = normalize_graph_features(graphs)

        # Update graph references
        for d, g in zip(dataset, graphs):
            d['graph'] = g

        # Split
        from utils.training import split_dataset, Trainer
        train_data, val_data, test_data = split_dataset(dataset, seed=42)

        train_graphs = [d['graph'] for d in train_data]
        val_graphs = [d['graph'] for d in val_data]
        test_graphs = [d['graph'] for d in test_data]

        try:
            from torch_geometric.loader import DataLoader

            train_loader = DataLoader(train_graphs, batch_size=args.batch_size,
                                      shuffle=True)
            val_loader = DataLoader(val_graphs, batch_size=args.batch_size)
            test_loader = DataLoader(test_graphs, batch_size=args.batch_size)

            # Build model
            config['model']['in_channels'] = train_graphs[0].x.shape[1]
            config['model']['out_channels'] = 2
            model = build_model(config['model'])
            print(f"\nModel: {config['model']['type']}")
            print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

            # Train
            trainer = Trainer(model, config['training'], device)
            history = trainer.train(train_loader, val_loader,
                                    epochs=args.epochs,
                                    save_dir=args.output_dir)

            # Test evaluation
            test_metrics = trainer.evaluate(test_loader)
            print(f"\n{'='*60}")
            print("TEST RESULTS")
            print(f"{'='*60}")
            for k, v in test_metrics.items():
                print(f"  {k:15s}: {v:.4f}")

            # Save test results
            with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
                json.dump(test_metrics, f, indent=2)

            # Extract and save embeddings
            all_loader = DataLoader(graphs, batch_size=args.batch_size)
            embeddings, labels, _ = trainer.extract_embeddings(all_loader)
            save_embeddings_for_viz(embeddings, labels, args.output_dir)

            # Save training history
            with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
                json.dump(history, f, indent=2)

            print(f"\nAll outputs saved to {args.output_dir}/")

        except ImportError:
            print("\n[WARNING] torch_geometric not installed.")
            print("Install: pip install torch-geometric")
            print("Feature analysis completed, but model training skipped.")

    # ===========================
    # MODE: Train (real data)
    # ===========================
    elif args.mode == 'train':
        import csv
        from utils.pipeline import process_dataset

        # Load labels
        labels = {}
        with open(args.labels, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels[row['patient_id']] = int(row['label'])

        print(f"Loaded {len(labels)} patient labels")

        # Process dataset
        dataset = process_dataset(
            args.data_dir, labels, config,
            cache_dir=os.path.join(args.data_dir, '..', 'cache')
        )

        if len(dataset) < 10:
            print("Too few patients processed. Check data paths.")
            return

        # Continue with training (same as demo)
        analyze_features(dataset, args.output_dir)
        graphs = normalize_graph_features([d['graph'] for d in dataset])
        for d, g in zip(dataset, graphs):
            d['graph'] = g

        from utils.training import split_dataset, Trainer
        from torch_geometric.loader import DataLoader

        train_data, val_data, test_data = split_dataset(dataset, seed=42)
        train_loader = DataLoader([d['graph'] for d in train_data],
                                  batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader([d['graph'] for d in val_data],
                                batch_size=args.batch_size)
        test_loader = DataLoader([d['graph'] for d in test_data],
                                 batch_size=args.batch_size)

        config['model']['in_channels'] = graphs[0].x.shape[1]
        config['model']['out_channels'] = 2
        model = build_model(config['model'])

        trainer = Trainer(model, config['training'], device)
        history = trainer.train(train_loader, val_loader,
                                epochs=args.epochs, save_dir=args.output_dir)

        test_metrics = trainer.evaluate(test_loader)
        print(f"\nTest: {test_metrics}")

        with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
            json.dump(test_metrics, f, indent=2)

    # ===========================
    # MODE: Features only
    # ===========================
    elif args.mode == 'features':
        import csv
        from utils.pipeline import process_dataset

        labels = {}
        with open(args.labels, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels[row['patient_id']] = int(row['label'])

        dataset = process_dataset(
            args.data_dir, labels, config,
            cache_dir=os.path.join(args.data_dir, '..', 'cache')
        )

        analyze_features(dataset, args.output_dir)

    # ===========================
    # MODE: Cross-validation
    # ===========================
    elif args.mode == 'cv':
        import csv
        from utils.pipeline import process_dataset
        from utils.training import cross_validate

        labels = {}
        with open(args.labels, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels[row['patient_id']] = int(row['label'])

        dataset = process_dataset(
            args.data_dir, labels, config,
            cache_dir=os.path.join(args.data_dir, '..', 'cache')
        )

        graphs = normalize_graph_features([d['graph'] for d in dataset])
        for d, g in zip(dataset, graphs):
            d['graph'] = g

        config['model']['in_channels'] = graphs[0].x.shape[1]
        config['model']['out_channels'] = 2

        cv_results = cross_validate(dataset, config, n_folds=5, device=device)

        with open(os.path.join(args.output_dir, 'cv_results.json'), 'w') as f:
            json.dump(cv_results, f, indent=2)


if __name__ == '__main__':
    main()
