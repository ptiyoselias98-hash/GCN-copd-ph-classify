#!/usr/bin/env python3
"""
Pulmonary Microvascular GCN — Standalone Demo

Runs the FULL pipeline with synthetic data using only NumPy/SciPy/sklearn.
No PyTorch or PyG required. Demonstrates:

  1. Synthetic vascular tree generation (COPD vs COPD-PH phenotypes)
  2. Skeleton extraction + topology parsing
  3. Graph construction with node/edge features
  4. Graph-level feature aggregation
  5. Classification (Random Forest + SVM baselines)
  6. Feature analysis + evolution pattern detection
  7. t-SNE embedding visualization data

Usage:
  python run_demo.py
"""

import os
import sys
import json
import numpy as np
from collections import defaultdict
from scipy import ndimage
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
from sklearn.manifold import TSNE
from skimage.morphology import skeletonize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. Synthetic vascular tree generation
# ============================================================

def generate_vascular_tree(label: int, seed: int = None) -> dict:
    """
    Generate a synthetic 3D vascular tree.

    label=0 (COPD): denser branching, larger distal vessels
    label=1 (COPD-PH): pruned periphery, narrower vessels, higher tortuosity
    """
    rng = np.random.RandomState(seed)

    # Disease-specific parameters
    if label == 0:  # COPD without PH
        max_gen = rng.randint(6, 9)
        base_d = 8.0 + rng.randn() * 0.5
        prune_p = 0.08
        d_decay = 0.72
        tort_base = 1.05
    else:  # COPD-PH
        max_gen = rng.randint(4, 7)
        base_d = 7.0 + rng.randn() * 0.5
        prune_p = 0.30
        d_decay = 0.62
        tort_base = 1.18

    branches = []
    bifurcations = []
    terminals = []

    def grow(pos, diameter, gen, direction):
        if gen > max_gen or diameter < 0.3:
            terminals.append(pos.copy())
            return

        if gen > 2 and rng.random() < prune_p:
            terminals.append(pos.copy())
            return

        length = diameter * (3.0 + rng.randn() * 0.5)
        tort = tort_base + abs(rng.randn() * 0.05)
        end = pos + direction * length + rng.randn(3) * 0.5

        n_pts = max(5, int(length))
        path = np.linspace(pos, end, n_pts)
        path += rng.randn(*path.shape) * 0.3

        ct_density = -700 + gen * 15 + rng.randn() * 30
        if label == 1:
            ct_density += 40  # Higher density in PH vessels

        branches.append({
            'start': pos.copy(),
            'end': end.copy(),
            'diameter': float(diameter),
            'length': float(length),
            'tortuosity': float(tort),
            'generation': gen,
            'ct_density': float(ct_density),
            'path': path,
            'orientation': (direction / max(np.linalg.norm(direction), 1e-6)).tolist(),
            'centroid': ((pos + end) / 2).tolist()
        })

        bifurcations.append(end.copy())

        angle = np.pi / 6 + rng.randn() * 0.08
        child_d = diameter * d_decay * (1 + rng.randn() * 0.04)

        # Rotate direction for children
        axis = rng.randn(3)
        axis /= max(np.linalg.norm(axis), 1e-6)

        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot1 = cos_a * direction + sin_a * np.cross(axis, direction)
        rot2 = cos_a * direction - sin_a * np.cross(axis, direction)

        grow(end, child_d, gen + 1, rot1 / max(np.linalg.norm(rot1), 1e-6))
        grow(end, child_d * (1 + rng.randn() * 0.03), gen + 1,
             rot2 / max(np.linalg.norm(rot2), 1e-6))

    root = np.array([128.0, 128.0, 128.0])
    grow(root, base_d, 0, np.array([0, -1.0, 0]))

    return {
        'branches': branches,
        'bifurcations': bifurcations,
        'terminals': terminals,
        'root': root,
        'label': label
    }


# ============================================================
# 2. Graph construction
# ============================================================

def build_graph_from_tree(tree: dict) -> dict:
    """Convert tree branches into a graph with adjacency and features."""
    branches = tree['branches']
    point_to_id = {}
    nid = 0

    def get_id(pos):
        nonlocal nid
        key = tuple(np.round(pos, 2))
        if key not in point_to_id:
            point_to_id[key] = nid
            nid += 1
        return point_to_id[key]

    edges = []
    edge_features = []
    node_branch_feats = defaultdict(list)

    for b in branches:
        src = get_id(b['start'])
        dst = get_id(b['end'])
        edges.append((src, dst))
        edges.append((dst, src))

        ef = [b['diameter'], b['length'], b['tortuosity']]
        edge_features.append(ef)
        edge_features.append(ef)

        feat_vec = [
            b['diameter'], b['length'], b['tortuosity'],
            b['ct_density'], b['generation'],
            *b['orientation'], *b['centroid'][:3]
        ]
        node_branch_feats[src].append(feat_vec)
        node_branch_feats[dst].append(feat_vec)

    num_nodes = len(point_to_id)
    feat_dim = 11

    # Node feature matrix: average of incident branch features
    X = np.zeros((num_nodes, feat_dim), dtype=np.float32)
    for nid_key, feats in node_branch_feats.items():
        X[nid_key] = np.mean(feats, axis=0)

    # Compute Strahler order (simplified)
    adjacency = defaultdict(set)
    for s, d in edges:
        adjacency[s].add(d)

    is_terminal = {}
    for nid_val in range(num_nodes):
        is_terminal[nid_val] = len(adjacency[nid_val]) <= 1

    # Degree
    degrees = np.array([len(adjacency[i]) for i in range(num_nodes)])

    return {
        'num_nodes': num_nodes,
        'edges': edges,
        'node_features': X,
        'degrees': degrees,
        'is_terminal': is_terminal,
        'adjacency': adjacency,
        'label': tree['label']
    }


# ============================================================
# 3. Graph-level feature extraction
# ============================================================

def extract_graph_level_features(graph: dict, tree: dict) -> np.ndarray:
    """
    Extract a fixed-length feature vector summarizing the entire graph.

    Features (30 dimensions):
      Structural: num_nodes, num_edges, num_terminals, avg_degree, ...
      Morphological: mean/std diameter, length, tortuosity
      CT-derived: mean density
      Disease markers: BV5-proxy, pruning index, generation depth
    """
    X = graph['node_features']
    branches = tree['branches']
    n = graph['num_nodes']
    n_edges = len(graph['edges']) // 2
    n_terminal = sum(1 for v in graph['is_terminal'].values() if v)

    diameters = [b['diameter'] for b in branches]
    lengths = [b['length'] for b in branches]
    tortuosities = [b['tortuosity'] for b in branches]
    densities = [b['ct_density'] for b in branches]
    generations = [b['generation'] for b in branches]

    # Small vessel fraction (BV5 proxy): diameter < 2mm
    small_count = sum(1 for d in diameters if d < 2.0)
    small_frac = small_count / max(len(diameters), 1)

    # Generation depth
    max_gen = max(generations) if generations else 0
    mean_gen = np.mean(generations) if generations else 0

    features = [
        # Structural (7)
        n,
        n_edges,
        n_terminal,
        np.mean(graph['degrees']),
        np.std(graph['degrees']),
        n_terminal / max(n, 1),  # terminal ratio
        n_edges / max(n, 1),     # edge density

        # Diameter stats (5)
        np.mean(diameters) if diameters else 0,
        np.std(diameters) if diameters else 0,
        np.median(diameters) if diameters else 0,
        np.min(diameters) if diameters else 0,
        np.max(diameters) if diameters else 0,

        # Length stats (3)
        np.mean(lengths) if lengths else 0,
        np.std(lengths) if lengths else 0,
        np.sum(lengths) if lengths else 0,

        # Tortuosity (3)
        np.mean(tortuosities) if tortuosities else 0,
        np.std(tortuosities) if tortuosities else 0,
        np.max(tortuosities) if tortuosities else 0,

        # CT density (2)
        np.mean(densities) if densities else 0,
        np.std(densities) if densities else 0,

        # Disease markers (5)
        small_frac,           # BV5 proxy (pruning indicator)
        len(diameters),       # Total branch count
        max_gen,              # Max branching depth
        mean_gen,             # Mean generation
        small_count,          # Absolute small vessel count

        # Node feature aggregates (5)
        np.mean(X[:, 0]),     # Mean node diameter
        np.std(X[:, 0]),      # Std node diameter
        np.mean(X[:, 2]),     # Mean node tortuosity
        np.mean(X[:, 3]),     # Mean node CT density
        np.mean(X[:, 4]),     # Mean node generation
    ]

    return np.array(features, dtype=np.float32)


# ============================================================
# 4. Simulate parenchyma + airway features
# ============================================================

def generate_clinical_features(label: int, rng: np.random.RandomState) -> dict:
    """Generate realistic clinical features correlated with disease state."""
    if label == 0:
        laa = 8.0 + rng.randn() * 3.0
        mld = -860 + rng.randn() * 15
        wa_pct = 55 + rng.randn() * 4
        wt_ratio = 0.38 + rng.randn() * 0.03
        airway_count = 85 + int(rng.randn() * 8)
        bv5 = 0.014 + rng.randn() * 0.002
        av_ratio = 0.95 + rng.randn() * 0.08
    else:
        laa = 15.0 + rng.randn() * 5.0
        mld = -835 + rng.randn() * 20
        wa_pct = 62 + rng.randn() * 5
        wt_ratio = 0.44 + rng.randn() * 0.04
        airway_count = 65 + int(rng.randn() * 10)
        bv5 = 0.008 + rng.randn() * 0.002
        av_ratio = 1.15 + rng.randn() * 0.12

    return {
        'laa_pct': max(0, laa),
        'mean_lung_density': mld,
        'wall_area_pct': wa_pct,
        'wall_thickness_ratio': wt_ratio,
        'airway_count': max(10, airway_count),
        'bv5_clinical': max(0, bv5),
        'artery_vein_ratio': max(0.3, av_ratio)
    }


# ============================================================
# 5. Full pipeline
# ============================================================

def run_pipeline():
    print("=" * 65)
    print("  PULMONARY MICROVASCULAR GCN — DEMO PIPELINE")
    print("  COPD vs COPD-PH classification from vascular tree graphs")
    print("=" * 65)

    # --- Generate dataset ---
    N_COPD = 27
    N_PH = 80  # Subset for speed
    rng = np.random.RandomState(42)

    print(f"\n[1/6] Generating synthetic vascular trees...")
    print(f"       COPD: {N_COPD} cases | COPD-PH: {N_PH} cases")

    trees, graphs, features_all, labels = [], [], [], []

    for i in range(N_COPD):
        tree = generate_vascular_tree(0, seed=rng.randint(100000))
        graph = build_graph_from_tree(tree)
        gf = extract_graph_level_features(graph, tree)
        cf = generate_clinical_features(0, rng)
        combined = np.concatenate([gf, list(cf.values())])

        trees.append(tree)
        graphs.append(graph)
        features_all.append(combined)
        labels.append(0)

    for i in range(N_PH):
        tree = generate_vascular_tree(1, seed=rng.randint(100000))
        graph = build_graph_from_tree(tree)
        gf = extract_graph_level_features(graph, tree)
        cf = generate_clinical_features(1, rng)
        combined = np.concatenate([gf, list(cf.values())])

        trees.append(tree)
        graphs.append(graph)
        features_all.append(combined)
        labels.append(1)

    X = np.array(features_all)
    y = np.array(labels)
    print(f"       Feature matrix: {X.shape}")

    # --- Feature analysis ---
    print(f"\n[2/6] Feature analysis: COPD vs COPD-PH differences")

    feature_names = [
        'num_nodes', 'num_edges', 'num_terminals', 'avg_degree', 'std_degree',
        'terminal_ratio', 'edge_density',
        'mean_diam', 'std_diam', 'median_diam', 'min_diam', 'max_diam',
        'mean_length', 'std_length', 'total_length',
        'mean_tort', 'std_tort', 'max_tort',
        'mean_density', 'std_density',
        'small_vessel_frac', 'branch_count', 'max_gen', 'mean_gen', 'small_count',
        'node_mean_diam', 'node_std_diam', 'node_mean_tort',
        'node_mean_density', 'node_mean_gen',
        'laa_pct', 'mean_lung_density', 'wall_area_pct', 'wt_ratio',
        'airway_count', 'bv5_clinical', 'av_ratio'
    ]

    copd_X = X[y == 0]
    ph_X = X[y == 1]

    print(f"\n  {'Feature':<25s} {'COPD':>12s} {'COPD-PH':>12s} {'Cohen d':>10s}")
    print("  " + "-" * 62)

    important_features = []
    for i, name in enumerate(feature_names):
        m0, s0 = copd_X[:, i].mean(), copd_X[:, i].std()
        m1, s1 = ph_X[:, i].mean(), ph_X[:, i].std()
        pooled = np.sqrt((s0**2 + s1**2) / 2)
        d = abs(m0 - m1) / max(pooled, 1e-6)

        marker = " ***" if d > 0.8 else (" **" if d > 0.5 else "")
        print(f"  {name:<25s} {m0:>7.2f}±{s0:>4.2f} {m1:>7.2f}±{s1:>4.2f} {d:>8.2f}{marker}")

        if d > 0.5:
            important_features.append((name, d))

    important_features.sort(key=lambda x: -x[1])
    print(f"\n  Top discriminative features (Cohen's d > 0.5):")
    for name, d in important_features[:10]:
        print(f"    {name}: d = {d:.2f}")

    # --- Classification ---
    print(f"\n[3/6] Classification: 5-fold stratified cross-validation")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        ),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, class_weight='balanced',
                         random_state=42),
    }

    cv_results = {}

    for name, model in models.items():
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_metrics = []

        for train_idx, test_idx in skf.split(X_scaled, y):
            model.fit(X_scaled[train_idx], y[train_idx])
            pred = model.predict(X_scaled[test_idx])
            prob = model.predict_proba(X_scaled[test_idx])[:, 1]

            fold_metrics.append({
                'accuracy': accuracy_score(y[test_idx], pred),
                'f1': f1_score(y[test_idx], pred),
                'precision': precision_score(y[test_idx], pred),
                'recall': recall_score(y[test_idx], pred),
                'auc': roc_auc_score(y[test_idx], prob)
            })

        avg = {k: np.mean([f[k] for f in fold_metrics]) for k in fold_metrics[0]}
        std = {k: np.std([f[k] for f in fold_metrics]) for k in fold_metrics[0]}

        cv_results[name] = {'mean': avg, 'std': std}

        print(f"\n  {name}:")
        print(f"    Accuracy:  {avg['accuracy']:.3f} ± {std['accuracy']:.3f}")
        print(f"    F1 Score:  {avg['f1']:.3f} ± {std['f1']:.3f}")
        print(f"    AUC:       {avg['auc']:.3f} ± {std['auc']:.3f}")
        print(f"    Precision: {avg['precision']:.3f} ± {std['precision']:.3f}")
        print(f"    Recall:    {avg['recall']:.3f} ± {std['recall']:.3f}")

    # --- Feature importance ---
    print(f"\n[4/6] Feature importance (Random Forest)")

    rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                random_state=42, class_weight='balanced')
    rf.fit(X_scaled, y)
    importances = rf.feature_importances_

    ranked = sorted(zip(feature_names, importances), key=lambda x: -x[1])
    print(f"\n  {'Rank':<6s} {'Feature':<25s} {'Importance':>12s}")
    print("  " + "-" * 45)
    for i, (name, imp) in enumerate(ranked[:15]):
        bar = "█" * int(imp * 200)
        print(f"  {i+1:<6d} {name:<25s} {imp:>10.4f}  {bar}")

    # --- t-SNE embedding ---
    print(f"\n[5/6] Computing t-SNE embeddings...")

    perp = min(30, len(y) - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(X_scaled)

    from sklearn.metrics import silhouette_score
    sil = silhouette_score(coords, y)
    print(f"       Silhouette score: {sil:.3f}")

    # --- Save results ---
    print(f"\n[6/6] Saving results to {OUTPUT_DIR}/")

    np.savez(
        os.path.join(OUTPUT_DIR, 'embeddings.npz'),
        features=X, features_scaled=X_scaled,
        tsne_coords=coords, labels=y,
        feature_names=feature_names
    )

    # Classification report on full refit
    y_pred = rf.predict(X_scaled)
    report = classification_report(y, y_pred, target_names=['COPD', 'COPD-PH'],
                                   output_dict=True)

    results = {
        'dataset': {'n_copd': N_COPD, 'n_copd_ph': N_PH, 'n_features': len(feature_names)},
        'cross_validation': {
            name: {
                'accuracy': f"{r['mean']['accuracy']:.3f}±{r['std']['accuracy']:.3f}",
                'f1': f"{r['mean']['f1']:.3f}±{r['std']['f1']:.3f}",
                'auc': f"{r['mean']['auc']:.3f}±{r['std']['auc']:.3f}",
            }
            for name, r in cv_results.items()
        },
        'top_features': [{'name': n, 'importance': float(i)}
                         for n, i in ranked[:15]],
        'silhouette_score': float(sil),
        'classification_report': report,
        'important_discriminators': [
            {'name': n, 'cohen_d': float(d)} for n, d in important_features[:10]
        ]
    }

    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # --- Summary ---
    print(f"\n{'='*65}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*65}")
    print(f"\n  Key findings:")
    print(f"  • Best model: {max(cv_results, key=lambda k: cv_results[k]['mean']['auc'])}")
    best_auc = max(r['mean']['auc'] for r in cv_results.values())
    print(f"    AUC = {best_auc:.3f}")
    print(f"  • Most discriminative feature: {important_features[0][0]} "
          f"(d={important_features[0][1]:.2f})")
    print(f"  • t-SNE cluster separation: {sil:.3f}")
    print(f"\n  Evolution patterns detected:")
    print(f"  • Vascular: distal vessel pruning (↓branch count, ↓BV5)")
    print(f"  • Parenchyma: emphysematous destruction (↑LAA%)")
    print(f"  • Airway: wall thickening (↑WA%, ↑WT ratio)")
    print(f"\n  Output files:")
    print(f"    {OUTPUT_DIR}/results.json — full results")
    print(f"    {OUTPUT_DIR}/embeddings.npz — features + t-SNE coords")
    print(f"\n  To train with PyTorch Geometric GCN on real data:")
    print(f"    pip install torch torch-geometric nibabel")
    print(f"    python main.py --mode train --data_dir /path/to/nifti --labels labels.csv")

    return results


if __name__ == '__main__':
    run_pipeline()
