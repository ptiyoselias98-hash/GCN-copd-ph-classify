#!/usr/bin/env python3
"""
rebuild_cache_separate.py — Phase 2: build per-structure graphs from raw masks.

This script runs on the remote server where the raw NIfTI masks live.
It replaces the heuristic partition (Phase 1) with ground-truth
structure-specific skeletons.

For each patient:
  1. Skeletonize artery.nii.gz → Graph_A
  2. Skeletonize vein.nii.gz   → Graph_V
  3. Skeletonize airway.nii.gz → Graph_W
  4. Save as {case_id}_tri.pkl

The TriStructureGCN model works identically with either cache format.

Usage (on remote server):
    python rebuild_cache_separate.py \\
        --data_dir /path/to/raw \\
        --labels /path/to/labels.csv \\
        --output_cache ./cache_tri
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


def _safe_import():
    """Import heavy dependencies; fail gracefully if not on server."""
    try:
        import nibabel as nib
        from skimage.morphology import skeletonize
        from scipy import ndimage
        import torch
        from torch_geometric.data import Data
        return nib, skeletonize, ndimage, torch, Data
    except ImportError as e:
        log.error("Missing dependency: %s", e)
        log.error("This script runs on the GPU server with nibabel, skimage, scipy, torch, torch_geometric.")
        sys.exit(1)


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """3D binary thinning → centerline skeleton."""
    _, skeletonize_fn, _, _, _ = _safe_import()
    binary = (mask > 0).astype(np.uint8)
    skel = skeletonize_fn(binary)
    return (skel > 0).astype(np.uint8)


def classify_voxels(skeleton: np.ndarray) -> dict:
    """Classify skeleton voxels by 26-connectivity neighbor count."""
    _, _, ndimage, _, _ = _safe_import()
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    nc = ndimage.convolve(skeleton, kernel, mode='constant', cval=0) * skeleton
    return {
        'endpoints': np.argwhere((nc == 1) & (skeleton > 0)),
        'bifurcations': np.argwhere((nc >= 3) & (skeleton > 0)),
        'segments': np.argwhere((nc == 2) & (skeleton > 0)),
    }


def trace_branches(skeleton, classified, min_branch_length=3):
    """Trace branches between key points (same as skeleton.py)."""
    key_points = []
    if len(classified['bifurcations']) > 0:
        key_points.extend(classified['bifurcations'].tolist())
    if len(classified['endpoints']) > 0:
        key_points.extend(classified['endpoints'].tolist())

    if not key_points:
        return []

    key_set = set(tuple(p) for p in key_points)
    visited = np.zeros_like(skeleton, dtype=bool)
    branches = []

    offsets = [(dz, dy, dx) for dz in [-1, 0, 1] for dy in [-1, 0, 1]
               for dx in [-1, 0, 1] if not (dz == 0 and dy == 0 and dx == 0)]

    def get_neighbors(pos, skel):
        z, y, x = pos
        return [(z + dz, y + dy, x + dx) for dz, dy, dx in offsets
                if 0 <= z + dz < skel.shape[0] and 0 <= y + dy < skel.shape[1]
                and 0 <= x + dx < skel.shape[2] and skel[z + dz, y + dy, x + dx] > 0]

    for kp in key_points:
        kp_t = tuple(kp)
        for nbr in get_neighbors(kp_t, skeleton):
            if visited[nbr[0], nbr[1], nbr[2]] and nbr not in key_set:
                continue
            path = [kp_t]
            current = nbr
            branch_vis = {kp_t}
            while current not in key_set or current == kp_t:
                if current in branch_vis:
                    break
                path.append(current)
                branch_vis.add(current)
                visited[current[0], current[1], current[2]] = True
                if current in key_set and current != kp_t:
                    break
                nxt = [n for n in get_neighbors(current, skeleton) if n not in branch_vis or n in key_set]
                if not nxt:
                    break
                kn = [n for n in nxt if n in key_set]
                current = kn[0] if kn else nxt[0]
            if current in key_set and current != kp_t:
                path.append(current)
            if len(path) >= min_branch_length:
                branches.append({
                    'start': path[0], 'end': path[-1],
                    'path': np.array(path), 'length_voxels': len(path),
                })

    # Deduplicate
    unique = []
    seen = set()
    for b in branches:
        key = (min(b['start'], b['end']), max(b['start'], b['end']))
        if key not in seen:
            seen.add(key)
            unique.append(b)
    return unique


def compute_branch_features(branch, mask, ct_volume=None, spacing=(1, 1, 1), dt=None):
    """Compute morphological features per branch.

    `dt` (precomputed distance transform) should be passed in when processing
    multiple branches of the same mask — recomputing it per branch turns a
    single ~5 s DT into hours of wall-clock over 200-500 branches.
    """
    _, _, ndimage, _, _ = _safe_import()
    path = branch['path']
    spacing = np.array(spacing)

    if dt is None:
        dt = ndimage.distance_transform_edt(mask > 0, sampling=spacing)
    radii = [dt[int(p[0]), int(p[1]), int(p[2])]
             for p in path if 0 <= int(p[0]) < dt.shape[0]
             and 0 <= int(p[1]) < dt.shape[1] and 0 <= int(p[2]) < dt.shape[2]]
    diameter = 2.0 * np.mean(radii) if radii else 0.0

    diffs = np.diff(path.astype(float), axis=0) * spacing
    length = np.sum(np.sqrt(np.sum(diffs ** 2, axis=1)))

    start_p = path[0].astype(float) * spacing
    end_p = path[-1].astype(float) * spacing
    euclidean = np.linalg.norm(end_p - start_p)
    tortuosity = length / max(euclidean, 1e-6)

    mean_density = 0.0
    if ct_volume is not None:
        dens = [float(ct_volume[int(p[0]), int(p[1]), int(p[2])])
                for p in path if 0 <= int(p[0]) < ct_volume.shape[0]
                and 0 <= int(p[1]) < ct_volume.shape[1] and 0 <= int(p[2]) < ct_volume.shape[2]]
        mean_density = np.mean(dens) if dens else 0.0

    direction = (path[-1].astype(float) - path[0].astype(float)) * spacing
    norm = np.linalg.norm(direction)
    direction = direction / max(norm, 1e-6)
    centroid = np.mean(path.astype(float) * spacing, axis=0)

    return {
        'diameter': diameter, 'length': length, 'tortuosity': tortuosity,
        'mean_ct_density': mean_density, 'orientation': direction.tolist(),
        'centroid': centroid.tolist(), 'num_voxels': len(path),
    }


def build_graph_from_mask(
    mask: np.ndarray,
    ct_volume: Optional[np.ndarray] = None,
    spacing: Tuple[float, ...] = (1, 1, 1),
    label: int = 0,
    spatial_threshold: float = 15.0,
):
    """
    Full pipeline: mask → skeleton → branches → PyG graph.
    Returns Data with (N, 12) node features.
    """
    _, _, ndimage, torch, Data = _safe_import()
    from collections import defaultdict

    skel = skeletonize_mask(mask)
    classified = classify_voxels(skel)
    branches = trace_branches(skel, classified)

    if not branches:
        return Data(
            x=torch.zeros((1, 12), dtype=torch.float),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            pos=torch.zeros((1, 3), dtype=torch.float),
            y=torch.tensor([label], dtype=torch.long),
            num_nodes=1,
        )

    # DT is shared across all branches of this mask (huge perf win).
    dt_shared = ndimage.distance_transform_edt(mask > 0, sampling=spacing)
    branch_features = [compute_branch_features(b, mask, ct_volume, spacing, dt=dt_shared)
                       for b in branches]

    # Build graph (same logic as graph_builder.py)
    point_to_id = {}
    node_positions = []
    nid = 0
    for b in branches:
        for pt_key in ['start', 'end']:
            pt = tuple(b[pt_key])
            if pt not in point_to_id:
                point_to_id[pt] = nid
                node_positions.append(pt)
                nid += 1

    num_nodes = len(point_to_id)
    edge_src, edge_dst, edge_feats = [], [], []
    node_feat_accum = defaultdict(list)

    for b, feat in zip(branches, branch_features):
        src = point_to_id[tuple(b['start'])]
        dst = point_to_id[tuple(b['end'])]
        edge_src.extend([src, dst])
        edge_dst.extend([dst, src])
        ef = [feat['diameter'], feat['length'], feat['tortuosity']]
        edge_feats.extend([ef, ef])

        vec = np.zeros(10, dtype=np.float32)
        vec[0] = feat['diameter']
        vec[1] = feat['length']
        vec[2] = feat['tortuosity']
        vec[3] = feat['mean_ct_density']
        vec[4:7] = feat['orientation']
        vec[7:10] = feat['centroid']
        node_feat_accum[src].append(vec)
        node_feat_accum[dst].append(vec)

    x = np.zeros((num_nodes, 12), dtype=np.float32)
    for n in range(num_nodes):
        if node_feat_accum[n]:
            feats = np.array(node_feat_accum[n])
            x[n, :10] = np.mean(feats, axis=0)
        x[n, 11] = len(node_feat_accum.get(n, []))

    return Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long) if edge_src else torch.zeros((2, 0), dtype=torch.long),
        pos=torch.tensor(np.array(node_positions, dtype=np.float32)),
        y=torch.tensor([label], dtype=torch.long),
        num_nodes=num_nodes,
    )


def process_patient(patient_dir: Path, label: int, spacing=(1, 1, 1)):
    """Build tri-structure graphs for one patient from NIfTI masks."""
    nib, _, _, _, _ = _safe_import()

    ct_path = patient_dir / "ct.nii.gz"
    ct_vol = nib.load(str(ct_path)).get_fdata() if ct_path.exists() else None

    result = {"label": label, "patient_id": patient_dir.name}

    for struct, mask_name in [("artery", "artery.nii.gz"), ("vein", "vein.nii.gz"), ("airway", "airway.nii.gz")]:
        mask_path = patient_dir / mask_name
        if mask_path.exists():
            mask = nib.load(str(mask_path)).get_fdata()
            graph = build_graph_from_mask(mask, ct_vol, spacing, label)
            result[struct] = graph
            log.info("  %s: %d nodes, %d edges", struct, graph.num_nodes, graph.edge_index.shape[1] if hasattr(graph, 'edge_index') else 0)
        else:
            log.warning("  %s mask not found: %s", struct, mask_path)
            import torch
            from torch_geometric.data import Data
            result[struct] = Data(
                x=torch.zeros((1, 12), dtype=torch.float),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                pos=torch.zeros((1, 3), dtype=torch.float),
                y=torch.tensor([label], dtype=torch.long),
                num_nodes=1,
            )

    return result


def _worker(args_tuple):
    """Multiprocessing worker: process a single patient end-to-end."""
    case_id, label, data_dir_str, output_cache_str, spacing = args_tuple
    data_dir = Path(data_dir_str)
    output_cache = Path(output_cache_str)
    patient_dir = data_dir / case_id
    out_path = output_cache / f"{case_id}_tri.pkl"

    if not patient_dir.is_dir():
        return (case_id, "missing_dir", None)
    if out_path.exists():
        return (case_id, "skipped_exists", None)

    try:
        result = process_patient(patient_dir, label, spacing)
        with out_path.open("wb") as f:
            pickle.dump(result, f)
        shapes = {s: (result[s].num_nodes, result[s].edge_index.shape[1])
                  for s in ("artery", "vein", "airway") if s in result}
        return (case_id, "done", shapes)
    except Exception as e:
        import traceback
        return (case_id, f"error: {e!r}", traceback.format_exc())


def main():
    logging.basicConfig(level=logging.INFO)

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", required=True, help="Raw data root with patient folders")
    p.add_argument("--labels", required=True, help="CSV: patient_id, label")
    p.add_argument("--output_cache", default="./cache_tri", help="Output cache directory")
    p.add_argument("--spacing", nargs=3, type=float, default=[1, 1, 1])
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel workers (1=serial). Each patient is independent so "
                        "output is identical to serial. 8 is safe on a 24-core box.")
    args = p.parse_args()

    labels = {}
    with open(args.labels, "r") as f:
        for row in csv.DictReader(f):
            cid = row.get("case_id") or row.get("patient_id")
            if cid:
                labels[cid] = int(row["label"])

    os.makedirs(args.output_cache, exist_ok=True)
    spacing = tuple(args.spacing)
    tasks = [
        (case_id, label, args.data_dir, args.output_cache, spacing)
        for case_id, label in labels.items()
    ]
    total = len(tasks)
    log.info("Total cases: %d  |  workers: %d  |  output: %s",
             total, args.workers, args.output_cache)

    done = skipped = missing = errored = 0

    if args.workers <= 1:
        for t in tasks:
            cid, status, info = _worker(t)
            if status == "done":
                done += 1
                log.info("[%d/%d done] %s  %s", done + skipped, total, cid, info)
            elif status == "skipped_exists":
                skipped += 1
            elif status == "missing_dir":
                missing += 1
                log.warning("missing dir: %s", cid)
            else:
                errored += 1
                log.error("error on %s: %s\n%s", cid, status, info)
    else:
        import multiprocessing as mp
        # `fork` context — workers inherit imports; each still calls _safe_import
        # inside the nibabel/skimage functions so modules are re-resolved safely.
        ctx = mp.get_context("fork")
        with ctx.Pool(args.workers) as pool:
            for cid, status, info in pool.imap_unordered(_worker, tasks):
                if status == "done":
                    done += 1
                    log.info("[%d/%d done] %s  %s",
                             done + skipped, total, cid, info)
                elif status == "skipped_exists":
                    skipped += 1
                    log.info("[%d/%d skip] %s (pkl already present)",
                             done + skipped, total, cid)
                elif status == "missing_dir":
                    missing += 1
                    log.warning("[%d/%d miss] %s (no patient dir)",
                                done + skipped + missing, total, cid)
                else:
                    errored += 1
                    log.error("[%d/%d err ] %s  %s\n%s",
                              done + skipped + missing + errored, total,
                              cid, status, info)

    log.info("Summary: done=%d skipped=%d missing=%d errored=%d (total=%d)",
             done, skipped, missing, errored, total)
    log.info("Cache written to %s", args.output_cache)


if __name__ == "__main__":
    main()
