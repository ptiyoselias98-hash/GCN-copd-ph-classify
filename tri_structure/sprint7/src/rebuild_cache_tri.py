#!/usr/bin/env python3
"""rebuild_cache_tri.py -- Sprint 7 Task 1: per-structure cache from raw masks.

Delta from rebuild_cache_separate.py (Phase 2):
  * Step 2  -- keep largest 26-connected component per mask before skeletonise.
  * Step 7  -- airway fallback: graphs with <3 nodes collapse to 1-node placeholder.
  * Step 6  -- Strahler feature (dim 10) defaults to 1 instead of 0.
  * --overwrite flag (Phase 2 skipped existing pkls, which would mask the fixes).

Run on the GPU server where nibabel / skimage / torch_geometric are available.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


def _deps():
    try:
        import nibabel as nib
        from skimage.morphology import skeletonize
        from scipy import ndimage
        import torch
        from torch_geometric.data import Data
        return nib, skeletonize, ndimage, torch, Data
    except ImportError as e:
        log.error("Missing dep: %s. Run inside pulmonary_bv5_py39.", e)
        sys.exit(1)


AIRWAY_MIN_NODES = 3  # spec Step 7


def keep_largest_component(mask_bin: np.ndarray) -> np.ndarray:
    """Return mask keeping only the largest 26-connected component.

    Small fragments (<100 voxels) produce spurious 1-2-node subgraphs that
    add noise without topology. Sprint 7 Step 2.
    """
    _, _, ndimage, _, _ = _deps()
    labelled, n = ndimage.label(mask_bin, structure=np.ones((3, 3, 3)))
    if n <= 1:
        return mask_bin.astype(np.uint8)
    sizes = ndimage.sum(mask_bin, labelled, range(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    return (labelled == keep).astype(np.uint8)


def skeletonise_mask(raw_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Binarise -> largest component -> 3D Lee thinning.

    Returns (mask_bin_single_cc, skeleton) -- both uint8.
    """
    _, skeletonize_fn, _, _, _ = _deps()
    mask_bin = (raw_mask > 0).astype(np.uint8)
    mask_bin = keep_largest_component(mask_bin)
    skel = skeletonize_fn(mask_bin)
    return mask_bin, (skel > 0).astype(np.uint8)


def classify_voxels(skeleton: np.ndarray) -> dict:
    _, _, ndimage, _, _ = _deps()
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    nc = ndimage.convolve(skeleton, kernel, mode="constant", cval=0) * skeleton
    return {
        "endpoints": np.argwhere((nc == 1) & (skeleton > 0)),
        "bifurcations": np.argwhere((nc >= 3) & (skeleton > 0)),
        "segments": np.argwhere((nc == 2) & (skeleton > 0)),
    }


def trace_branches(skeleton, classified, min_branch_length=3):
    key_points = []
    if len(classified["bifurcations"]) > 0:
        key_points.extend(classified["bifurcations"].tolist())
    if len(classified["endpoints"]) > 0:
        key_points.extend(classified["endpoints"].tolist())
    if not key_points:
        return []

    key_set = set(tuple(p) for p in key_points)
    visited = np.zeros_like(skeleton, dtype=bool)
    branches = []
    offsets = [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1)
               for dx in (-1, 0, 1) if not (dz == 0 and dy == 0 and dx == 0)]

    def neighbours(pos):
        z, y, x = pos
        out = []
        for dz, dy, dx in offsets:
            nz, ny, nx = z + dz, y + dy, x + dx
            if (0 <= nz < skeleton.shape[0] and 0 <= ny < skeleton.shape[1]
                    and 0 <= nx < skeleton.shape[2]
                    and skeleton[nz, ny, nx] > 0):
                out.append((nz, ny, nx))
        return out

    for kp in key_points:
        kp_t = tuple(kp)
        for nbr in neighbours(kp_t):
            if visited[nbr] and nbr not in key_set:
                continue
            path = [kp_t]
            current = nbr
            seen_here = {kp_t}
            while current not in key_set or current == kp_t:
                if current in seen_here:
                    break
                path.append(current)
                seen_here.add(current)
                visited[current] = True
                if current in key_set and current != kp_t:
                    break
                nxt = [n for n in neighbours(current) if n not in seen_here or n in key_set]
                if not nxt:
                    break
                kn = [n for n in nxt if n in key_set]
                current = kn[0] if kn else nxt[0]
            if current in key_set and current != kp_t:
                path.append(current)
            if len(path) >= min_branch_length:
                branches.append({
                    "start": path[0], "end": path[-1],
                    "path": np.array(path), "length_voxels": len(path),
                })

    # Dedup by undirected endpoint pair
    seen = set()
    uniq = []
    for b in branches:
        key = (min(b["start"], b["end"]), max(b["start"], b["end"]))
        if key not in seen:
            seen.add(key)
            uniq.append(b)
    return uniq


def compute_branch_features(branch, ct_volume, spacing, dt):
    """All branches of one mask share the same DT (computed from that mask only).

    Per Sprint 7 Step 5 -- DT uses this structure's own mask, not merged.
    """
    path = branch["path"]
    spacing_arr = np.array(spacing)

    radii = []
    for p in path:
        z, y, x = int(p[0]), int(p[1]), int(p[2])
        if 0 <= z < dt.shape[0] and 0 <= y < dt.shape[1] and 0 <= x < dt.shape[2]:
            radii.append(dt[z, y, x])
    diameter = 2.0 * float(np.mean(radii)) if radii else 0.0

    diffs = np.diff(path.astype(float), axis=0) * spacing_arr
    length = float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))))
    start_p = path[0].astype(float) * spacing_arr
    end_p = path[-1].astype(float) * spacing_arr
    euclidean = float(np.linalg.norm(end_p - start_p))
    tortuosity = length / max(euclidean, 1e-6)

    mean_density = 0.0
    if ct_volume is not None:
        dens = []
        for p in path:
            z, y, x = int(p[0]), int(p[1]), int(p[2])
            if (0 <= z < ct_volume.shape[0] and 0 <= y < ct_volume.shape[1]
                    and 0 <= x < ct_volume.shape[2]):
                dens.append(float(ct_volume[z, y, x]))
        mean_density = float(np.mean(dens)) if dens else 0.0

    direction = (path[-1].astype(float) - path[0].astype(float)) * spacing_arr
    norm = float(np.linalg.norm(direction))
    direction = direction / max(norm, 1e-6)
    centroid = np.mean(path.astype(float) * spacing_arr, axis=0)

    return {
        "diameter": diameter, "length": length, "tortuosity": tortuosity,
        "mean_ct_density": mean_density,
        "orientation": direction.tolist(),
        "centroid": centroid.tolist(),
        "num_voxels": len(path),
    }


def _placeholder_graph(label: int):
    _, _, _, torch, Data = _deps()
    return Data(
        x=torch.zeros((1, 12), dtype=torch.float),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        pos=torch.zeros((1, 3), dtype=torch.float),
        y=torch.tensor([label], dtype=torch.long),
        num_nodes=1,
    )


def build_graph_from_mask(
    raw_mask: np.ndarray,
    ct_volume: Optional[np.ndarray],
    spacing: Tuple[float, float, float],
    label: int,
):
    """raw mask -> (largest CC) -> skeleton -> branches -> PyG Data (N,12)."""
    _, _, ndimage, torch, Data = _deps()

    mask_bin, skel = skeletonise_mask(raw_mask)
    classified = classify_voxels(skel)
    branches = trace_branches(skel, classified)

    if not branches:
        return _placeholder_graph(label)

    dt = ndimage.distance_transform_edt(mask_bin > 0, sampling=spacing)
    branch_feats = [compute_branch_features(b, ct_volume, spacing, dt) for b in branches]

    # map endpoint voxels -> node id
    point_to_id = {}
    node_positions = []
    nid = 0
    for b in branches:
        for pt_key in ("start", "end"):
            pt = tuple(b[pt_key])
            if pt not in point_to_id:
                point_to_id[pt] = nid
                node_positions.append(pt)
                nid += 1

    num_nodes = len(point_to_id)
    edge_src, edge_dst = [], []
    node_feat_accum = defaultdict(list)

    for b, feat in zip(branches, branch_feats):
        src = point_to_id[tuple(b["start"])]
        dst = point_to_id[tuple(b["end"])]
        edge_src.extend([src, dst])
        edge_dst.extend([dst, src])

        vec = np.zeros(10, dtype=np.float32)
        vec[0] = feat["diameter"]
        vec[1] = feat["length"]
        vec[2] = feat["tortuosity"]
        vec[3] = feat["mean_ct_density"]
        vec[4:7] = feat["orientation"]
        vec[7:10] = feat["centroid"]
        node_feat_accum[src].append(vec)
        node_feat_accum[dst].append(vec)

    x = np.zeros((num_nodes, 12), dtype=np.float32)
    for n in range(num_nodes):
        vecs = node_feat_accum.get(n, [])
        if vecs:
            x[n, :10] = np.mean(np.array(vecs), axis=0)
        # dim 10: Strahler default 1 (spec: "computed if tree-structured, else 1")
        x[n, 10] = 1.0
        # dim 11: degree (count of incident branches)
        x[n, 11] = float(len(vecs))

    edge_index = (torch.tensor([edge_src, edge_dst], dtype=torch.long)
                  if edge_src else torch.zeros((2, 0), dtype=torch.long))
    return Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=edge_index,
        pos=torch.tensor(np.array(node_positions, dtype=np.float32)),
        y=torch.tensor([label], dtype=torch.long),
        num_nodes=num_nodes,
    )


def process_patient(patient_dir: Path, label: int):
    """Build artery/vein/airway graphs for one patient from NIfTI masks."""
    nib, _, _, _, _ = _deps()

    ct_path = patient_dir / "ct.nii.gz"
    ct_vol = None
    spacing = (1.0, 1.0, 1.0)
    if ct_path.exists():
        ct_img = nib.load(str(ct_path))
        ct_vol = ct_img.get_fdata()
        spacing = tuple(float(s) for s in ct_img.header.get_zooms()[:3])

    result = {"patient_id": patient_dir.name, "label": label, "spacing": spacing}

    for struct, fname in (("artery", "artery.nii.gz"),
                          ("vein",   "vein.nii.gz"),
                          ("airway", "airway.nii.gz")):
        mpath = patient_dir / fname
        if not mpath.exists():
            log.warning("  %s mask missing: %s -- placeholder", struct, mpath)
            result[struct] = _placeholder_graph(label)
            continue

        # header spacing may differ per mask in principle, use mask's own
        img = nib.load(str(mpath))
        mask_spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
        mask = img.get_fdata()
        graph = build_graph_from_mask(mask, ct_vol, mask_spacing, label)

        # Spec Step 7: airway with <3 nodes -> placeholder (avoid degenerate
        # empty-edge graphs that pool to zeros and wash out attention).
        if struct == "airway" and int(graph.num_nodes) < AIRWAY_MIN_NODES:
            log.warning("  airway yielded only %d nodes -- fallback to placeholder",
                        int(graph.num_nodes))
            graph = _placeholder_graph(label)

        n_edge = int(graph.edge_index.shape[1]) if graph.edge_index.numel() else 0
        log.info("  %s: nodes=%d edges=%d", struct, int(graph.num_nodes), n_edge)
        result[struct] = graph

    return result


def _worker(args_tuple):
    case_id, label, data_dir, out_cache, overwrite = args_tuple
    patient_dir = Path(data_dir) / case_id
    out_path = Path(out_cache) / f"{case_id}_tri.pkl"

    if not patient_dir.is_dir():
        return case_id, "missing_dir", None
    if out_path.exists() and not overwrite:
        return case_id, "skipped_exists", None

    try:
        result = process_patient(patient_dir, label)
        with out_path.open("wb") as f:
            pickle.dump(result, f)
        shapes = {
            s: (int(result[s].num_nodes),
                int(result[s].edge_index.shape[1]) if result[s].edge_index.numel() else 0)
            for s in ("artery", "vein", "airway") if s in result
        }
        return case_id, "done", shapes
    except Exception as e:
        import traceback
        return case_id, f"error: {e!r}", traceback.format_exc()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--output_cache", default="./cache_tri")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--overwrite", action="store_true",
                   help="Rebuild pkls even if they already exist in output_cache.")
    args = p.parse_args()

    labels = {}
    with open(args.labels, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row.get("case_id") or row.get("patient_id")
            if cid:
                labels[cid] = int(row["label"])

    os.makedirs(args.output_cache, exist_ok=True)
    tasks = [(cid, lab, args.data_dir, args.output_cache, args.overwrite)
             for cid, lab in labels.items()]
    total = len(tasks)
    log.info("rebuild_cache_tri: cases=%d workers=%d overwrite=%s output=%s",
             total, args.workers, args.overwrite, args.output_cache)

    done = skipped = missing = errored = 0

    def _log_result(cid, status, info):
        nonlocal done, skipped, missing, errored
        if status == "done":
            done += 1
            log.info("[%d/%d done] %s  %s", done + skipped, total, cid, info)
        elif status == "skipped_exists":
            skipped += 1
            log.info("[%d/%d skip] %s", done + skipped, total, cid)
        elif status == "missing_dir":
            missing += 1
            log.warning("[%d/%d miss] %s (no raw dir)", done + skipped + missing, total, cid)
        else:
            errored += 1
            log.error("[err] %s -- %s\n%s", cid, status, info)

    if args.workers <= 1:
        for t in tasks:
            _log_result(*_worker(t))
    else:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        with ctx.Pool(args.workers) as pool:
            for cid, status, info in pool.imap_unordered(_worker, tasks):
                _log_result(cid, status, info)

    log.info("summary: done=%d skipped=%d missing=%d errored=%d (total=%d)",
             done, skipped, missing, errored, total)
    log.info("cache written to %s", args.output_cache)


if __name__ == "__main__":
    main()
