"""
Skeleton extraction and topology parsing from 3D vascular masks.

Pipeline:
  1. Load binary vessel mask (NIfTI)
  2. 3D thinning → centerline skeleton
  3. Parse skeleton into graph topology (bifurcations, terminals, segments)
  4. Compute per-node morphological features
"""

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize
from collections import deque
from typing import Dict, List, Tuple, Optional
import warnings


class VesselSkeleton:
    """Extract and parse 3D vascular skeleton from binary mask."""

    def __init__(self, min_branch_length: int = 3):
        self.min_branch_length = min_branch_length

    def extract_skeleton(self, mask: np.ndarray) -> np.ndarray:
        """
        Skeletonize a 3D binary vessel mask.

        Args:
            mask: Binary 3D array (H, W, D), 1 = vessel

        Returns:
            skeleton: Binary 3D array, 1 = centerline voxel
        """
        mask_binary = (mask > 0).astype(np.uint8)
        skeleton = skeletonize(mask_binary)
        return (skeleton > 0).astype(np.uint8)

    def classify_voxels(self, skeleton: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Classify skeleton voxels into endpoints, bifurcations, and segments.

        Uses 26-connectivity neighbor counting:
          - 1 neighbor  → endpoint (terminal)
          - 2 neighbors → segment (pass-through)
          - 3+ neighbors → bifurcation (junction)

        Returns:
            dict with keys 'endpoints', 'bifurcations', 'segments'
            each containing Nx3 coordinate arrays
        """
        # Build 26-connectivity kernel
        kernel = np.ones((3, 3, 3), dtype=np.uint8)
        kernel[1, 1, 1] = 0

        # Count neighbors for each skeleton voxel
        neighbor_count = ndimage.convolve(skeleton, kernel, mode='constant', cval=0)
        neighbor_count = neighbor_count * skeleton  # Only skeleton voxels

        endpoints = np.argwhere((neighbor_count == 1) & (skeleton > 0))
        bifurcations = np.argwhere((neighbor_count >= 3) & (skeleton > 0))
        segments = np.argwhere((neighbor_count == 2) & (skeleton > 0))

        return {
            'endpoints': endpoints,
            'bifurcations': bifurcations,
            'segments': segments
        }

    def trace_branches(
        self,
        skeleton: np.ndarray,
        classified: Dict[str, np.ndarray]
    ) -> List[Dict]:
        """
        Trace individual branches between junction/terminal points.

        Each branch is a path of connected skeleton voxels between two
        key points (bifurcation or endpoint).

        Returns:
            List of branch dicts:
              {
                'start': (z,y,x),
                'end': (z,y,x),
                'path': Nx3 array of voxel coordinates,
                'length_voxels': int
              }
        """
        # Create label map: key points get unique IDs
        key_points = []
        if len(classified['bifurcations']) > 0:
            key_points.extend(classified['bifurcations'].tolist())
        if len(classified['endpoints']) > 0:
            key_points.extend(classified['endpoints'].tolist())

        if len(key_points) == 0:
            return []

        key_set = set(tuple(p) for p in key_points)
        visited = np.zeros_like(skeleton, dtype=bool)
        branches = []

        # 26-connected neighborhood offsets
        offsets = []
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    offsets.append((dz, dy, dx))

        def get_neighbors(pos, skel):
            z, y, x = pos
            nbrs = []
            for dz, dy, dx in offsets:
                nz, ny, nx = z + dz, y + dy, x + dx
                if (0 <= nz < skel.shape[0] and
                    0 <= ny < skel.shape[1] and
                    0 <= nx < skel.shape[2] and
                    skel[nz, ny, nx] > 0):
                    nbrs.append((nz, ny, nx))
            return nbrs

        # Trace from each key point
        for kp in key_points:
            kp_tuple = tuple(kp)
            neighbors = get_neighbors(kp_tuple, skeleton)

            for nbr in neighbors:
                if visited[nbr[0], nbr[1], nbr[2]] and nbr not in key_set:
                    continue

                # Trace path from kp through nbr until we hit another key point
                path = [kp_tuple]
                current = nbr
                branch_visited = {kp_tuple}

                while current not in key_set or current == kp_tuple:
                    if current in branch_visited:
                        break
                    path.append(current)
                    branch_visited.add(current)
                    visited[current[0], current[1], current[2]] = True

                    if current in key_set and current != kp_tuple:
                        break

                    next_nbrs = [
                        n for n in get_neighbors(current, skeleton)
                        if n not in branch_visited or n in key_set
                    ]

                    if len(next_nbrs) == 0:
                        break
                    elif len(next_nbrs) == 1:
                        current = next_nbrs[0]
                    else:
                        # Prefer key points
                        key_nbrs = [n for n in next_nbrs if n in key_set]
                        current = key_nbrs[0] if key_nbrs else next_nbrs[0]

                if current in key_set and current != kp_tuple:
                    path.append(current)

                if len(path) >= self.min_branch_length:
                    branches.append({
                        'start': path[0],
                        'end': path[-1],
                        'path': np.array(path),
                        'length_voxels': len(path)
                    })

        # Deduplicate branches (A→B == B→A)
        unique_branches = []
        seen = set()
        for b in branches:
            key = (
                min(b['start'], b['end']),
                max(b['start'], b['end'])
            )
            if key not in seen:
                seen.add(key)
                unique_branches.append(b)

        return unique_branches

    def compute_branch_features(
        self,
        branch: Dict,
        mask: np.ndarray,
        ct_volume: Optional[np.ndarray] = None,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        dt: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute morphological features for a single branch.

        Features:
          - diameter: mean cross-sectional diameter (from distance transform)
          - length: physical path length in mm
          - tortuosity: path length / Euclidean distance
          - mean_ct_density: average HU along the branch (if CT provided)
          - orientation: principal direction vector (3 components)

        Args:
            branch: Branch dict from trace_branches
            mask: Original binary vessel mask
            ct_volume: Optional CT volume for density features
            spacing: Voxel spacing in mm (z, y, x)

        Returns:
            Feature dict
        """
        path = branch['path']
        spacing = np.array(spacing)

        # --- Diameter from distance transform (passed in to avoid recomputation) ---
        if dt is None:
            dt = ndimage.distance_transform_edt(mask > 0, sampling=spacing)
        radii = []
        for pt in path:
            z, y, x = int(pt[0]), int(pt[1]), int(pt[2])
            if 0 <= z < dt.shape[0] and 0 <= y < dt.shape[1] and 0 <= x < dt.shape[2]:
                radii.append(dt[z, y, x])
        diameter = 2.0 * np.mean(radii) if radii else 0.0

        # --- Physical length ---
        diffs = np.diff(path.astype(float), axis=0) * spacing
        segment_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
        length = np.sum(segment_lengths)

        # --- Tortuosity ---
        start_phys = path[0].astype(float) * spacing
        end_phys = path[-1].astype(float) * spacing
        euclidean = np.linalg.norm(end_phys - start_phys)
        tortuosity = length / max(euclidean, 1e-6)

        # --- CT density ---
        mean_density = 0.0
        if ct_volume is not None:
            densities = []
            for pt in path:
                z, y, x = int(pt[0]), int(pt[1]), int(pt[2])
                if (0 <= z < ct_volume.shape[0] and
                    0 <= y < ct_volume.shape[1] and
                    0 <= x < ct_volume.shape[2]):
                    densities.append(float(ct_volume[z, y, x]))
            if densities:
                mean_density = np.mean(densities)

        # --- Orientation (principal direction) ---
        if len(path) >= 2:
            direction = (path[-1].astype(float) - path[0].astype(float)) * spacing
            norm = np.linalg.norm(direction)
            direction = direction / max(norm, 1e-6)
        else:
            direction = np.array([0.0, 0.0, 0.0])

        # --- Centroid ---
        centroid = np.mean(path.astype(float) * spacing, axis=0)

        return {
            'diameter': diameter,
            'length': length,
            'tortuosity': tortuosity,
            'mean_ct_density': mean_density,
            'orientation': direction.tolist(),
            'centroid': centroid.tolist(),
            'num_voxels': len(path)
        }


def compute_strahler_order(
    adjacency: Dict[int, List[int]],
    root: int,
    is_terminal: Dict[int, bool]
) -> Dict[int, int]:
    """
    Compute Strahler stream order for a tree-structured graph.

    Strahler order is a hierarchical ordering where:
      - Terminal (leaf) branches = order 1
      - A parent's order = max child order if children differ,
        or max child order + 1 if two or more children share the max

    Args:
        adjacency: node_id → list of child node_ids
        root: root node id
        is_terminal: node_id → True if terminal

    Returns:
        Dict mapping node_id → Strahler order
    """
    order = {}

    def _compute(node, parent=None):
        children = [c for c in adjacency.get(node, []) if c != parent]

        if not children or is_terminal.get(node, False):
            order[node] = 1
            return 1

        child_orders = []
        for c in children:
            child_orders.append(_compute(c, parent=node))

        child_orders.sort(reverse=True)
        if len(child_orders) >= 2 and child_orders[0] == child_orders[1]:
            order[node] = child_orders[0] + 1
        else:
            order[node] = child_orders[0]

        return order[node]

    _compute(root)
    return order
