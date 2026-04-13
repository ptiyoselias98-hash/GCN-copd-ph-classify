"""
Graph construction from vascular skeleton topology.

Converts parsed skeleton branches into a PyTorch Geometric Data object
with node features, edge indices, and edge attributes.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

try:
    from torch_geometric.data import Data
except ImportError:
    # Fallback: define minimal Data class for environments without PyG
    class Data:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


class VascularGraphBuilder:
    """
    Build a graph representation of the pulmonary vascular tree.

    Nodes = bifurcation/terminal points
    Edges = vessel segments connecting them
    Node features = morphological + CT-derived features
    """

    def __init__(
        self,
        spatial_edge_threshold: float = 15.0,
        add_spatial_edges: bool = True,
        use_directed: bool = False
    ):
        self.spatial_edge_threshold = spatial_edge_threshold
        self.add_spatial_edges = add_spatial_edges
        self.use_directed = use_directed

    def build_graph(
        self,
        branches: List[Dict],
        branch_features: List[Dict],
        strahler_orders: Optional[Dict[int, int]] = None,
        label: Optional[int] = None
    ) -> Data:
        """
        Construct a PyTorch Geometric graph from branches.

        Args:
            branches: List of branch dicts from VesselSkeleton.trace_branches()
            branch_features: List of feature dicts from compute_branch_features()
            strahler_orders: Optional Strahler orders for nodes
            label: Optional graph-level label (0=COPD, 1=COPD-PH)

        Returns:
            PyG Data object with:
              x: (N, F) node features
              edge_index: (2, E) edge connectivity
              edge_attr: (E, D) edge features
              y: graph label
              pos: (N, 3) node spatial positions
        """
        # --- Step 1: Identify unique nodes ---
        point_to_id = {}
        node_positions = []
        node_id = 0

        for branch in branches:
            for pt_key in ['start', 'end']:
                pt = tuple(branch[pt_key])
                if pt not in point_to_id:
                    point_to_id[pt] = node_id
                    node_positions.append(pt)
                    node_id += 1

        num_nodes = len(point_to_id)
        if num_nodes == 0:
            return self._empty_graph(label)

        # --- Step 2: Build edges from branches ---
        edge_src, edge_dst = [], []
        edge_features_list = []

        for i, (branch, feat) in enumerate(zip(branches, branch_features)):
            src = point_to_id[tuple(branch['start'])]
            dst = point_to_id[tuple(branch['end'])]

            edge_src.append(src)
            edge_dst.append(dst)

            # Edge features: diameter, length, tortuosity
            e_feat = [feat['diameter'], feat['length'], feat['tortuosity']]
            edge_features_list.append(e_feat)

            if not self.use_directed:
                edge_src.append(dst)
                edge_dst.append(src)
                edge_features_list.append(e_feat)

        # --- Step 3: Add spatial proximity edges ---
        if self.add_spatial_edges and num_nodes > 1:
            pos_array = np.array(node_positions, dtype=float)
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    dist = np.linalg.norm(pos_array[i] - pos_array[j])
                    if dist < self.spatial_edge_threshold:
                        # Check if this edge already exists
                        pair_exists = False
                        for si, di in zip(edge_src, edge_dst):
                            if (si == i and di == j) or (si == j and di == i):
                                pair_exists = True
                                break
                        if not pair_exists:
                            edge_src.extend([i, j])
                            edge_dst.extend([j, i])
                            e_spatial = [0.0, dist, 1.0]  # diameter=0, length=dist, tortuosity=1
                            edge_features_list.extend([e_spatial, e_spatial])

        # --- Step 4: Compute node features ---
        # Aggregate branch features to nodes (average over incident branches)
        node_feat_accum = defaultdict(list)

        for branch, feat in zip(branches, branch_features):
            src = point_to_id[tuple(branch['start'])]
            dst = point_to_id[tuple(branch['end'])]

            feat_vec = self._branch_feat_to_vec(feat)
            node_feat_accum[src].append(feat_vec)
            node_feat_accum[dst].append(feat_vec)

        # Build feature matrix
        feat_dim = 12  # diameter, length, tortuosity, density, orient(3), centroid(3), strahler, degree
        node_features = np.zeros((num_nodes, feat_dim), dtype=np.float32)

        for nid in range(num_nodes):
            if nid in node_feat_accum and len(node_feat_accum[nid]) > 0:
                feats = np.array(node_feat_accum[nid])
                # Average features from incident branches
                node_features[nid, :min(feats.shape[1], 10)] = np.mean(feats, axis=0)[:10]

            # Strahler order
            pt = node_positions[nid]
            if strahler_orders and nid in strahler_orders:
                node_features[nid, 10] = strahler_orders[nid]

            # Degree (number of incident branches)
            degree = len(node_feat_accum.get(nid, []))
            node_features[nid, 11] = degree

        # --- Step 5: Assemble PyG Data ---
        x = torch.tensor(node_features, dtype=torch.float)
        pos = torch.tensor(np.array(node_positions, dtype=np.float32))

        if len(edge_src) > 0:
            edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
            edge_attr = torch.tensor(
                np.array(edge_features_list, dtype=np.float32)
            )
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 3), dtype=torch.float)

        y = torch.tensor([label], dtype=torch.long) if label is not None else None

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos,
            y=y,
            num_nodes=num_nodes
        )

        return data

    def _branch_feat_to_vec(self, feat: Dict) -> np.ndarray:
        """Convert branch feature dict to fixed-length vector."""
        vec = np.zeros(10, dtype=np.float32)
        vec[0] = feat.get('diameter', 0.0)
        vec[1] = feat.get('length', 0.0)
        vec[2] = feat.get('tortuosity', 1.0)
        vec[3] = feat.get('mean_ct_density', 0.0)
        orient = feat.get('orientation', [0, 0, 0])
        vec[4:7] = orient
        centroid = feat.get('centroid', [0, 0, 0])
        vec[7:10] = centroid
        return vec

    def _empty_graph(self, label=None):
        """Return a minimal graph with one dummy node."""
        x = torch.zeros((1, 12), dtype=torch.float)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 3), dtype=torch.float)
        pos = torch.zeros((1, 3), dtype=torch.float)
        y = torch.tensor([label], dtype=torch.long) if label is not None else None
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, y=y, num_nodes=1)


def normalize_graph_features(data_list: List[Data]) -> List[Data]:
    """
    Normalize node features across an entire dataset.
    Uses z-score normalization per feature dimension.
    """
    all_feats = torch.cat([d.x for d in data_list], dim=0)
    mean = all_feats.mean(dim=0)
    std = all_feats.std(dim=0)
    std[std < 1e-6] = 1.0  # Avoid division by zero

    for d in data_list:
        d.x = (d.x - mean) / std

    return data_list
