"""
graph_partitioner.py — Build per-structure graphs from existing cache or raw masks.

Two modes:
  A) from_cache: partition the unified graph using heuristic (Strahler + degree)
     → quick, works now, good for development and initial experiments
  B) from_masks: re-skeletonize each mask independently
     → correct, used for final experiments with real data

Both modes produce the same output format: a dict per patient with
  {"artery": Data, "vein": Data, "airway": Data, "label": int, ...}
"""

import numpy as np
import torch
import pickle
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

try:
    from torch_geometric.data import Data
except ImportError:
    class Data:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Mode A: heuristic partition from unified cached graph
# ─────────────────────────────────────────────────────────────────────

def _subgraph(graph: Data, node_indices: list, label: int) -> Data:
    """Extract induced subgraph for the given node set."""
    if len(node_indices) == 0:
        feat_dim = graph.x.shape[1] if hasattr(graph, 'x') else 15
        return Data(
            x=torch.zeros((1, feat_dim), dtype=torch.float),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            pos=torch.zeros((1, 3), dtype=torch.float),
            y=torch.tensor([label], dtype=torch.long),
            num_nodes=1,
        )

    idx_set = set(node_indices)
    remap = {old: new for new, old in enumerate(node_indices)}

    x = graph.x[node_indices]
    pos = graph.pos[node_indices] if hasattr(graph, 'pos') and graph.pos is not None else torch.zeros((len(node_indices), 3))

    if hasattr(graph, 'edge_index') and graph.edge_index is not None and graph.edge_index.shape[1] > 0:
        ei = graph.edge_index
        mask = torch.tensor([
            int(ei[0, e]) in idx_set and int(ei[1, e]) in idx_set
            for e in range(ei.shape[1])
        ], dtype=torch.bool)
        if mask.any():
            kept = ei[:, mask]
            new_src = torch.tensor([remap[int(s)] for s in kept[0]], dtype=torch.long)
            new_dst = torch.tensor([remap[int(d)] for d in kept[1]], dtype=torch.long)
            edge_index = torch.stack([new_src, new_dst])
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(
        x=x, edge_index=edge_index, pos=pos,
        y=torch.tensor([label], dtype=torch.long),
        num_nodes=len(node_indices),
    )


def partition_unified_graph(graph: Data, topology: dict, label: int) -> Dict[str, Data]:
    """
    Split a unified vascular graph into artery and vein subgraphs.

    Heuristic: nodes with higher Strahler order and larger diameter are more
    likely arterial (closer to hilum, upstream). Nodes with lower Strahler
    and smaller diameter are more likely venous (drainage, downstream).

    This is an approximation — the real solution is Mode B (separate masks).
    """
    n = graph.x.shape[0]
    if n < 4:
        return {
            "artery": _subgraph(graph, list(range(n)), label),
            "vein": _subgraph(graph, [], label),
        }

    artery_frac = topology.get("artery_frac", 0.5)
    vein_frac = topology.get("vein_frac", 0.3)
    total = artery_frac + vein_frac
    if total < 1e-6:
        a_ratio = 0.5
    else:
        a_ratio = artery_frac / total

    x = graph.x.numpy()
    strahler = x[:, 10] if x.shape[1] > 10 else np.zeros(n)
    diameter = x[:, 0] if x.shape[1] > 0 else np.ones(n)

    # Composite score: higher = more likely arterial
    score = strahler * 2.0 + diameter + np.random.randn(n) * 0.01
    rank = np.argsort(-score)

    n_artery = max(2, int(n * a_ratio))
    artery_nodes = rank[:n_artery].tolist()
    vein_nodes = rank[n_artery:].tolist()

    return {
        "artery": _subgraph(graph, artery_nodes, label),
        "vein": _subgraph(graph, vein_nodes, label),
    }


def build_airway_pseudograph(features: dict, label: int, feat_dim: int = 15) -> Data:
    """
    Build a minimal placeholder graph from cached airway scalar features.

    Since the current cache has no airway skeleton graph, we create a
    single-node graph whose features encode the airway scalars.  This is
    explicitly a placeholder — Experiment Phase 2 will replace this with
    real airway skeleton graphs built from airway.nii.gz masks.

    The feature vector is padded/truncated to match feat_dim so that all
    three encoders can share the same architecture.
    """
    airway = features.get("airway", {})
    scalars = [
        airway.get("wall_area_pct", 0.0),
        airway.get("wall_thickness_ratio", 0.0),
        airway.get("airway_volume_ml", 0.0),
        airway.get("airway_count", 0.0),
        airway.get("mean_airway_hu", 0.0),
        airway.get("airway_to_lung_ratio", 0.0),
    ]

    # Pad to feat_dim
    vec = np.zeros(feat_dim, dtype=np.float32)
    for i, v in enumerate(scalars):
        if i < feat_dim:
            vec[i] = float(v) if v is not None and np.isfinite(float(v)) else 0.0

    x = torch.tensor(vec, dtype=torch.float).unsqueeze(0)  # (1, feat_dim)

    return Data(
        x=x,
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        pos=torch.zeros((1, 3), dtype=torch.float),
        y=torch.tensor([label], dtype=torch.long),
        num_nodes=1,
    )


# ─────────────────────────────────────────────────────────────────────
# Data loader
# ─────────────────────────────────────────────────────────────────────

def load_labels(path: Path) -> Dict[str, int]:
    labels = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row.get("case_id") or row.get("patient_id")
            if cid:
                labels[cid] = int(row["label"])
    return labels


def load_mpap(path: Optional[Path]) -> Dict[str, Optional[float]]:
    if path is None or not path.exists():
        return {}
    with path.open("r") as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        out[k] = float(v) if v is not None else None
    return out


def load_tri_structure_dataset(
    cache_dir: Path,
    labels: Dict[str, int],
    mpap_lookup: Optional[Dict[str, Optional[float]]] = None,
) -> List[dict]:
    """
    Load cached graphs → partition into tri-structure format.

    Returns list of dicts:
      {
        "case_id": str,
        "label": int,
        "mpap": float or None,
        "artery": Data,
        "vein": Data,
        "airway": Data,   # placeholder if no airway skeleton
        "features": dict,  # original cached scalars
        "topology": dict,
      }
    """
    dataset = []

    for case_id, label in labels.items():
        tri_path = cache_dir / f"{case_id}_tri.pkl"
        legacy_path = cache_dir / f"{case_id}.pkl"

        if tri_path.exists():
            # Phase 2 schema: per-structure graphs already separated.
            with tri_path.open("rb") as f:
                entry = pickle.load(f)

            for struct in ("artery", "vein", "airway"):
                g = entry[struct]
                g.y = torch.tensor([label], dtype=torch.long)

            sample = {
                "case_id": case_id,
                "label": label,
                "mpap": None,
                "artery": entry["artery"],
                "vein": entry["vein"],
                "airway": entry["airway"],
                "features": {},
                "topology": {},
            }
        elif legacy_path.exists():
            # Phase 1 schema: unified graph + heuristic partition.
            with legacy_path.open("rb") as f:
                entry = pickle.load(f)

            graph = entry["graph"]
            topology = entry.get("topology", {})
            features = entry.get("features", {})

            graph.y = torch.tensor([label], dtype=torch.long)
            av_graphs = partition_unified_graph(graph, topology, label)
            feat_dim = graph.x.shape[1]
            airway_graph = build_airway_pseudograph(features, label, feat_dim)

            sample = {
                "case_id": case_id,
                "label": label,
                "mpap": None,
                "artery": av_graphs["artery"],
                "vein": av_graphs["vein"],
                "airway": airway_graph,
                "features": features,
                "topology": topology,
            }
        else:
            continue

        if mpap_lookup and case_id in mpap_lookup:
            sample["mpap"] = mpap_lookup[case_id]

        # Interpretable per-patient graph signature (computed on raw graphs,
        # BEFORE per-structure node-feature normalization — stats should reflect
        # true topology, not normalized features).
        sample["signature"] = compute_graph_signature(sample)

        dataset.append(sample)

    log.info("Loaded %d tri-structure samples from %s", len(dataset), cache_dir)
    return dataset


def normalize_per_structure(dataset: List[dict], structures=("artery", "vein", "airway")):
    """Z-score normalize node features per structure type across all patients."""
    for struct in structures:
        all_x = []
        for d in dataset:
            g = d[struct]
            if g.x.shape[0] > 0 and not (g.x.shape[0] == 1 and g.x.abs().sum() < 1e-8):
                all_x.append(g.x)

        if not all_x:
            continue

        cat = torch.cat(all_x, dim=0)
        mean = cat.mean(dim=0)
        std = cat.std(dim=0)
        std[std < 1e-6] = 1.0

        for d in dataset:
            g = d[struct]
            if g.x.shape[0] > 0:
                g.x = (g.x - mean) / std

    return dataset


# ─────────────────────────────────────────────────────────────────────
# Graph signature — hand-crafted interpretable per-patient features
# ─────────────────────────────────────────────────────────────────────

# Keep ordering stable across patients.
SIGNATURE_FIELDS_PER_STRUCT = (
    "n_nodes",
    "n_edges",
    "mean_degree",
    "max_degree_frac",     # max_degree / n_nodes (scale invariant)
    "edge_density",        # 2E / (N*(N-1))
    "n_components",
    "largest_cc_frac",     # largest CC size / n_nodes
)
SIGNATURE_CROSS_FIELDS = (
    "artery_vein_node_ratio",
    "airway_vein_node_ratio",
    "total_nodes_log",
)
SIGNATURE_STRUCTS = ("artery", "vein", "airway")
SIGNATURE_DIM = len(SIGNATURE_FIELDS_PER_STRUCT) * len(SIGNATURE_STRUCTS) + len(SIGNATURE_CROSS_FIELDS)


def _component_stats(num_nodes: int, edge_index: torch.Tensor) -> Tuple[int, int]:
    """
    Union-find over undirected edges. Returns (n_components, largest_cc_size).
    edge_index may be directed; we treat each (u, v) as an undirected link.
    """
    if num_nodes <= 0:
        return 0, 0
    parent = list(range(num_nodes))

    def find(u: int) -> int:
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u: int, v: int) -> None:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv

    if edge_index is not None and edge_index.numel() > 0:
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        for u, v in zip(src, dst):
            if 0 <= u < num_nodes and 0 <= v < num_nodes:
                union(u, v)

    sizes: Dict[int, int] = defaultdict(int)
    for i in range(num_nodes):
        sizes[find(i)] += 1
    return len(sizes), max(sizes.values())


def _struct_stats(graph: Data) -> Dict[str, float]:
    n = int(getattr(graph, "num_nodes", 0) or 0)
    ei = getattr(graph, "edge_index", None)
    if ei is None:
        e = 0
    else:
        e = int(ei.shape[1])

    if n <= 0:
        return {
            "n_nodes": 0.0, "n_edges": 0.0, "mean_degree": 0.0,
            "max_degree_frac": 0.0, "edge_density": 0.0,
            "n_components": 0.0, "largest_cc_frac": 0.0,
        }

    # Per-node degree (counts each (u->v) once; may double-count if undirected edges
    # are stored both ways — consistent across structures/patients so fine as a feature).
    deg = torch.zeros(n, dtype=torch.long)
    if e > 0:
        src = ei[0].long().clamp_(min=0, max=n - 1)
        deg.scatter_add_(0, src, torch.ones_like(src))
    mean_deg = float(deg.float().mean().item())
    max_deg = int(deg.max().item()) if e > 0 else 0
    density = (2.0 * e) / (n * (n - 1)) if n > 1 else 0.0
    n_cc, largest = _component_stats(n, ei if e > 0 else None)

    return {
        "n_nodes": float(n),
        "n_edges": float(e),
        "mean_degree": mean_deg,
        "max_degree_frac": float(max_deg) / n if n > 0 else 0.0,
        "edge_density": float(density),
        "n_components": float(n_cc),
        "largest_cc_frac": float(largest) / n if n > 0 else 0.0,
    }


def compute_graph_signature(sample: Dict) -> np.ndarray:
    """
    Hand-crafted interpretable signature for one patient.

    Returns a 1-D array of length SIGNATURE_DIM with a stable ordering:
      [<artery fields>..., <vein fields>..., <airway fields>..., <cross fields>...]
    """
    per: Dict[str, Dict[str, float]] = {}
    for s in SIGNATURE_STRUCTS:
        per[s] = _struct_stats(sample[s])

    vec: List[float] = []
    for s in SIGNATURE_STRUCTS:
        for f in SIGNATURE_FIELDS_PER_STRUCT:
            vec.append(per[s][f])

    # Cross-structure ratios (with small epsilon to keep bounded).
    eps = 1.0
    v_nodes = per["vein"]["n_nodes"]
    vec.append(per["artery"]["n_nodes"] / (v_nodes + eps))
    vec.append(per["airway"]["n_nodes"] / (v_nodes + eps))
    total = per["artery"]["n_nodes"] + per["vein"]["n_nodes"] + per["airway"]["n_nodes"]
    vec.append(float(np.log1p(total)))

    return np.asarray(vec, dtype=np.float32)


def signature_feature_names() -> List[str]:
    """Parallel list of feature names matching compute_graph_signature output."""
    names: List[str] = []
    for s in SIGNATURE_STRUCTS:
        for f in SIGNATURE_FIELDS_PER_STRUCT:
            names.append(f"{s}.{f}")
    names.extend(SIGNATURE_CROSS_FIELDS)
    return names


# ─────────────────────────────────────────────────────────────────────
# Mode B: from raw NIfTI masks (for future use on the server)
# ─────────────────────────────────────────────────────────────────────

def rebuild_from_masks_stub():
    """
    Placeholder for Phase 2: rebuild per-structure graphs from raw masks.

    On the remote server, each patient has:
      ct.nii.gz, artery.nii.gz, vein.nii.gz, lung.nii.gz, airway.nii.gz

    Phase 2 will:
      1. Skeletonize artery.nii.gz → skeleton_a → trace_branches → Graph_A
      2. Skeletonize vein.nii.gz   → skeleton_v → trace_branches → Graph_V
      3. Skeletonize airway.nii.gz → skeleton_w → trace_branches → Graph_W
      4. Save as {case_id}_tri.pkl with all three graphs

    This eliminates the heuristic partition and gives ground-truth
    structure-specific graphs.  The same TriStructureGCN model works
    with either cache format — only the data loading changes.
    """
    raise NotImplementedError(
        "Phase 2: rebuild_cache_tri_structure.py — "
        "skeletonize each mask independently.  "
        "Run on server with access to raw NIfTI masks."
    )
