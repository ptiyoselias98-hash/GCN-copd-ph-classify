"""Post-hoc augmentation of baseline graphs.

After the 2026-04-13 diagnosis (broadcasting graph-level scalars to every node
diluted the genuine local morphology signal and destroyed `enhanced/hybrid`),
the layout is split cleanly:

    node features (13D, per-node):
      [0]  diameter_calibrated   ← × sqrt(commercial_total_vol / pipeline_total_vol)
      [1]  length                ← unchanged
      [2]  tortuosity            ← unchanged
      [3]  ct_density_patched    ← overwritten with commercial artery/vein mean HU
      [4..6] orientation         ← unchanged
      [7..9] centroid            ← unchanged
      [10] strahler              ← unchanged
      [11] degree                ← unchanged
      [12] curvature             ← computed from pos+edge_index (genuinely local)

    graph-level globals (12 scalars, stored on ``data.global_features`` with
    shape (1, 12) so DataLoader stacks to (B, 12)):
      fractal_dim, artery_density, vein_density, vein_bv5, vein_branch_count,
      bv5_ratio, artery_vein_vol_ratio, total_bv5, lung_density_std,
      vein_bv10, total_branch_count, vessel_tortuosity

``HybridGCN.forward`` consumes ``global_features`` by concatenating them onto
the pooled graph embedding — never broadcasting them back to nodes.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data


EXPECTED_OUT_DIM = 13          # baseline 12 + curvature
BASELINE_IN_DIM = 12
GLOBAL_FEATURE_DIM = 12        # matches augment_graph's `globals_row`


def compute_node_curvature(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Per-node mean 1 − cos(angle) across all pairs of incident edges.

    Higher value = more bent connectivity at that node.
    Flat/straight (collinear edges) → 0. Orthogonal → 1. Reversed → 2.
    """
    n = pos.size(0)
    out = torch.zeros(n, dtype=torch.float32)
    if edge_index.numel() == 0 or n < 2:
        return out

    src, dst = edge_index[0], edge_index[1]
    vec = (pos[dst] - pos[src]).float()
    norms = vec.norm(dim=1, keepdim=True).clamp(min=1e-6)
    unit = vec / norms

    by_node: dict[int, list[torch.Tensor]] = {}
    for i in range(src.size(0)):
        s = int(src[i])
        by_node.setdefault(s, []).append(unit[i])

    for nid, vecs in by_node.items():
        if len(vecs) < 2:
            continue
        stacked = torch.stack(vecs, dim=0)
        dots = stacked @ stacked.T
        k = stacked.size(0)
        mask = ~torch.eye(k, dtype=torch.bool)
        off = dots[mask]
        out[nid] = (1.0 - off).mean().clamp(0.0, 2.0)
    return out


def _as_float(x: Optional[float]) -> float:
    """Coerce to float, mapping None/NaN/Inf to 0.0."""
    if x is None:
        return 0.0
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(v):
        return 0.0
    return v


def augment_graph(
    graph: Data,
    *,
    commercial_total_vol_ml: Optional[float],
    commercial_fractal_dim: Optional[float],
    commercial_artery_density: Optional[float],
    commercial_vein_density: Optional[float],
    pipeline_total_vol_ml: Optional[float],
    # --- new top-10 driven globals (optional; defaulted to 0 if absent) ---
    commercial_vein_bv5: Optional[float] = None,
    commercial_vein_branch_count: Optional[float] = None,
    commercial_bv5_ratio: Optional[float] = None,
    commercial_artery_vein_vol_ratio: Optional[float] = None,
    commercial_total_bv5: Optional[float] = None,
    commercial_lung_density_std: Optional[float] = None,
    commercial_vein_bv10: Optional[float] = None,
    commercial_total_branch_count: Optional[float] = None,
    commercial_vessel_tortuosity: Optional[float] = None,
) -> Data:
    """Return a *new* Data with 13-D x (baseline + curvature) and a graph-level
    ``global_features`` tensor of shape (1, 12) carrying the commercial scalars.

    The globals are intentionally NOT broadcast to nodes — that strategy was
    shown to dilute the real per-node morphology signal during message
    passing. ``HybridGCN.forward`` consumes ``data.global_features`` by
    concatenating onto the pooled graph embedding.
    """
    x = graph.x.clone()
    n = x.size(0)
    if x.size(1) != BASELINE_IN_DIM:
        raise ValueError(
            f"expected {BASELINE_IN_DIM}D node features, got {x.size(1)}"
        )

    # 1) diameter calibration (per-node: rescales local diameters)
    if (commercial_total_vol_ml and pipeline_total_vol_ml and
            pipeline_total_vol_ml > 1e-6):
        scale = float(np.sqrt(commercial_total_vol_ml / pipeline_total_vol_ml))
        x[:, 0] = x[:, 0] * scale

    # 2) density patch (overwrite bugged 0.0 column with a plausible HU constant)
    dens_vals = [commercial_artery_density, commercial_vein_density]
    dens_vals = [v for v in dens_vals if v is not None and np.isfinite(v)]
    if dens_vals:
        x[:, 3] = float(np.mean(dens_vals))

    # 3) curvature (genuinely per-node — stays in node features)
    if hasattr(graph, "pos") and graph.pos is not None:
        curvature = compute_node_curvature(graph.pos, graph.edge_index)
    else:
        curvature = torch.zeros(n, dtype=torch.float32)
    curvature = curvature.view(-1, 1)

    x_new = torch.cat([x, curvature], dim=1)  # (N, 13)
    assert x_new.size(1) == EXPECTED_OUT_DIM, (
        f"augment_graph produced {x_new.size(1)}D, expected {EXPECTED_OUT_DIM}"
    )

    # 4) graph-level globals (stored, NOT broadcast)
    globals_row = [
        _as_float(commercial_fractal_dim),
        _as_float(commercial_artery_density),
        _as_float(commercial_vein_density),
        _as_float(commercial_vein_bv5),
        _as_float(commercial_vein_branch_count),
        _as_float(commercial_bv5_ratio),
        _as_float(commercial_artery_vein_vol_ratio),
        _as_float(commercial_total_bv5),
        _as_float(commercial_lung_density_std),
        _as_float(commercial_vein_bv10),
        _as_float(commercial_total_branch_count),
        _as_float(commercial_vessel_tortuosity),
    ]
    assert len(globals_row) == GLOBAL_FEATURE_DIM

    new = graph.clone()
    new.x = x_new
    # shape (1, G) so PyG DataLoader stacks to (B, G) across the batch
    new.global_features = torch.tensor([globals_row], dtype=torch.float32)
    return new
