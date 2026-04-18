"""Sprint 2 — Post-hoc augmentation of 12D → 16D node features.

Works directly on cached PyG Data objects (no raw CT re-processing):

  [0] diameter_calibrated   ← multiplied by sqrt(commercial_total_vol / pipeline_total_vol)
  [1] length                ← unchanged
  [2] tortuosity            ← unchanged
  [3] ct_density_patched    ← replaced by commercial artery/vein mean HU
                             (current value is bugged 0.0 in cache)
  [4..6] orientation        ← unchanged
  [7..9] centroid           ← unchanged
  [10] strahler             ← unchanged
  [11] degree               ← unchanged
  [12] curvature (NEW)      ← mean angle deviation at incident edges, from pos+edge_index
  [13] fractal_dim (NEW)    ← broadcast commercial 肺血管分形维度
  [14] artery_density (NEW) ← broadcast commercial 动脉平均密度HU
  [15] vein_density (NEW)   ← broadcast commercial 静脉平均密度HU
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data


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
    # vector per edge from src to dst in 3D
    vec = (pos[dst] - pos[src]).float()
    norms = vec.norm(dim=1, keepdim=True).clamp(min=1e-6)
    unit = vec / norms

    # group unit edges by their src node (outgoing directions)
    by_node: dict[int, list[torch.Tensor]] = {}
    for i in range(src.size(0)):
        s = int(src[i])
        by_node.setdefault(s, []).append(unit[i])

    for nid, vecs in by_node.items():
        if len(vecs) < 2:
            continue
        stacked = torch.stack(vecs, dim=0)  # (k, 3)
        dots = stacked @ stacked.T  # (k, k)
        k = stacked.size(0)
        mask = ~torch.eye(k, dtype=torch.bool)
        off = dots[mask]
        out[nid] = (1.0 - off).mean().clamp(0.0, 2.0)
    return out


def augment_graph(
    graph: Data,
    *,
    commercial_total_vol_ml: float | None,
    commercial_fractal_dim: float | None,
    commercial_artery_density: float | None,
    commercial_vein_density: float | None,
    pipeline_total_vol_ml: float | None,
) -> Data:
    """Return a *new* Data with 16D x."""
    x = graph.x.clone()  # (N, 12)
    n = x.size(0)
    if x.size(1) != 12:
        raise ValueError(f"expected 12D node features, got {x.size(1)}")

    # --- 1) diameter calibration
    if (commercial_total_vol_ml and pipeline_total_vol_ml and
            pipeline_total_vol_ml > 1e-6):
        scale = float(np.sqrt(commercial_total_vol_ml / pipeline_total_vol_ml))
        x[:, 0] = x[:, 0] * scale

    # --- 2) density patch (overwrite bugged 0.0 column)
    if commercial_artery_density is not None and commercial_vein_density is not None:
        density_mean = 0.5 * (float(commercial_artery_density) +
                              float(commercial_vein_density))
    elif commercial_artery_density is not None:
        density_mean = float(commercial_artery_density)
    elif commercial_vein_density is not None:
        density_mean = float(commercial_vein_density)
    else:
        density_mean = None
    if density_mean is not None:
        x[:, 3] = density_mean

    # --- 3) compute curvature
    curvature = compute_node_curvature(graph.pos, graph.edge_index) if hasattr(graph, "pos") else torch.zeros(n)
    curvature = curvature.view(-1, 1)

    # --- 4) broadcast globals
    fd = float(commercial_fractal_dim) if commercial_fractal_dim is not None else 0.0
    ad = float(commercial_artery_density) if commercial_artery_density is not None else 0.0
    vd = float(commercial_vein_density) if commercial_vein_density is not None else 0.0

    extras = torch.tensor(
        [[fd, ad, vd]] * n, dtype=torch.float32
    )  # (N, 3)

    x_new = torch.cat([x, curvature, extras], dim=1)  # (N, 16)

    new = graph.clone()
    new.x = x_new
    return new
