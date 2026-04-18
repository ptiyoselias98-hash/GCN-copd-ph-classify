"""Sprint 1.2 — HybridGCN: GCN graph embedding + commercial CT radiomics fusion.

Drop-in replacement for VascularSAGE: takes graph data plus a per-sample
radiomics vector (commercial CT features from xlsx), concatenates the
pooled GCN embedding with a normalized radiomics embedding, and classifies
through a small MLP.

Input dims (defaults):
    gcn_in        = 12     (node features produced by graph_builder)
    gcn_hidden    = 64
    radiomics_dim = 45     (see data/copd_ph_radiomics.csv)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    SAGEConv, BatchNorm, global_mean_pool, global_max_pool,
)


class HybridGCN(nn.Module):
    """GraphSAGE backbone + radiomics branch + fusion MLP.

    Modes:
        mode='hybrid'       → use graph + radiomics
        mode='gcn_only'     → zero out radiomics branch
        mode='radiomics_only' → zero out graph branch
    """

    def __init__(
        self,
        gcn_in: int = 12,
        gcn_hidden: int = 64,
        radiomics_dim: int = 45,
        out_channels: int = 2,
        num_layers: int = 3,
        dropout: float = 0.3,
        pooling: str = "mean",
        mode: str = "hybrid",
    ) -> None:
        super().__init__()
        if mode not in ("hybrid", "gcn_only", "radiomics_only"):
            raise ValueError(f"unknown mode={mode}")
        self.mode = mode
        self.dropout = dropout

        # --- GCN backbone
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(SAGEConv(gcn_in, gcn_hidden))
        self.bns.append(BatchNorm(gcn_hidden))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(gcn_hidden, gcn_hidden))
            self.bns.append(BatchNorm(gcn_hidden))

        self.pool = global_max_pool if pooling == "max" else global_mean_pool

        # --- Radiomics branch: BN to stabilize heterogeneous feature scales
        self.radiomics_bn = nn.BatchNorm1d(radiomics_dim)

        # --- Fusion
        if mode == "gcn_only":
            fused_dim = gcn_hidden
        elif mode == "radiomics_only":
            fused_dim = radiomics_dim
        else:
            fused_dim = gcn_hidden + radiomics_dim

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, out_channels),
        )

        # Kept for compatibility with existing evaluate_model() in run_cv_full.py,
        # which unpacks (logits, embedding, node_embeddings).
        self.embedding_head = nn.Linear(fused_dim, fused_dim)

    def _gcn_forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        node_emb = x
        graph_emb = self.pool(x, batch)
        return graph_emb, node_emb

    def forward(self, x, edge_index, batch=None, radiomics=None):
        graph_emb, node_emb = self._gcn_forward(x, edge_index, batch)

        if self.mode == "gcn_only":
            fused = graph_emb
        elif self.mode == "radiomics_only":
            if radiomics is None:
                raise ValueError("radiomics tensor required in radiomics_only mode")
            fused = self.radiomics_bn(radiomics)
        else:
            if radiomics is None:
                raise ValueError("radiomics tensor required in hybrid mode")
            rad_emb = self.radiomics_bn(radiomics)
            fused = torch.cat([graph_emb, rad_emb], dim=1)

        logits = self.classifier(fused)
        embedding = self.embedding_head(fused)
        return logits, embedding, node_emb
