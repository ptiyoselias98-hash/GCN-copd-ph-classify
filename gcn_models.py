"""
Graph Neural Network models for pulmonary vascular classification.

Three architectures:
  - GCN: Standard graph convolution (Kipf & Welling 2017)
  - GraphSAGE: Inductive learning with neighbor sampling
  - GAT: Graph attention network with multi-head attention

All models output:
  - Graph-level embedding (for classification/clustering)
  - Node-level embeddings (for identifying affected regions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import (
        GCNConv, SAGEConv, GATConv,
        global_mean_pool, global_max_pool, global_add_pool,
        BatchNorm
    )
    from torch_geometric.nn import AttentionalAggregation
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("[WARNING] torch_geometric not installed. Using fallback implementations.")


# ============================================================
# Attention-based graph pooling
# ============================================================

class AttentionPooling(nn.Module):
    """Learnable attention pooling for graph-level readout."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.gate_nn = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, 1)
        )

    def forward(self, x, batch):
        gate = self.gate_nn(x)  # (N, 1)
        gate = torch.softmax(gate, dim=0)  # Normalize
        # Weighted sum per graph
        weighted = x * gate
        if batch is not None:
            return global_add_pool(weighted, batch)
        return weighted.sum(dim=0, keepdim=True)


# ============================================================
# Model: GCN
# ============================================================

class VascularGCN(nn.Module):
    """
    Standard GCN for vascular graph classification.

    Architecture:
      Input → [GCNConv + BN + ReLU + Dropout] × L → Pool → MLP → Output
    """

    def __init__(
        self,
        in_channels: int = 12,
        hidden_channels: int = 64,
        out_channels: int = 2,
        num_layers: int = 3,
        dropout: float = 0.3,
        pooling: str = 'mean'
    ):
        super().__init__()
        assert HAS_PYG, "torch_geometric required for VascularGCN"

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout

        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(BatchNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))

        # Pooling
        if pooling == 'attention':
            self.pool = AttentionPooling(hidden_channels)
        elif pooling == 'max':
            self.pool = global_max_pool
        else:
            self.pool = global_mean_pool

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )

        # Embedding head (for analysis / clustering)
        self.embedding_head = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, batch=None):
        # Message passing
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        node_embeddings = x

        # Graph-level readout
        if isinstance(self.pool, AttentionPooling):
            graph_emb = self.pool(x, batch)
        else:
            graph_emb = self.pool(x, batch)

        # Classification
        logits = self.classifier(graph_emb)
        embedding = self.embedding_head(graph_emb)

        return logits, embedding, node_embeddings


# ============================================================
# Model: GraphSAGE
# ============================================================

class VascularSAGE(nn.Module):
    """
    GraphSAGE for inductive vascular graph classification.

    Better for generalizing across patients with different graph sizes.
    """

    def __init__(
        self,
        in_channels: int = 12,
        hidden_channels: int = 64,
        out_channels: int = 2,
        num_layers: int = 3,
        dropout: float = 0.3,
        pooling: str = 'mean'
    ):
        super().__init__()
        assert HAS_PYG, "torch_geometric required for VascularSAGE"

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(BatchNorm(hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))

        if pooling == 'attention':
            self.pool = AttentionPooling(hidden_channels)
        elif pooling == 'max':
            self.pool = global_max_pool
        else:
            self.pool = global_mean_pool

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        self.embedding_head = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, batch=None):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        node_embeddings = x

        if isinstance(self.pool, AttentionPooling):
            graph_emb = self.pool(x, batch)
        else:
            graph_emb = self.pool(x, batch)

        logits = self.classifier(graph_emb)
        embedding = self.embedding_head(graph_emb)

        return logits, embedding, node_embeddings


# ============================================================
# Model: GAT
# ============================================================

class VascularGAT(nn.Module):
    """
    Graph Attention Network for vascular classification.

    Multi-head attention allows learning which neighboring vessels
    are most important for each node's representation.
    """

    def __init__(
        self,
        in_channels: int = 12,
        hidden_channels: int = 64,
        out_channels: int = 2,
        num_layers: int = 3,
        dropout: float = 0.3,
        heads: int = 4,
        pooling: str = 'mean'
    ):
        super().__init__()
        assert HAS_PYG, "torch_geometric required for VascularGAT"

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout

        # First layer: multi-head
        self.convs.append(GATConv(in_channels, hidden_channels // heads,
                                   heads=heads, dropout=dropout))
        self.bns.append(BatchNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels // heads,
                                       heads=heads, dropout=dropout))
            self.bns.append(BatchNorm(hidden_channels))

        # Final conv: single head
        self.convs.append(GATConv(hidden_channels, hidden_channels,
                                   heads=1, concat=False, dropout=dropout))
        self.bns.append(BatchNorm(hidden_channels))

        if pooling == 'attention':
            self.pool = AttentionPooling(hidden_channels)
        elif pooling == 'max':
            self.pool = global_max_pool
        else:
            self.pool = global_mean_pool

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        self.embedding_head = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, batch=None):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        node_embeddings = x

        if isinstance(self.pool, AttentionPooling):
            graph_emb = self.pool(x, batch)
        else:
            graph_emb = self.pool(x, batch)

        logits = self.classifier(graph_emb)
        embedding = self.embedding_head(graph_emb)

        return logits, embedding, node_embeddings


# ============================================================
# Model factory
# ============================================================

def build_model(config: dict) -> nn.Module:
    """Build model from config dict."""
    model_type = config.get('type', 'GraphSAGE')
    params = {
        'in_channels': config.get('in_channels', 12),
        'hidden_channels': config.get('hidden_channels', 64),
        'out_channels': config.get('out_channels', 2),
        'num_layers': config.get('num_layers', 3),
        'dropout': config.get('dropout', 0.3),
        'pooling': config.get('pooling', 'mean'),
    }

    if model_type == 'GCN':
        return VascularGCN(**params)
    elif model_type == 'GAT':
        params['heads'] = config.get('heads', 4)
        return VascularGAT(**params)
    else:  # GraphSAGE
        return VascularSAGE(**params)
