"""
models.py — Tri-Structure GCN with cross-structure attention fusion.

Architecture:
    Artery graph  → GCN_A → z_A (D)
    Vein graph    → GCN_V → z_V (D)     → Cross-structure attention → z_fused (D)
    Airway graph  → GCN_W → z_W (D)

    z_fused → Classification head → PH / non-PH
    z_fused → (same vector) → clustering / phenotype discovery

The attention mechanism learns per-patient importance weights over the three
structures. In a patient where arterial pruning dominates, attention will
concentrate on z_A.  In one where airway remodeling is the leading change,
z_W gets higher weight.  This is biologically meaningful: PH is heterogeneous,
and the attention distribution itself becomes a phenotype descriptor.

Design decisions:
  - Each structure gets its OWN encoder (no weight sharing), because artery,
    vein, and airway have different topology statistics. Shared weights would
    force one set of message-passing filters to handle all three — the exact
    problem with the current unified graph.
  - Attention is computed in a shared key-query space so that structures can
    "see" each other. The query is a learned global context vector (not one
    of the structures), ensuring no structure gets structurally privileged.
  - The embedding dimension is kept small (D=64) to control parameter count
    on n=106 cases.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

try:
    from torch_geometric.nn import (
        SAGEConv, GINConv, global_mean_pool, global_add_pool, GlobalAttention,
    )
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class StructureEncoder(nn.Module):
    """
    Independent GCN encoder for one anatomical structure.

    Uses GraphSAGE convolutions (inductive, handles varying graph sizes).
    Returns a graph-level embedding of dimension `hidden`.
    """

    def __init__(self, in_dim: int, hidden: int = 64, n_layers: int = 3,
                 dropout: float = 0.3, pool: str = "mean"):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        self.pool_mode = pool

        # LayerNorm instead of BatchNorm: airway pseudo-graph is 1 node;
        # BN on (1, hidden) errors at train time. LayerNorm is batch-independent.
        self.convs.append(SAGEConv(in_dim, hidden))
        self.bns.append(nn.LayerNorm(hidden))
        for _ in range(n_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
            self.bns.append(nn.LayerNorm(hidden))

        if pool == "attn":
            # Learnable gate over nodes; node_i weight = softmax_i(gate_nn(h_i))
            gate_nn = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, 1),
            )
            self.attn_pool = GlobalAttention(gate_nn=gate_nn)
            self.pool_fn = None
        elif pool == "add":
            self.pool_fn = global_add_pool
            self.attn_pool = None
        else:
            self.pool_fn = global_mean_pool
            self.attn_pool = None

    def forward(self, x, edge_index, batch=None):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        node_emb = x
        if self.attn_pool is not None:
            if batch is None:
                batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            graph_emb = self.attn_pool(x, batch)
        else:
            graph_emb = self.pool_fn(x, batch)
        return graph_emb, node_emb


class CrossStructureAttention(nn.Module):
    """
    Learnable attention over K structure embeddings.

    Given z_1, z_2, ..., z_K  (each shape (B, D)):
      - Project each through a shared key network: k_i = W_k z_i
      - Compute attention with a learned query q: a_i = softmax(q · k_i / sqrt(d))
      - Output: z_fused = sum_i a_i * z_i

    The attention weights a_i are interpretable: they tell us which structure
    the model considers most informative for each patient.

    We also return the raw attention weights for phenotype analysis.
    """

    def __init__(self, embed_dim: int, n_structures: int = 3):
        super().__init__()
        self.key_net = nn.Linear(embed_dim, embed_dim)
        # Learned global query — not tied to any structure
        self.query = nn.Parameter(torch.randn(1, embed_dim) * 0.02)
        self.scale = embed_dim ** 0.5
        self.n_structures = n_structures

    def forward(self, structure_embeddings: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            structure_embeddings: list of K tensors, each (B, D)

        Returns:
            fused: (B, D) attention-weighted combination
            attn_weights: (B, K) attention distribution per patient
        """
        B = structure_embeddings[0].shape[0]
        K = len(structure_embeddings)

        # Stack: (B, K, D)
        stacked = torch.stack(structure_embeddings, dim=1)

        # Keys: (B, K, D)
        keys = self.key_net(stacked)

        # Query: (1, D) → (B, 1, D) via broadcast
        q = self.query.expand(B, -1).unsqueeze(1)  # (B, 1, D)

        # Attention scores: (B, 1, K) = (B,1,D) @ (B,D,K)
        scores = torch.bmm(q, keys.transpose(1, 2)) / self.scale  # (B, 1, K)
        attn = F.softmax(scores, dim=-1)  # (B, 1, K)

        # Weighted sum: (B, 1, K) @ (B, K, D) → (B, 1, D) → (B, D)
        fused = torch.bmm(attn, stacked).squeeze(1)  # (B, D)
        attn_weights = attn.squeeze(1)  # (B, K)

        return fused, attn_weights


class TriStructureGCN(nn.Module):
    """
    Full model: three independent GCN encoders → cross-structure attention →
    shared embedding → classification head.

    The shared embedding (z_fused) is THE output for both classification and
    clustering.  No separate embedding heads — one representation serves all
    downstream tasks.

    Optional mPAP regression auxiliary head for semi-supervised signal.
    """

    def __init__(
        self,
        in_dim_artery: int = 15,
        in_dim_vein: int = 15,
        in_dim_airway: int = 15,
        hidden: int = 64,
        n_layers: int = 3,
        dropout: float = 0.3,
        n_classes: int = 2,
        use_mpap_aux: bool = False,
        pool: str = "mean",
    ):
        super().__init__()

        self.enc_artery = StructureEncoder(in_dim_artery, hidden, n_layers, dropout, pool=pool)
        self.enc_vein = StructureEncoder(in_dim_vein, hidden, n_layers, dropout, pool=pool)
        self.enc_airway = StructureEncoder(in_dim_airway, hidden, n_layers, dropout, pool=pool)

        self.attention = CrossStructureAttention(hidden, n_structures=3)

        # Classification head: from shared embedding
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

        # Optional mPAP regression auxiliary
        self.use_mpap_aux = use_mpap_aux
        if use_mpap_aux:
            self.mpap_head = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, 1),
            )

    def forward(
        self,
        artery_data: Data,
        vein_data: Data,
        airway_data: Data,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with:
            logits:       (B, n_classes) classification logits
            embedding:    (B, hidden) THE shared embedding for clustering
            attn_weights: (B, 3) attention over [artery, vein, airway]
            mpap_pred:    (B, 1) if use_mpap_aux, else None
        """
        z_a, _ = self.enc_artery(artery_data.x, artery_data.edge_index,
                                  getattr(artery_data, 'batch', None))
        z_v, _ = self.enc_vein(vein_data.x, vein_data.edge_index,
                                getattr(vein_data, 'batch', None))
        z_w, _ = self.enc_airway(airway_data.x, airway_data.edge_index,
                                  getattr(airway_data, 'batch', None))

        z_fused, attn = self.attention([z_a, z_v, z_w])

        logits = self.classifier(z_fused)

        out = {
            "logits": logits,
            "embedding": z_fused,          # THIS is the shared embedding
            "attn_weights": attn,           # (B, 3) — interpretable
            "z_artery": z_a,
            "z_vein": z_v,
            "z_airway": z_w,
        }

        if self.use_mpap_aux:
            out["mpap_pred"] = self.mpap_head(z_fused)

        return out


class DualStructureGCN(nn.Module):
    """
    Fallback for cases where airway graphs are unavailable.
    Two encoders (artery + vein) with 2-way attention.
    """

    def __init__(
        self,
        in_dim_artery: int = 15,
        in_dim_vein: int = 15,
        hidden: int = 64,
        n_layers: int = 3,
        dropout: float = 0.3,
        n_classes: int = 2,
        use_mpap_aux: bool = False,
    ):
        super().__init__()

        self.enc_artery = StructureEncoder(in_dim_artery, hidden, n_layers, dropout)
        self.enc_vein = StructureEncoder(in_dim_vein, hidden, n_layers, dropout)
        self.attention = CrossStructureAttention(hidden, n_structures=2)

        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

        self.use_mpap_aux = use_mpap_aux
        if use_mpap_aux:
            self.mpap_head = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, 1),
            )

    def forward(self, artery_data, vein_data):
        z_a, _ = self.enc_artery(artery_data.x, artery_data.edge_index,
                                  getattr(artery_data, 'batch', None))
        z_v, _ = self.enc_vein(vein_data.x, vein_data.edge_index,
                                getattr(vein_data, 'batch', None))

        z_fused, attn = self.attention([z_a, z_v])
        logits = self.classifier(z_fused)

        out = {
            "logits": logits,
            "embedding": z_fused,
            "attn_weights": attn,
            "z_artery": z_a,
            "z_vein": z_v,
        }
        if self.use_mpap_aux:
            out["mpap_pred"] = self.mpap_head(z_fused)

        return out
