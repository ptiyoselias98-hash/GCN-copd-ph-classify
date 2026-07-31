"""
Step 3: Train GCN — HybridGCN with stratified 5-fold CV.
==========================================================

Modes:
  - radiomics_only : MLP on  radiomics vector (no graph)
  - gcn_only       : GraphSAGE on vessel graph (no radiomics)
  - hybrid         : GraphSAGE ⊕ radiomics → MLP classifier

Training config (Sprint 5 style):
  - Focal loss (γ=2) with class-balanced α
  - Youden's J threshold per fold (primary metric), argmax (0.5) for comparison
  - Node-drop augmentation (p=0.1) for gcn/hybrid modes
  - mPAP regression auxiliary head (λ=0.1) if mPAP available
  - Global features (12 commercial scalars) appended to pooled GCN embedding

Reports 6 metrics: AUC, Accuracy, Precision, Sensitivity, F1, Specificity.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Add repo to path
REPO_ROOT = Path(__file__).resolve().parent.parent
REPRO_DIR = REPO_ROOT / "reproduction"
CACHE_DIR = REPRO_DIR / "cache"
OUT_DIR   = REPRO_DIR / "outputs"

sys.path.insert(0, str(REPO_ROOT / "copdph-gcn-repo"))

from hybrid_gcn import HybridGCN
from enhance_features import augment_graph, GLOBAL_FEATURE_DIM


# ═══════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════

def full_metrics(y_true, y_pred, y_score=None):
    """Compute 6 standard metrics."""
    return {
        "AUC": float(roc_auc_score(y_true, y_score)) if y_score is not None and len(set(y_true)) > 1 else 0.0,
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Specificity": _specificity(y_true, y_pred),
    }


def _specificity(y_true, y_pred):
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def youden_threshold(y_true, y_score):
    """Find threshold maximizing sensitivity + specificity - 1."""
    if len(y_true) < 2 or len(set(y_true)) < 2:
        return 0.5
    best_thr, best_j = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 91):
        y_pred = (np.array(y_score) >= thr).astype(int)
        sens = recall_score(y_true, y_pred, zero_division=0)
        spec = _specificity(np.array(y_true), y_pred)
        j = sens + spec - 1.0
        if j > best_j:
            best_j, best_thr = j, thr
    return best_thr


# ═══════════════════════════════════════════════════════════════════════
# Loss
# ═══════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """Focal loss for binary classification."""
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # per-class weight tensor
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return focal.mean()
        return focal.sum()


def build_loss(loss_type, n0, n1, device):
    """Build loss function based on class counts."""
    if loss_type == "focal":
        total = n0 + n1
        # Class-balanced alpha
        alpha = torch.tensor([total / (2.0 * max(n0, 1)), total / (2.0 * max(n1, 1))],
                             dtype=torch.float, device=device)
        return FocalLoss(gamma=2.0, alpha=alpha)
    elif loss_type == "cb":
        # Class-balanced loss
        beta = 0.9999
        effective_n0 = (1 - beta ** n0) / (1 - beta) if n0 > 0 else 0
        effective_n1 = (1 - beta ** n1) / (1 - beta) if n1 > 0 else 0
        w0 = 1.0 / max(effective_n0, 1.0)
        w1 = 1.0 / max(effective_n1, 1.0)
        ws = torch.tensor([w0, w1], dtype=torch.float, device=device)
        ws = ws / ws.sum() * 2.0
        return nn.CrossEntropyLoss(weight=ws)
    else:
        # weighted cross-entropy
        total = n0 + n1
        ws = torch.tensor([total / (2.0 * max(n0, 1)), total / (2.0 * max(n1, 1))],
                         dtype=torch.float, device=device)
        return nn.CrossEntropyLoss(weight=ws)


# ═══════════════════════════════════════════════════════════════════════
# Node-drop augmentation
# ═══════════════════════════════════════════════════════════════════════

def drop_leaf_nodes(edge_index, num_nodes, p=0.1):
    """Randomly drop degree-1 leaf nodes with probability p."""
    if p <= 0 or edge_index.numel() == 0:
        return torch.ones(num_nodes, dtype=torch.bool, device=edge_index.device)

    deg = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    deg.scatter_add_(0, edge_index[0],
                     torch.ones(edge_index.size(1), dtype=torch.long,
                                device=edge_index.device))
    leaves = (deg == 1)
    drop = torch.zeros_like(leaves)
    drop[leaves] = (torch.rand(int(leaves.sum()),
                               device=edge_index.device) < p)
    return ~drop


def apply_node_mask(batch, keep_mask):
    """Subset batch by node mask, remapping edge indices."""
    if keep_mask.all():
        return batch
    old_to_new = -torch.ones(keep_mask.size(0), dtype=torch.long,
                             device=keep_mask.device)
    old_to_new[keep_mask] = torch.arange(int(keep_mask.sum()),
                                         device=keep_mask.device)
    ei = batch.edge_index
    edge_keep = keep_mask[ei[0]] & keep_mask[ei[1]]
    new_ei = old_to_new[ei[:, edge_keep]]
    batch.x = batch.x[keep_mask]
    batch.edge_index = new_ei
    if hasattr(batch, "batch") and batch.batch is not None:
        batch.batch = batch.batch[keep_mask]
    if hasattr(batch, "pos") and batch.pos is not None:
        batch.pos = batch.pos[keep_mask]
    return batch


# ═══════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════

def load_dataset(cache_dir, labels_df, radiomics_df, enhanced=False):
    """Load cached graphs and attach radiomics."""
    dataset = []
    cache_path = Path(cache_dir)

    rad_cols = [c for c in radiomics_df.columns if c != "name"]
    # Build patient_id → radiomics_vector mapping
    rad_map = {}
    for pid in radiomics_df.index:
        try:
            row = radiomics_df.loc[pid][rad_cols].values.astype(np.float32)
            # Replace NaN with 0
            row = np.nan_to_num(row, nan=0.0)
            rad_map[str(pid)] = row
        except Exception:
            rad_map[str(pid)] = np.zeros(len(rad_cols), dtype=np.float32)

    for _, row in labels_df.iterrows():
        pid = str(row["patient_id"])
        pkl_path = cache_path / f"{pid}.pkl"

        if not pkl_path.exists():
            continue

        try:
            with open(pkl_path, "rb") as f:
                entry = pickle.load(f)
        except Exception:
            continue

        if entry is None:
            continue

        graph = entry.get("graph")
        if graph is None:
            continue

        # Choose baseline or enhanced graph
        if enhanced and "graph_enhanced" in entry and entry["graph_enhanced"] is not None:
            graph = entry["graph_enhanced"]

        # Clone to avoid modifying cache
        graph = graph.clone() if hasattr(graph, 'clone') else graph
        graph.y = torch.tensor([int(row["label"])], dtype=torch.long)

        # Attach radiomics vector
        rad_vec = rad_map.get(pid, np.zeros(len(rad_cols), dtype=np.float32))
        graph.radiomics = torch.tensor(rad_vec, dtype=torch.float32)

        # Attach mPAP
        mpap_val = row.get("mPAP")
        try:
            mpap_val = float(mpap_val)
        except (ValueError, TypeError):
            mpap_val = float("nan")
        graph.mpap = torch.tensor([mpap_val], dtype=torch.float32)

        dataset.append({
            "patient_id": pid,
            "graph": graph,
            "label": int(row["label"]),
            "mpap": mpap_val,
        })

    return dataset


# ═══════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════

def train_one_fold(
    train_dataset, val_dataset, mode, radiomics_dim, gcn_in,
    device, epochs=300, lr=1e-3, wd=5e-4, batch_size=8,
    loss_type="focal", global_dim=0, fusion="concat",
    node_drop_p=0.1, mpap_aux_weight=0.1, patience=40,
):
    """Train one fold. Returns (metrics dict, y_true list, y_score list)."""

    train_labels = [d["label"] for d in train_dataset]
    n0 = train_labels.count(0)
    n1 = train_labels.count(1)

    # Build model
    core = HybridGCN(
        gcn_in=gcn_in, gcn_hidden=64, radiomics_dim=radiomics_dim,
        num_layers=3, dropout=0.3, mode=mode,
        global_dim=global_dim if mode != "radiomics_only" else 0,
        fusion=fusion,
    ).to(device)

    criterion = build_loss(loss_type, n0, n1, device)
    mse = nn.MSELoss()
    opt = torch.optim.Adam(core.parameters(), lr=lr, weight_decay=wd)

    # DataLoaders
    train_graphs = [d["graph"] for d in train_dataset]
    val_graphs   = [d["graph"] for d in val_dataset]

    # Ensure graphs are not empty
    train_graphs = [g for g in train_graphs if hasattr(g, 'x') and g.x.size(0) > 0]
    val_graphs   = [g for g in val_graphs if hasattr(g, 'x') and g.x.size(0) > 0]

    if len(train_graphs) < 2 or len(val_graphs) < 2:
        empty_metrics = {"AUC": 0.5, "Accuracy": 0.5, "Precision": 0.0,
                         "Sensitivity": 0.0, "F1": 0.0, "Specificity": 0.0}
        return empty_metrics, [], []

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True,
                              drop_last=(len(train_graphs) > batch_size))
    val_loader   = DataLoader(val_graphs, batch_size=batch_size)

    best_auc, best_state, bad = -1.0, None, 0

    for epoch in range(epochs):
        core.train()
        for batch in train_loader:
            batch = batch.to(device)

            # Node-drop augmentation
            if node_drop_p > 0 and mode != "radiomics_only":
                keep = drop_leaf_nodes(batch.edge_index, batch.x.size(0), node_drop_p)
                batch = apply_node_mask(batch, keep)

            gf = getattr(batch, "global_features", None)
            rad = getattr(batch, "radiomics", None)

            logits, emb, _ = core(
                batch.x, batch.edge_index, batch.batch,
                radiomics=rad, global_features=gf,
            )

            loss = criterion(logits, batch.y.view(-1))

            # mPAP auxiliary loss
            if mpap_aux_weight > 0 and mode != "radiomics_only" and hasattr(batch, "mpap"):
                # Predict mPAP from embedding
                mpap_pred = nn.Linear(emb.size(1), 1).to(device)(emb).squeeze(-1)
                m_true = batch.mpap.view(-1).float().to(device)
                m_mask = ~torch.isnan(m_true)
                if m_mask.any():
                    loss = loss + mpap_aux_weight * mse(mpap_pred[m_mask], m_true[m_mask])

            opt.zero_grad()
            loss.backward()
            opt.step()

        # Validation
        core.eval()
        yt, pr = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                gf = getattr(batch, "global_features", None)
                rad = getattr(batch, "radiomics", None)
                logits, _, _ = core(
                    batch.x, batch.edge_index, batch.batch,
                    radiomics=rad, global_features=gf,
                )
                prob = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                yt += batch.y.view(-1).cpu().numpy().tolist()
                pr += prob.tolist()

        auc = roc_auc_score(yt, pr) if len(set(yt)) > 1 else 0.0
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    # Load best model and compute final metrics
    if best_state is not None:
        core.load_state_dict(best_state)

    core.eval()
    yt, pr = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            gf = getattr(batch, "global_features", None)
            rad = getattr(batch, "radiomics", None)
            logits, _, _ = core(
                batch.x, batch.edge_index, batch.batch,
                radiomics=rad, global_features=gf,
            )
            prob = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            yt += batch.y.view(-1).cpu().numpy().tolist()
            pr += prob.tolist()

    y_true = np.array(yt)
    y_score = np.array(pr)

    thr = youden_threshold(yt, pr)
    y_pred_y = (y_score >= thr).astype(int)
    y_pred_a = (y_score >= 0.5).astype(int)

    m_y = full_metrics(y_true, y_pred_y, y_score)
    m_a = full_metrics(y_true, y_pred_a, y_score)

    out = dict(m_y)
    out["threshold"] = float(thr)
    for k, v in m_a.items():
        out[f"{k}_argmax"] = v

    return out, yt, pr


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GCN COPD-PH Classification")
    parser.add_argument("--cache_dir", default=str(CACHE_DIR))
    parser.add_argument("--labels", default=str(REPRO_DIR / "labels.csv"))
    parser.add_argument("--radiomics", default=str(REPRO_DIR / "radiomics.csv"))
    parser.add_argument("--output_dir", default=str(OUT_DIR / "main_experiment"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--modes", default="radiomics_only,gcn_only,hybrid")
    parser.add_argument("--loss", default="focal",
                        choices=["focal", "cb", "weighted_ce"])
    parser.add_argument("--fusion", default="concat", choices=["concat", "gated"])
    parser.add_argument("--node_drop_p", type=float, default=0.1)
    parser.add_argument("--mpap_aux_weight", type=float, default=0.1)
    parser.add_argument("--enhanced", action="store_true", default=True,
                        help="Use enhanced (13D node + 12D global) features")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    print(f"\nLoading labels from {args.labels}...")
    labels_df = pd.read_csv(args.labels)
    print(f"  {len(labels_df)} patients: "
          f"PH={(labels_df['label']==1).sum()}, "
          f"nonPH={(labels_df['label']==0).sum()}")

    print(f"Loading radiomics from {args.radiomics}...")
    radiomics_df = pd.read_csv(args.radiomics, index_col=0)
    radiomics_dim = len([c for c in radiomics_df.columns if c != "name"])
    print(f"  {radiomics_df.shape[0]} patients × {radiomics_dim} features")

    # ── Build dataset ──
    print(f"\nBuilding dataset (enhanced={args.enhanced})...")
    t0 = time.time()
    dataset = load_dataset(args.cache_dir, labels_df, radiomics_df,
                           enhanced=args.enhanced)
    print(f"  Loaded {len(dataset)} valid graphs in {time.time()-t0:.1f}s")

    if len(dataset) < 10:
        print("ERROR: Too few valid graphs! Check cache directory.")
        return 1

    # ── Create stratified folds ──
    labels_arr = np.array([d["label"] for d in dataset])
    pids = np.array([d["patient_id"] for d in dataset])
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True,
                          random_state=args.seed)

    folds = []
    for train_idx, val_idx in skf.split(np.arange(len(dataset)), labels_arr):
        folds.append((
            pids[train_idx].tolist(),
            pids[val_idx].tolist(),
        ))

    pid_to_entry = {d["patient_id"]: d for d in dataset}

    # ── Determine input dimensions ──
    sample_graph = dataset[0]["graph"]
    gcn_in = int(sample_graph.x.size(1))
    has_global = hasattr(sample_graph, "global_features") and sample_graph.global_features is not None
    global_dim = GLOBAL_FEATURE_DIM if (has_global and args.enhanced) else 0
    print(f"  gcn_in={gcn_in}, radiomics_dim={radiomics_dim}, global_dim={global_dim}")

    # ── Run all modes ──
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    all_results = {
        "_config": {
            "loss": args.loss, "fusion": args.fusion,
            "node_drop_p": args.node_drop_p,
            "mpap_aux_weight": args.mpap_aux_weight,
            "n_folds": args.n_folds, "epochs": args.epochs,
            "lr": args.lr, "wd": args.wd, "batch_size": args.batch_size,
            "seed": args.seed, "gcn_in": gcn_in,
            "radiomics_dim": radiomics_dim, "global_dim": global_dim,
            "n_patients": len(dataset),
            "n_ph": int((labels_arr == 1).sum()),
            "n_nonph": int((labels_arr == 0).sum()),
        },
        "feature_set": "enhanced" if args.enhanced else "baseline",
        "results": {},
    }

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"MODE: {mode} (gcn_in={gcn_in}, global_dim={global_dim})")
        print(f"{'='*60}")

        fold_metrics = []
        all_yt, all_pr = [], []

        for fold_i, (train_pids, val_pids) in enumerate(folds, 1):
            train_data = [pid_to_entry[pid] for pid in train_pids
                         if pid in pid_to_entry]
            val_data   = [pid_to_entry[pid] for pid in val_pids
                         if pid in pid_to_entry]

            if len(train_data) < 2 or len(val_data) < 1:
                print(f"  Fold {fold_i}: SKIP (insufficient data)")
                continue

            t1 = time.time()
            metrics, yt, pr = train_one_fold(
                train_data, val_data,
                mode=mode,
                radiomics_dim=radiomics_dim,
                gcn_in=gcn_in,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                wd=args.wd,
                batch_size=args.batch_size,
                loss_type=args.loss,
                global_dim=global_dim if mode != "radiomics_only" else 0,
                fusion=args.fusion,
                node_drop_p=args.node_drop_p,
                mpap_aux_weight=args.mpap_aux_weight,
            )
            fold_metrics.append(metrics)
            all_yt += yt
            all_pr += pr

            print(f"  Fold {fold_i}/{args.n_folds} ({time.time()-t1:.0f}s): "
                  f"thr={metrics.get('threshold', 0.5):.3f} "
                  f"AUC={metrics['AUC']:.4f} Acc={metrics['Accuracy']:.4f} "
                  f"F1={metrics['F1']:.4f} Sens={metrics['Sensitivity']:.4f} "
                  f"Spec={metrics['Specificity']:.4f} Prec={metrics['Precision']:.4f}")

        # Aggregate
        keys = ["AUC", "Accuracy", "Precision", "Sensitivity", "F1", "Specificity"]
        mean_vals = {k: float(np.mean([fm[k] for fm in fold_metrics]))
                     for k in keys if k in fold_metrics[0]}
        std_vals  = {k: float(np.std([fm[k] for fm in fold_metrics]))
                     for k in keys if k in fold_metrics[0]}
        pooled_auc = float(roc_auc_score(all_yt, all_pr)) if len(set(all_yt)) > 1 else 0.0

        result_entry = {
            "folds": fold_metrics,
            "mean": mean_vals,
            "std": std_vals,
            "pooled_AUC": pooled_auc,
        }
        all_results["results"][mode] = result_entry

        print(f"\n  {mode} SUMMARY:")
        for k in keys:
            print(f"    {k:15s}: {mean_vals[k]:.4f} ± {std_vals[k]:.4f}")
        print(f"    {'pooled AUC':15s}: {pooled_auc:.4f}")

    # ── Save results ──
    results_path = out_dir / "classification_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_path}")

    # ── Final comparison table ──
    print(f"\n{'='*100}")
    print("FINAL COMPARISON")
    print(f"{'='*100}")
    header = f"{'Mode':<20} {'AUC':>8} {'Acc':>8} {'Sens':>8} {'Spec':>8} {'F1':>8} {'Prec':>8}  | pooled_AUC"
    print(header)
    print("-" * 100)
    for mode in modes:
        if mode not in all_results["results"]:
            continue
        r = all_results["results"][mode]
        m, p = r["mean"], r.get("pooled_AUC", 0)
        print(f"{mode:<20} {m['AUC']:8.4f} {m['Accuracy']:8.4f} "
              f"{m['Sensitivity']:8.4f} {m['Specificity']:8.4f} "
              f"{m['F1']:8.4f} {m['Precision']:8.4f}  | {p:.4f}")
    print(f"{'='*100}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
