"""Per-fold saliency visualization for all 5 CV folds.

For each fold k in 1..5:
    - train the enhanced-hybrid GCN on fold-k's training set
    - pick the first PH + first non-PH case from fold-k's val set
    - compute input-gradient saliency for those two cases
    - render a 3D vessel tree pair, coloured by saliency

Outputs in <output_dir>/:
    saliency_fold1.png ... saliency_fold5.png
    saliency_all_folds.png   (5×2 grid combined view)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from torch_geometric.data import Batch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hybrid_gcn import HybridGCN
from run_hybrid import (
    attach_radiomics, load_labels, load_radiomics, load_splits, make_loaders,
)
from utils.graph_builder import normalize_graph_features
from visualize import build_enhanced_dataset, render_tree

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("viz_folds")


def train_hybrid(dataset, train_ids, val_ids, radiomics_dim, device,
                 epochs=120, lr=1e-3, wd=5e-4):
    tl, vl, tr_labels = make_loaders(dataset, train_ids, val_ids,
                                     batch_size=8, radiomics_dim=radiomics_dim)
    first = next(iter(tl))
    node_dim = first.x.shape[1]
    model = HybridGCN(gcn_in=node_dim, gcn_hidden=64, radiomics_dim=radiomics_dim,
                      out_channels=2, num_layers=3, dropout=0.3, mode="hybrid").to(device)
    n1 = int((tr_labels == 1).sum()); n0 = int((tr_labels == 0).sum())
    w = torch.tensor([max(n1, 1) / max(n0, 1), 1.0], dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    from sklearn.metrics import roc_auc_score
    best_auc = -1; best_state = None
    for epoch in range(epochs):
        model.train()
        for batch in tl:
            batch = batch.to(device)
            logits, _, _ = model(batch.x, batch.edge_index, batch.batch,
                                 getattr(batch, "radiomics", None))
            loss = F.cross_entropy(logits, batch.y.view(-1), weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval(); yt, pr = [], []
            with torch.no_grad():
                for batch in vl:
                    batch = batch.to(device)
                    logits, _, _ = model(batch.x, batch.edge_index, batch.batch,
                                         getattr(batch, "radiomics", None))
                    prob = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    yt += batch.y.view(-1).cpu().numpy().tolist()
                    pr += prob.tolist()
            auc = roc_auc_score(yt, pr) if len(set(yt)) > 1 else 0.0
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc


def compute_radiomics_standardizer(dataset, train_ids):
    id2entry = {e["patient_id"]: e for e in dataset}
    tr = [id2entry[i] for i in train_ids if i in id2entry]
    rad_tr = np.stack([e["radiomics"] for e in tr])
    rad_tr = np.nan_to_num(rad_tr, nan=0.0, posinf=0.0, neginf=0.0)
    mu = rad_tr.mean(axis=0); sd = rad_tr.std(axis=0) + 1e-6

    def std(v):
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        return ((v - mu) / sd).astype(np.float32)
    return std


def saliency_for_entry(model, entry, std_fn, device):
    g = entry["graph"].clone()
    g = normalize_graph_features([g])[0]
    rad_std = std_fn(entry["radiomics"])
    g = attach_radiomics(g, rad_std)
    batch = Batch.from_data_list([g]).to(device)
    batch.x.requires_grad_(True)
    model.eval()
    logits, _, _ = model(batch.x, batch.edge_index, batch.batch,
                         getattr(batch, "radiomics", None))
    prob_ph = F.softmax(logits, dim=1)[:, 1].sum()
    prob_ph.backward()
    grad = batch.x.grad.detach().abs().sum(dim=1).cpu().numpy()
    return grad


def pick_val_samples(dataset, val_ids):
    id2e = {e["patient_id"]: e for e in dataset}
    val = [id2e[i] for i in val_ids if i in id2e]
    ph = next((e for e in val if e["label"] == 1), None)
    nonph = next((e for e in val if e["label"] == 0), None)
    return ph, nonph


def plot_pair(ax_nonph, ax_ph, nonph, ph, fold_k, val_auc):
    for ax, entry, name in [(ax_nonph, nonph, "non-PH"), (ax_ph, ph, "PH")]:
        g = entry["graph"]
        pos = g.pos.cpu().numpy()
        ei = g.edge_index.cpu().numpy()
        sal = entry["_saliency"]
        render_tree(ax, pos, ei, sal,
                    f"Fold {fold_k} · {name}: {entry['patient_id'][:36]}",
                    cmap="magma", colorbar_label="|∇ p(PH)|")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", default="./cache")
    p.add_argument("--radiomics", default="./data/copd_ph_radiomics.csv")
    p.add_argument("--labels", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--output_dir", default="outputs/viz_folds")
    p.add_argument("--epochs", type=int, default=120)
    args = p.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", device)

    labels = load_labels(args.labels)
    rad_df, feat_cols = load_radiomics(args.radiomics)
    dataset, _ = build_enhanced_dataset(args.cache_dir, labels, rad_df, feat_cols)
    logger.info("dataset: %d patients", len(dataset))

    folds = load_splits(args.splits)
    rad_dim = len(feat_cols)

    fold_results = []
    for k, (tr_ids, va_ids) in enumerate(folds, start=1):
        logger.info("=== Fold %d ===", k)
        ph, nonph = pick_val_samples(dataset, va_ids)
        if ph is None or nonph is None:
            logger.warning("fold %d missing PH or non-PH in val — skipping", k)
            continue
        model, auc = train_hybrid(dataset, tr_ids, va_ids, rad_dim, device,
                                  epochs=args.epochs)
        logger.info("fold %d best val AUC=%.4f", k, auc)
        std_fn = compute_radiomics_standardizer(dataset, tr_ids)
        for entry in (ph, nonph):
            entry["_saliency"] = saliency_for_entry(model, entry, std_fn, device)
        fold_results.append((k, auc, ph, nonph))

        # per-fold figure
        fig = plt.figure(figsize=(13, 6))
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        plot_pair(ax1, ax2, nonph, ph, k, auc)
        fig.suptitle(f"Fold {k} — enhanced-hybrid GCN saliency  (val AUC={auc:.3f})",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = out / f"saliency_fold{k}.png"
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        logger.info("wrote %s", path)

    # combined 5-row grid
    n = len(fold_results)
    if n:
        fig = plt.figure(figsize=(13, 5.2 * n))
        for i, (k, auc, ph, nonph) in enumerate(fold_results):
            ax1 = fig.add_subplot(n, 2, 2 * i + 1, projection="3d")
            ax2 = fig.add_subplot(n, 2, 2 * i + 2, projection="3d")
            plot_pair(ax1, ax2, nonph, ph, k, auc)
        fig.suptitle("Per-fold enhanced-hybrid GCN saliency (5-fold CV)",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = out / "saliency_all_folds.png"
        plt.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        logger.info("wrote %s", path)

    logger.info("done — %d folds rendered", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
