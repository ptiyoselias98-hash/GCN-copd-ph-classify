"""R13 — CORAL/MMD non-GRL deconfounder.

Drop-in alternative to run_sprint6_v2_grl_fix.py: replaces the gradient-
reversal adversarial head with a deep-CORAL feature-distribution alignment
penalty between nonPH-contrast and nonPH-plain-scan samples in each batch.
Optional --use_mmd flag swaps to a multi-kernel MMD penalty instead.

CORAL loss:  L_CORAL = (1/(4 d^2)) * || Cov(z|c=0) - Cov(z|c=1) ||_F^2
where z is the encoder embedding (z_proj), c is is_contrast, computed
within label==0 (nonPH) per batch. λ * L_CORAL added to encoder loss.

MMD loss (multi-kernel RBF):
    k(x,y) = sum_i exp(-||x-y||^2 / (2 sigma_i^2))
    L_MMD = mean_x mean_x' k - 2 mean_x mean_y k + mean_y mean_y' k

Re-uses run_sprint6_v2_grl_fix.py infrastructure by importing module-level
helpers; only train_fold + main are overridden.

Usage (remote):
    python run_sprint6_v2_coral.py --arm arm_a --coral_lambda 1.0 \
        --epochs 120 --seed 42 --output_dir outputs/sprint6_arm_a_coral_l1.0_s42
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse all helpers from the fixed-GRL training script
from run_sprint6_v2_grl_fix import (  # type: ignore
    apply_augmentation, build_cb_weights, build_dataset_expanded, FocalLoss,
    full_metrics, load_labels, load_pa_ao, load_radiomics, load_splits_folder,
    youden_threshold, _first_col, _get,
)
from sklearn.metrics import roc_auc_score, roc_curve
from torch_geometric.loader import DataLoader

from hybrid_gcn import HybridGCN
from enhance_features import GLOBAL_FEATURE_DIM, BASELINE_IN_DIM
from utils.graph_builder import normalize_graph_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("sprint6_coral")


def coral_loss(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    """Deep-CORAL feature alignment loss between two sets of embeddings."""
    if z_a.size(0) < 2 or z_b.size(0) < 2:
        return z_a.new_zeros(())
    d = z_a.size(1)
    z_a_c = z_a - z_a.mean(dim=0, keepdim=True)
    z_b_c = z_b - z_b.mean(dim=0, keepdim=True)
    cov_a = (z_a_c.T @ z_a_c) / max(z_a.size(0) - 1, 1)
    cov_b = (z_b_c.T @ z_b_c) / max(z_b.size(0) - 1, 1)
    diff = cov_a - cov_b
    return (diff * diff).sum() / (4.0 * d * d)


def mmd_loss(z_a: torch.Tensor, z_b: torch.Tensor,
             sigmas=(0.5, 1.0, 2.0, 4.0, 8.0)) -> torch.Tensor:
    """Multi-kernel RBF MMD^2 between two sample sets."""
    if z_a.size(0) < 2 or z_b.size(0) < 2:
        return z_a.new_zeros(())
    def _pairwise_sq(a, b):
        return ((a.unsqueeze(1) - b.unsqueeze(0)) ** 2).sum(dim=2)
    daa = _pairwise_sq(z_a, z_a)
    dbb = _pairwise_sq(z_b, z_b)
    dab = _pairwise_sq(z_a, z_b)
    total = z_a.new_zeros(())
    for s in sigmas:
        kaa = torch.exp(-daa / (2.0 * s * s)).mean()
        kbb = torch.exp(-dbb / (2.0 * s * s)).mean()
        kab = torch.exp(-dab / (2.0 * s * s)).mean()
        total = total + (kaa + kbb - 2.0 * kab)
    return total / float(len(sigmas))


def train_fold_coral(mode, radiomics_dim, gcn_in, tl, vl, trl, device,
                     epochs, lr, wd, patience=40, global_dim=0,
                     residual=False, jk="none", coral_lambda=0.0,
                     use_mmd=False):
    model = HybridGCN(gcn_in=gcn_in, gcn_hidden=64, radiomics_dim=radiomics_dim,
                      num_layers=3, dropout=0.3, mode=mode,
                      global_dim=global_dim, residual=residual, jk=jk).to(device)
    n1 = int((trl == 1).sum()); n0 = int((trl == 0).sum())
    w = build_cb_weights(n0, n1).to(device)
    criterion = FocalLoss(alpha=w, gamma=2.0)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_auc = -1.0
    best_state = None
    bad = 0
    align_log: list[float] = []

    for epoch in range(epochs):
        model.train()
        epoch_align = []
        for batch in tl:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch,
                        getattr(batch, "radiomics", None),
                        global_features=getattr(batch, "global_features", None))
            if len(out) == 4:
                logits, z_proj, _, _ = out
            elif len(out) == 3:
                logits, z_proj, _ = out
            else:
                logits = out[0]; z_proj = torch.zeros(logits.size(0), 1, device=logits.device)
            loss = criterion(logits, batch.y.view(-1))
            if coral_lambda > 0 and hasattr(batch, "is_contrast"):
                is_c = batch.is_contrast.view(-1).to(logits.device)
                y = batch.y.view(-1).to(logits.device)
                nph_mask = (y == 0)
                if nph_mask.sum() >= 4:
                    z_nph = z_proj[nph_mask]
                    c_nph = is_c[nph_mask]
                    z_contrast = z_nph[c_nph == 1]
                    z_plain = z_nph[c_nph == 0]
                    if z_contrast.size(0) >= 2 and z_plain.size(0) >= 2:
                        if use_mmd:
                            align_value = mmd_loss(z_contrast, z_plain)
                        else:
                            align_value = coral_loss(z_contrast, z_plain)
                        loss = loss + coral_lambda * align_value
                        epoch_align.append(float(align_value.detach().cpu()))
            opt.zero_grad()
            loss.backward()
            opt.step()

        if epoch_align:
            mean_align = sum(epoch_align) / len(epoch_align)
            align_log.append(mean_align)
            logger.info("    [%s] epoch=%d mean_align_loss=%.5f n_batches=%d",
                        "mmd" if use_mmd else "coral",
                        epoch, mean_align, len(epoch_align))

        model.eval()
        yt, pr = [], []
        with torch.no_grad():
            for batch in vl:
                batch = batch.to(device)
                logits, _, _ = model(batch.x, batch.edge_index, batch.batch,
                                     getattr(batch, "radiomics", None),
                                     global_features=getattr(batch, "global_features", None))
                prob = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                yt += batch.y.view(-1).cpu().numpy().tolist()
                pr += prob.tolist()
        auc = roc_auc_score(yt, pr) if len(set(yt)) > 1 else 0.0
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                logger.info("Early stop at epoch %d (best=%.4f)", epoch, best_auc)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc, align_log


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default="arm_a")
    p.add_argument("--cache_dir", default="cache_v2_tri_flat")
    p.add_argument("--radiomics", default="data/copd_ph_radiomics.csv")
    p.add_argument("--labels", default="data/labels_expanded_282.csv")
    p.add_argument("--splits", default="data/splits_expanded_282")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--augment", default="edge_drop,feature_mask")
    p.add_argument("--coral_lambda", type=float, default=0.0,
                   help="Weight on CORAL/MMD feature-alignment loss")
    p.add_argument("--use_mmd", action="store_true",
                   help="Use multi-kernel MMD instead of CORAL")
    p.add_argument("--protocol_csv", default="data/case_protocol.csv")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = out_dir / "embeddings"; emb_dir.mkdir(exist_ok=True)

    labels = load_labels(args.labels)
    rad_df, feat_cols = load_radiomics(args.radiomics)
    folds = load_splits_folder(args.splits)

    # Load protocol mapping for is_contrast attachment
    proto_map: dict[str, int] = {}
    with open(args.protocol_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            proto_map[row["case_id"].strip()] = 1 if row.get("protocol", "").lower() == "contrast" else 0

    augment = [a for a in args.augment.split(",") if a]

    # Single-pass dataset build (matches _grl_fix.py); then split per fold
    full_dataset = build_dataset_expanded(
        args.cache_dir, labels, rad_df, feat_cols,
        enhanced=False, require_radiomics=False)
    if not full_dataset:
        raise SystemExit("empty dataset")
    # Attach is_contrast onto each .graph (PyG Data object)
    for entry in full_dataset:
        cid = entry.get("patient_id")
        g = entry.get("graph")
        if g is None or cid is None:
            continue
        ic = proto_map.get(str(cid), 0)
        g.is_contrast = torch.tensor([ic], dtype=torch.long)
    by_id = {entry["patient_id"]: entry for entry in full_dataset}

    fold_metrics = []
    for k, (tr_ids, va_ids) in enumerate(folds, start=1):
        tr_entries = [by_id[c] for c in tr_ids if c in by_id]
        va_entries = [by_id[c] for c in va_ids if c in by_id]
        if not tr_entries or not va_entries:
            logger.warning("fold %d: empty after filter (tr=%d va=%d)", k, len(tr_entries), len(va_entries))
            continue
        tr_graphs = [e["graph"] for e in tr_entries]
        va_graphs = [e["graph"] for e in va_entries]
        tl = DataLoader(tr_graphs, batch_size=args.batch_size, shuffle=True)
        vl = DataLoader(va_graphs, batch_size=args.batch_size)
        trl = np.array([int(e["label"]) for e in tr_entries])
        gcn_in = tr_graphs[0].x.size(1)

        model, best_auc, align_log = train_fold_coral(
            "gcn_only", radiomics_dim=len(feat_cols), gcn_in=gcn_in,
            tl=tl, vl=vl, trl=trl, device=device,
            epochs=args.epochs, lr=args.lr, wd=args.wd,
            coral_lambda=args.coral_lambda, use_mmd=args.use_mmd)

        # Dump embeddings for downstream within-nonPH protocol probe
        model.eval()
        embs, ys, cids = [], [], []
        with torch.no_grad():
            for batch in vl:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch,
                            getattr(batch, "radiomics", None),
                            global_features=getattr(batch, "global_features", None))
                if len(out) >= 2:
                    z = out[1].cpu().numpy()
                else:
                    z = out[0].cpu().numpy()
                embs.append(z)
                ys += batch.y.view(-1).cpu().numpy().tolist()
        embs = np.concatenate(embs, axis=0)
        np.savez(emb_dir / f"emb_gcn_only_rep1_fold{k}.npz",
                 embeddings=embs, y_true=np.array(ys, int))

        fold_metrics.append({"fold": k, "AUC": float(best_auc),
                             "align_log_last": align_log[-1] if align_log else None})

    summary = {"_config": vars(args),
               "method": "MMD" if args.use_mmd else "CORAL",
               "fold_metrics": fold_metrics}
    (out_dir / "sprint6_results.json").write_text(json.dumps(summary, indent=2))
    print(f"Done. Best AUCs: {[m['AUC'] for m in fold_metrics]}")


if __name__ == "__main__":
    main()
