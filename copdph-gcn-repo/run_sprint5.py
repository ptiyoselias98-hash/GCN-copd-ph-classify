"""Sprint 5 — GCN improvements on top of Sprint 3 (focal_local4 baseline):

  (1) mPAP-stratified splits (loaded via --splits_json, a single JSON file)
  (2) Node-drop augmentation during training (drop leaf nodes with prob p)
  (3) mPAP regression auxiliary head (multi-task: CE + lambda * MSE(mpap))

All three toggles are independently enabled by flags; default is all on.

Output JSON schema matches sprint3 (adds top-level `_config.sprint5` block).

Usage (remote, launched by _remote_sprint5.py):
  python run_sprint5.py --cache_dir ./cache --radiomics ./data/copd_ph_radiomics.csv \
    --labels LABELS --splits_json ./data/splits_mpap_stratified.json \
    --mpap_lookup ./data/mpap_lookup.json \
    --output_dir ./outputs/sprint5_full \
    --epochs 300 --batch_size 8 --lr 1e-3 \
    --loss focal --globals_keep local4 \
    --node_drop_p 0.10 --mpap_aux_weight 0.1
"""
from __future__ import annotations

import argparse
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

from sklearn.metrics import roc_auc_score

from run_sprint2 import build_dataset_v2, make_loaders as _mk_loaders_base
from run_hybrid import load_labels, load_radiomics, full_metrics
from run_sprint3 import (
    apply_globals_keep, build_loss, youden_threshold, LOCAL4_IDX, GLOBAL_NAMES,
)
from hybrid_gcn import HybridGCN

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("sprint5")


# ---------------- Custom splits loader ----------------
def load_splits_json(path: str) -> list[tuple[list[str], list[str]]]:
    """Load mPAP-stratified splits. Schema: list of {train: [...], val: [...]}."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [(d["train"], d["val"]) for d in raw]


# ---------------- mPAP auxiliary head wrapper ----------------
class HybridGCNWithMPAP(nn.Module):
    """Wraps HybridGCN, adds a regression head that predicts mPAP from the
    fused embedding. The embedding is extracted via the same embedding_head
    as the classifier input."""

    def __init__(self, core: HybridGCN) -> None:
        super().__init__()
        self.core = core
        fused_dim = core.embedding_head.in_features
        self.mpap_head = nn.Sequential(
            nn.Linear(fused_dim, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x, edge_index, batch=None, radiomics=None,
                global_features=None):
        logits, emb, node_emb = self.core(
            x, edge_index, batch=batch, radiomics=radiomics,
            global_features=global_features,
        )
        mpap_pred = self.mpap_head(emb).squeeze(-1)
        return logits, emb, node_emb, mpap_pred


# ---------------- Node-drop augmentation ----------------
def drop_leaf_nodes(edge_index: torch.Tensor, num_nodes: int,
                    p: float = 0.1) -> torch.Tensor:
    """Return a boolean mask over nodes; True = keep. Drops degree-1 leaves
    with probability p. Never drops high-degree nodes (preserves topology)."""
    if p <= 0 or edge_index.numel() == 0:
        return torch.ones(num_nodes, dtype=torch.bool, device=edge_index.device)
    deg = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    deg.scatter_add_(0, edge_index[0],
                     torch.ones(edge_index.size(1), dtype=torch.long,
                                device=edge_index.device))
    leaves = (deg == 1)
    drop = torch.zeros_like(leaves)
    drop[leaves] = (torch.rand(int(leaves.sum()), device=edge_index.device) < p)
    return ~drop


def apply_node_mask(batch, keep_mask: torch.Tensor):
    """Subset batch.x / edge_index / batch / (optional) attrs using node mask.
    Preserves edges only if both endpoints are kept; remaps indices."""
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
    batch.batch = batch.batch[keep_mask]
    return batch


# ---------------- Training ----------------
def _train_fold_s5(mode, radiomics_dim, gcn_in, tl, vl, trl, device,
                   epochs, lr, wd, loss_type, patience=40,
                   global_dim=0, fusion="concat",
                   node_drop_p=0.0, mpap_aux_weight=0.0):
    core = HybridGCN(gcn_in=gcn_in, gcn_hidden=64, radiomics_dim=radiomics_dim,
                     num_layers=3, dropout=0.3, mode=mode,
                     global_dim=global_dim, fusion=fusion).to(device)
    use_mpap = mpap_aux_weight > 0
    model = HybridGCNWithMPAP(core).to(device) if use_mpap else core

    n1 = int((trl == 1).sum()); n0 = int((trl == 0).sum())
    criterion = build_loss(loss_type, n0, n1, device)
    mse = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_auc, best_state, bad = -1.0, None, 0

    for _ in range(epochs):
        model.train()
        for batch in tl:
            batch = batch.to(device)
            if node_drop_p > 0:
                keep = drop_leaf_nodes(batch.edge_index, batch.x.size(0),
                                       node_drop_p)
                batch = apply_node_mask(batch, keep)
            gf = getattr(batch, "global_features", None)
            if use_mpap:
                logits, _, _, mpap_pred = model(
                    batch.x, batch.edge_index, batch.batch,
                    getattr(batch, "radiomics", None), global_features=gf)
            else:
                logits, _, _ = model(
                    batch.x, batch.edge_index, batch.batch,
                    getattr(batch, "radiomics", None), global_features=gf)
            loss = criterion(logits, batch.y.view(-1))
            if use_mpap and hasattr(batch, "mpap"):
                m_true = batch.mpap.view(-1).float().to(device)
                m_mask = ~torch.isnan(m_true)
                if m_mask.any():
                    loss = loss + mpap_aux_weight * mse(
                        mpap_pred[m_mask], m_true[m_mask])
            opt.zero_grad(); loss.backward(); opt.step()

        # val
        model.eval(); yt, pr = [], []
        with torch.no_grad():
            for batch in vl:
                batch = batch.to(device)
                gf = getattr(batch, "global_features", None)
                out = model(batch.x, batch.edge_index, batch.batch,
                            getattr(batch, "radiomics", None),
                            global_features=gf)
                logits = out[0]
                prob = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                yt += batch.y.view(-1).cpu().numpy().tolist()
                pr += prob.tolist()
        auc = roc_auc_score(yt, pr) if len(set(yt)) > 1 else 0.0
        if auc > best_auc:
            best_auc = auc; bad = 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # final metrics
    model.eval(); yt, pr = [], []
    with torch.no_grad():
        for batch in vl:
            batch = batch.to(device)
            gf = getattr(batch, "global_features", None)
            out = model(batch.x, batch.edge_index, batch.batch,
                        getattr(batch, "radiomics", None),
                        global_features=gf)
            logits = out[0]
            prob = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            yt += batch.y.view(-1).cpu().numpy().tolist()
            pr += prob.tolist()
    y_true = np.array(yt); y_score = np.array(pr)
    thr = youden_threshold(yt, pr)
    y_pred_y = (y_score >= thr).astype(int)
    y_pred_a = (y_score >= 0.5).astype(int)
    m_y = full_metrics(y_true, y_pred_y, y_score)
    m_a = full_metrics(y_true, y_pred_a, y_score)
    out = dict(m_y); out["threshold"] = float(thr)
    for k, v in m_a.items():
        out[f"{k}_argmax"] = v
    return out, yt, pr


def run_all(dataset, folds, device, args, radiomics_dim, gcn_in, global_dim):
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    out = {}
    for mode in modes:
        logger.info("=== mode=%s gcn_in=%d global_dim=%d loss=%s "
                    "node_drop=%.2f mpap_aux=%.2f ===",
                    mode, gcn_in, global_dim, args.loss,
                    args.node_drop_p, args.mpap_aux_weight)
        fold_metrics, yts, prs = [], [], []
        for k, (tr, va) in enumerate(folds, 1):
            tl, vl, trl = _mk_loaders_base(dataset, tr, va, args.batch_size)
            m, yt, pr = _train_fold_s5(
                mode, radiomics_dim, gcn_in, tl, vl, trl, device,
                args.epochs, args.lr, args.wd, args.loss,
                global_dim=global_dim, fusion=args.fusion,
                node_drop_p=args.node_drop_p,
                mpap_aux_weight=args.mpap_aux_weight,
            )
            logger.info("  fold %d thr=%.3f AUC=%.4f Acc=%.4f F1=%.4f "
                        "Sens=%.4f Spec=%.4f", k, m["threshold"], m["AUC"],
                        m["Accuracy"], m["F1"], m["Sensitivity"], m["Specificity"])
            fold_metrics.append(m); yts += yt; prs += pr
        keys = [k for k in fold_metrics[0].keys() if k != "threshold"]
        mean = {k: float(np.mean([fm[k] for fm in fold_metrics])) for k in keys}
        std = {k: float(np.std([fm[k] for fm in fold_metrics])) for k in keys}
        pooled = float(roc_auc_score(yts, prs)) if len(set(yts)) > 1 else 0.0
        out[mode] = {"folds": fold_metrics, "mean": mean, "std": std,
                     "pooled_AUC": pooled}
    return out


def attach_mpap(dataset: list[dict], mpap_lookup: dict[str, float]) -> int:
    """Attach `data.mpap` (scalar per graph) from a {pid: mpap} dict."""
    miss = 0
    for e in dataset:
        pid = e["patient_id"]
        v = mpap_lookup.get(pid)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            v = float("nan"); miss += 1
        e["graph"].mpap = torch.tensor([float(v)], dtype=torch.float32)
    return miss


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", default="./cache")
    p.add_argument("--radiomics", default="./data/copd_ph_radiomics.csv")
    p.add_argument("--labels", required=True)
    p.add_argument("--splits_json", required=True,
                   help="path to mPAP-stratified splits JSON")
    p.add_argument("--mpap_lookup", default="",
                   help="optional: JSON {pid: mpap} for regression aux loss")
    p.add_argument("--output_dir", default="./outputs/sprint5")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--modes", default="radiomics_only,gcn_only,hybrid")
    p.add_argument("--feature_sets", default="baseline,enhanced")
    p.add_argument("--loss", default="focal", choices=["focal", "cb", "weighted_ce"])
    p.add_argument("--globals_keep", default="local4",
                   choices=["all", "local4", "none"])
    p.add_argument("--fusion", default="concat", choices=["concat", "gated"])
    p.add_argument("--node_drop_p", type=float, default=0.10)
    p.add_argument("--mpap_aux_weight", type=float, default=0.10)
    args = p.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("[sprint5-config] loss=%s globals_keep=%s node_drop=%.3f "
                "mpap_aux=%.3f epochs=%d", args.loss, args.globals_keep,
                args.node_drop_p, args.mpap_aux_weight, args.epochs)

    labels = load_labels(args.labels)
    folds = load_splits_json(args.splits_json)
    rad_df, feat_cols = load_radiomics(args.radiomics)

    mpap_lookup: dict[str, float] = {}
    if args.mpap_lookup and Path(args.mpap_lookup).exists():
        with open(args.mpap_lookup, "r", encoding="utf-8") as f:
            mpap_lookup = {k: float(v) if v is not None else float("nan")
                           for k, v in json.load(f).items()}
        logger.info("loaded mpap_lookup: %d cases", len(mpap_lookup))

    all_results = {"_config": {
        "sprint5": True,
        "loss": args.loss, "globals_keep": args.globals_keep,
        "node_drop_p": args.node_drop_p,
        "mpap_aux_weight": args.mpap_aux_weight,
        "splits_json": args.splits_json,
        "epochs": args.epochs, "lr": args.lr, "wd": args.wd,
        "batch_size": args.batch_size, "seed": args.seed,
        "local4_indices": LOCAL4_IDX,
        "local4_names": [GLOBAL_NAMES[i] for i in LOCAL4_IDX],
    }}

    for fs in [s.strip() for s in args.feature_sets.split(",") if s.strip()]:
        enhanced = fs == "enhanced"
        dataset = build_dataset_v2(args.cache_dir, labels, rad_df, feat_cols,
                                   enhanced=enhanced)
        if mpap_lookup:
            miss = attach_mpap(dataset, mpap_lookup)
            logger.info("[%s] attached mpap: miss=%d / %d",
                        fs, miss, len(dataset))
        id_set = {e["patient_id"] for e in dataset}
        filtered = [([i for i in tr if i in id_set],
                     [i for i in va if i in id_set]) for tr, va in folds]
        gcn_in = dataset[0]["graph"].x.size(1) if dataset else (13 if enhanced else 12)
        if enhanced:
            global_dim = apply_globals_keep(dataset, args.globals_keep)
        else:
            global_dim = 0
        all_results[fs] = run_all(dataset, filtered, device, args,
                                  len(feat_cols), gcn_in, global_dim)

    with (out_dir / "sprint5_results.json").open("w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("saved %s", out_dir / "sprint5_results.json")

    print(f"\n{'=' * 120}")
    print(f"SPRINT 5 SUMMARY (node_drop={args.node_drop_p} "
          f"mpap_aux={args.mpap_aux_weight})")
    print("=" * 120)
    keys = ["AUC", "Accuracy", "Precision", "Sensitivity", "F1", "Specificity"]
    for fs, modes in all_results.items():
        if fs == "_config":
            continue
        for mode, r in modes.items():
            m, s = r["mean"], r["std"]
            print(f"{fs:<10} {mode:<16} "
                  + " ".join(f"{m[k]:.4f}±{s[k]:.4f}" for k in keys)
                  + f"  pooled={r['pooled_AUC']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
