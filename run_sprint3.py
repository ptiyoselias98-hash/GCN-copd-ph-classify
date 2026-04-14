"""Sprint 3 — P0 improvements over Sprint 2:
  (1) Focal / class-balanced loss instead of weighted CE
  (2) Youden's J threshold calibration on val per fold
  (3) globals subset control: all (12) / local4 ([bv5_ratio, total_bv5,
      total_branch_count, vessel_tortuosity]) / none (zero vector)

Same 5-fold CV, 3 modes × 2 feat sets. JSON schema extends sprint2: every
fold entry now carries {AUC, Accuracy, ..., threshold, Accuracy_argmax, ...}
so downstream reports can pick Youden-calibrated or argmax metrics.
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

from sklearn.metrics import roc_auc_score, roc_curve

from run_sprint2 import build_dataset_v2, make_loaders as _mk_loaders_base
from run_hybrid import load_labels, load_splits, load_radiomics, full_metrics
from hybrid_gcn import HybridGCN

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("sprint3")

# Index layout of the 12 globals (see enhance_features.py::augment_graph::globals_row)
GLOBAL_NAMES = [
    "fractal_dim", "artery_density", "vein_density", "vein_bv5",
    "vein_branch_count", "bv5_ratio", "artery_vein_vol_ratio", "total_bv5",
    "lung_density_std", "vein_bv10", "total_branch_count", "vessel_tortuosity",
]
LOCAL4_IDX = [5, 7, 10, 11]  # bv5_ratio, total_bv5, total_branch_count, vessel_tortuosity


# ---------------- Losses ----------------
class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin 2017)."""

    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha  # (C,) class weights or None
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def build_cb_weights(n0: int, n1: int, beta: float = 0.9999) -> torch.Tensor:
    """Class-balanced weights (Cui 2019)."""
    samples = np.array([max(n0, 1), max(n1, 1)], dtype=np.float64)
    eff = (1.0 - np.power(beta, samples)) / (1.0 - beta)
    w = 1.0 / eff
    w = w / w.sum() * 2.0  # normalize so mean weight = 1
    return torch.tensor(w, dtype=torch.float32)


def build_loss(loss_type: str, n0: int, n1: int, device) -> nn.Module:
    if loss_type == "focal":
        w = build_cb_weights(n0, n1).to(device)
        return FocalLoss(alpha=w, gamma=2.0)
    if loss_type == "cb":
        w = build_cb_weights(n0, n1).to(device)
        return nn.CrossEntropyLoss(weight=w)
    # weighted_ce — sprint2 default
    w = torch.tensor([max(n1, 1) / max(n0, 1), 1.0], dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=w)


# ---------------- Threshold calibration ----------------
def youden_threshold(y_true: list[int], y_score: list[float]) -> float:
    if len(set(y_true)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, y_score)
    j = tpr - fpr
    best = int(np.argmax(j))
    t = float(thr[best])
    # roc_curve can return thr > 1 at the first step; clamp to [0,1]
    return min(max(t, 1e-4), 1.0 - 1e-4)


# ---------------- globals subset ----------------
def apply_globals_keep(dataset: list[dict], keep: str) -> int:
    """Slice/zero graph.global_features in place. Returns resulting dim."""
    if not dataset:
        return 0
    g0 = dataset[0]["graph"]
    if not hasattr(g0, "global_features") or g0.global_features is None:
        return 0
    if keep == "all":
        return int(g0.global_features.size(-1))
    if keep == "none":
        for e in dataset:
            g = e["graph"]
            g.global_features = torch.zeros_like(g.global_features)
        return int(g0.global_features.size(-1))  # dim unchanged, zeros inside
    if keep == "local4":
        idx = torch.tensor(LOCAL4_IDX, dtype=torch.long)
        for e in dataset:
            g = e["graph"]
            g.global_features = g.global_features.index_select(-1, idx)
        return len(LOCAL4_IDX)
    raise ValueError(f"unknown globals_keep={keep}")


# ---------------- Training ----------------
def _train_fold(mode, radiomics_dim, gcn_in, tl, vl, trl, device, epochs, lr, wd,
                loss_type, patience=40, global_dim=0):
    model = HybridGCN(gcn_in=gcn_in, gcn_hidden=64, radiomics_dim=radiomics_dim,
                      num_layers=3, dropout=0.3, mode=mode,
                      global_dim=global_dim).to(device)
    n1 = int((trl == 1).sum()); n0 = int((trl == 0).sum())
    criterion = build_loss(loss_type, n0, n1, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_auc = -1.0; best_state = None; bad = 0
    for _ in range(epochs):
        model.train()
        for batch in tl:
            batch = batch.to(device)
            logits, _, _ = model(batch.x, batch.edge_index, batch.batch,
                                 getattr(batch, "radiomics", None),
                                 global_features=getattr(batch, "global_features", None))
            loss = criterion(logits, batch.y.view(-1))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); yt, pr = [], []
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
            best_auc = auc; bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    # final val pass: compute probs + two metric sets
    model.eval(); yt, pr = [], []
    with torch.no_grad():
        for batch in vl:
            batch = batch.to(device)
            logits, _, _ = model(batch.x, batch.edge_index, batch.batch,
                                 getattr(batch, "radiomics", None),
                                 global_features=getattr(batch, "global_features", None))
            prob = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            yt += batch.y.view(-1).cpu().numpy().tolist()
            pr += prob.tolist()
    y_true = np.array(yt); y_score = np.array(pr)
    thr = youden_threshold(yt, pr)
    y_pred_y = (y_score >= thr).astype(int)
    y_pred_a = (y_score >= 0.5).astype(int)
    m_y = full_metrics(y_true, y_pred_y, y_score)
    m_a = full_metrics(y_true, y_pred_a, y_score)
    # primary metrics = Youden; argmax metrics prefixed
    out = dict(m_y)
    out["threshold"] = float(thr)
    for k, v in m_a.items():
        out[f"{k}_argmax"] = v
    return out, yt, pr


def run_all(dataset, folds, device, args, radiomics_dim, gcn_in, global_dim):
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    out = {}
    for mode in modes:
        logger.info("=== mode=%s  gcn_in=%d  global_dim=%d  loss=%s ===",
                    mode, gcn_in, global_dim, args.loss)
        fold_metrics, yts, prs = [], [], []
        for k, (tr, va) in enumerate(folds, 1):
            tl, vl, trl = _mk_loaders_base(dataset, tr, va, args.batch_size)
            m, yt, pr = _train_fold(mode, radiomics_dim, gcn_in, tl, vl, trl,
                                    device, args.epochs, args.lr, args.wd,
                                    args.loss, global_dim=global_dim)
            logger.info("  fold %d thr=%.3f AUC=%.4f Acc=%.4f F1=%.4f Sens=%.4f Spec=%.4f",
                        k, m["threshold"], m["AUC"], m["Accuracy"], m["F1"],
                        m["Sensitivity"], m["Specificity"])
            fold_metrics.append(m); yts += yt; prs += pr

        keys = [k for k in fold_metrics[0].keys() if k != "threshold"]
        mean = {k: float(np.mean([fm[k] for fm in fold_metrics])) for k in keys}
        std = {k: float(np.std([fm[k] for fm in fold_metrics])) for k in keys}
        pooled = float(roc_auc_score(yts, prs)) if len(set(yts)) > 1 else 0.0
        out[mode] = {"folds": fold_metrics, "mean": mean, "std": std,
                     "pooled_AUC": pooled}
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", default="./cache")
    p.add_argument("--radiomics", default="./data/copd_ph_radiomics.csv")
    p.add_argument("--labels", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--output_dir", default="./outputs/sprint3")
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
    args = p.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("[config] loss=%s globals_keep=%s modes=%s feat_sets=%s epochs=%d",
                args.loss, args.globals_keep, args.modes, args.feature_sets,
                args.epochs)

    labels = load_labels(args.labels)
    folds = load_splits(args.splits)
    rad_df, feat_cols = load_radiomics(args.radiomics)

    all_results = {"_config": {
        "loss": args.loss, "globals_keep": args.globals_keep,
        "epochs": args.epochs, "lr": args.lr, "wd": args.wd,
        "batch_size": args.batch_size, "seed": args.seed,
        "local4_indices": LOCAL4_IDX,
        "local4_names": [GLOBAL_NAMES[i] for i in LOCAL4_IDX],
    }}

    for fs in [s.strip() for s in args.feature_sets.split(",") if s.strip()]:
        enhanced = fs == "enhanced"
        dataset = build_dataset_v2(args.cache_dir, labels, rad_df, feat_cols,
                                   enhanced=enhanced)
        id_set = {e["patient_id"] for e in dataset}
        filtered = [([i for i in tr if i in id_set], [i for i in va if i in id_set])
                    for tr, va in folds]
        gcn_in = dataset[0]["graph"].x.size(1) if dataset else (13 if enhanced else 12)

        if enhanced:
            global_dim = apply_globals_keep(dataset, args.globals_keep)
            logger.info("[%s] globals_keep=%s → global_dim=%d",
                        fs, args.globals_keep, global_dim)
        else:
            global_dim = 0

        all_results[fs] = run_all(dataset, filtered, device, args,
                                  len(feat_cols), gcn_in, global_dim)

    with (out_dir / "sprint3_results.json").open("w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("saved %s", out_dir / "sprint3_results.json")

    # summary
    print(f"\n{'='*120}")
    print(f"SPRINT 3 SUMMARY (loss={args.loss} globals_keep={args.globals_keep})")
    print("="*120)
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
