"""
utils/training_plus.py — enhanced Trainer + cross_validate_plus.

Adds the three Sprint 5 v2 tricks on top of the Sprint 5 Task 5 Trainer:
  (1) focal loss (class-balanced alpha, gamma=2)
  (2) node-drop augmentation (p=0.1) during training only
  (3) mPAP regression auxiliary head (weight=0.1)

Also supports **preset splits** via a JSON file of fold assignments
(list of {"train": [case_id,...], "val": [case_id,...]}), used for both the
gold-subset short-term validation and the mPAP-stratified medium-term run.

Drop in next to the existing utils/training.py — does not replace it.
"""
from __future__ import annotations

import json
import logging
import math
import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


# ─────────────────────────── metrics ───────────────────────────
def _metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                          threshold: float = 0.5,
                          preds: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute the six classification metrics at a given probability threshold."""
    if preds is None:
        preds = (y_prob >= threshold).astype(int)
    tn = int(((y_true == 0) & (preds == 0)).sum())
    fp = int(((y_true == 0) & (preds == 1)).sum())
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    m: Dict[str, float] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, preds)) if len(y_true) else 0.0,
        "f1": float(f1_score(y_true, preds, zero_division=0)) if len(y_true) else 0.0,
        "precision": float(precision_score(y_true, preds, zero_division=0)) if len(y_true) else 0.0,
        "recall": float(recall_score(y_true, preds, zero_division=0)) if len(y_true) else 0.0,
        "sensitivity": float(recall_score(y_true, preds, zero_division=0)) if len(y_true) else 0.0,
        "specificity": spec,
    }
    try:
        m["auc"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    except Exception:
        m["auc"] = 0.0
    return m


def _youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return the probability threshold that maximises Youden's J = TPR - FPR.

    Tie-break by preferring thresholds closer to 0.5 (less extreme)."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    best = np.where(j == j.max())[0]
    # tie-break: closest to 0.5
    bi = best[np.argmin(np.abs(thr[best] - 0.5))]
    t = float(thr[bi])
    # roc_curve can emit inf for the first threshold — clamp to [0,1].
    return max(0.0, min(1.0, t))


# ─────────────────────────── losses ───────────────────────────
class FocalLoss(nn.Module):
    """Class-balanced focal loss (Lin et al.).  alpha derived from inverse freq."""

    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("alpha", alpha.float())
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, y, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def _class_balanced_alpha(labels: List[int], num_classes: int = 2,
                          beta: float = 0.9999) -> torch.Tensor:
    """Cui et al. 'class-balanced loss' effective-number weighting."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.clip(counts, 1.0, None)
    eff = 1.0 - np.power(beta, counts)
    w = (1.0 - beta) / eff
    w = w / w.sum() * num_classes
    return torch.tensor(w, dtype=torch.float)


# ─────────────────────────── augmentations ───────────────────────────
def node_drop(graph, p: float = 0.1):
    """Feature-level node drop: zero out a random fraction of nodes' features.

    Safer than topology-level node removal (no edge_index re-indexing, no risk
    of leaving GCN message passing with stale indices). Matches the effect
    typically called "DropNode" in GraphCL-style augmentation.
    """
    if p <= 0.0:
        return graph
    n = int(graph.x.size(0))
    if n < 4:
        return graph
    keep_mask = (torch.rand(n, device=graph.x.device) > p).float().unsqueeze(1)
    new = graph.clone()
    new.x = graph.x * keep_mask
    return new


# ─────────────────────────── trainer ───────────────────────────
class TrainerPlus:
    """Trainer with focal loss + node-drop + mPAP regression aux head."""

    def __init__(
        self,
        model: nn.Module,
        cfg: dict,
        device: str = "cuda",
        *,
        use_focal: bool = False,
        node_drop_p: float = 0.0,
        mpap_aux: bool = False,
        mpap_aux_weight: float = 0.1,
        hidden_dim: int = 128,
    ):
        self.model = model.to(device)
        self.device = device
        self.cfg = cfg
        self.use_focal = use_focal
        self.node_drop_p = node_drop_p
        self.mpap_aux = mpap_aux
        self.mpap_aux_weight = mpap_aux_weight

        # Optional mPAP regression head — lives on the graph embedding z.
        self.mpap_head: Optional[nn.Module] = None
        if mpap_aux:
            self.mpap_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            ).to(device)

        params = list(self.model.parameters())
        if self.mpap_head is not None:
            params += list(self.mpap_head.parameters())
        self.optimizer = torch.optim.AdamW(
            params, lr=cfg.get("lr", 1e-3),
            weight_decay=cfg.get("weight_decay", 5e-4),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, cfg.get("epochs", 200))
        )
        self.cls_loss_fn: Optional[nn.Module] = None

    # -----
    def _forward(self, batch):
        x = batch.x.to(self.device)
        ei = batch.edge_index.to(self.device)
        b = batch.batch.to(self.device) if hasattr(batch, "batch") else None
        nt = getattr(batch, "node_type", None)
        if nt is not None:
            nt = nt.to(self.device)
        rad = getattr(batch, "radiomics", None)
        if rad is not None:
            rad = rad.to(self.device)
            # PyG concatenates radiomics rows along dim=0 as (B*D,) — reshape to (B, D).
            if b is not None and rad.dim() == 1:
                bsz = int(b.max().item()) + 1
                rad = rad.view(bsz, -1)
            elif rad.dim() == 1:
                rad = rad.unsqueeze(0)
        out = self.model(x, ei, b, node_type=nt, radiomics=rad)
        if isinstance(out, tuple):
            if len(out) == 4:
                logits, z, node_emb, attn = out
            elif len(out) == 3:
                logits, z, node_emb = out
                attn = None
            else:
                raise RuntimeError(f"unexpected return arity: {len(out)}")
        else:
            logits, z, node_emb, attn = out, None, None, None
        return logits, z, node_emb, attn

    def _apply_node_drop(self, data_list):
        if self.node_drop_p <= 0.0:
            return data_list
        return [node_drop(g, self.node_drop_p) for g in data_list]

    def train(
        self,
        train_data: List,
        val_data: List,
        *,
        epochs: int = 200,
        batch_size: int = 8,
        save_dir: str = "./outputs",
        mpap_train: Optional[Dict[int, float]] = None,
        mpap_val: Optional[Dict[int, float]] = None,
    ) -> dict:
        """train_data/val_data: list of PyG Data objects (graph). mpap_{train,val}:
        dict index_in_list -> mPAP (float) or NaN/None."""
        from torch_geometric.loader import DataLoader

        os.makedirs(save_dir, exist_ok=True)

        labels = [int(g.y.item()) for g in train_data]
        if self.use_focal:
            alpha = _class_balanced_alpha(labels)
            self.cls_loss_fn = FocalLoss(alpha=alpha, gamma=2.0).to(self.device)
            logger.info("focal loss alpha=%s gamma=2", alpha.tolist())
        else:
            counts = np.bincount(labels, minlength=2).astype(float)
            counts = np.clip(counts, 1.0, None)
            w = counts.sum() / (2 * counts)
            self.cls_loss_fn = nn.CrossEntropyLoss(
                weight=torch.tensor(w, dtype=torch.float).to(self.device)
            )
            logger.info("CE loss weights=%s", w.tolist())

        # Attach mPAP target as `data.mpap` tensor for DataLoader propagation.
        def _attach(gs, lookup):
            if lookup is None:
                return gs
            out = []
            for i, g in enumerate(gs):
                m = lookup.get(i)
                g = g.clone()
                if m is None or (isinstance(m, float) and math.isnan(m)):
                    g.mpap = torch.tensor([float("nan")], dtype=torch.float)
                else:
                    g.mpap = torch.tensor([float(m)], dtype=torch.float)
                out.append(g)
            return out

        if self.mpap_aux:
            train_data = _attach(train_data, mpap_train)
            val_data = _attach(val_data, mpap_val)

        best_auc = -1.0
        best_state = None
        patience = self.cfg.get("patience", 30)
        stale = 0
        history: Dict[str, List[float]] = {
            "train_loss": [], "val_auc": [], "val_f1": [], "val_acc": [],
        }

        for epoch in range(1, epochs + 1):
            self.model.train()
            if self.mpap_head is not None:
                self.mpap_head.train()
            # Apply node-drop augmentation per epoch (new random mask each time).
            aug = self._apply_node_drop(train_data) if self.node_drop_p > 0 else train_data
            loader = DataLoader(aug, batch_size=batch_size, shuffle=True,
                                drop_last=True)

            tot, n = 0.0, 0
            for batch in loader:
                self.optimizer.zero_grad()
                logits, z, _, _ = self._forward(batch)
                y = batch.y.to(self.device).view(-1)
                loss = self.cls_loss_fn(logits, y)

                if self.mpap_aux and hasattr(batch, "mpap") and z is not None:
                    mpap_target = batch.mpap.to(self.device).view(-1)
                    mask = ~torch.isnan(mpap_target)
                    if mask.sum() > 0:
                        pred = self.mpap_head(z).view(-1)
                        mpap_loss = F.smooth_l1_loss(pred[mask], mpap_target[mask])
                        loss = loss + self.mpap_aux_weight * mpap_loss

                loss.backward()
                self.optimizer.step()
                tot += loss.item() * y.size(0)
                n += y.size(0)
            self.scheduler.step()
            train_loss = tot / max(n, 1)

            val = self.evaluate(val_data, batch_size=batch_size)
            history["train_loss"].append(train_loss)
            history["val_auc"].append(val["auc"])
            history["val_f1"].append(val["f1"])
            history["val_acc"].append(val["accuracy"])

            improved = val["auc"] > best_auc
            if improved:
                best_auc = val["auc"]
                best_state = deepcopy(self.model.state_dict())
                stale = 0
                torch.save(best_state, os.path.join(save_dir, "best_model.pt"))
            else:
                stale += 1

            if epoch % 5 == 0 or epoch == 1 or improved:
                logger.info(
                    "epoch %3d | loss=%.4f | val auc=%.4f f1=%.4f acc=%.4f%s",
                    epoch, train_loss, val["auc"], val["f1"], val["accuracy"],
                    "  *best" if improved else "",
                )
            if stale >= patience:
                logger.info("early stopping at epoch %d", epoch)
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        with open(os.path.join(save_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        return history

    @torch.no_grad()
    def evaluate(self, data: List, *, batch_size: int = 8,
                 return_probs: bool = False) -> Dict[str, float]:
        from torch_geometric.loader import DataLoader

        self.model.eval()
        if self.mpap_head is not None:
            self.mpap_head.eval()
        loader = DataLoader(data, batch_size=batch_size)

        probs, preds, ys = [], [], []
        for batch in loader:
            logits, _, _, _ = self._forward(batch)
            y = batch.y.to(self.device).view(-1)
            p = F.softmax(logits, dim=1).detach().cpu().numpy()
            probs.append(p[:, 1])
            preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            ys.extend(y.cpu().numpy().tolist())

        y_true = np.array(ys)
        y_pred = np.array(preds)
        y_prob = np.concatenate(probs) if probs else np.zeros_like(y_true, dtype=float)

        m = _metrics_at_threshold(y_true, y_prob, threshold=0.5, preds=y_pred)
        if return_probs:
            m["_y_true"] = y_true
            m["_y_prob"] = y_prob
        return m


# ─────────────────────────── splits ───────────────────────────
def _bucketize_mpap(v: Optional[float]) -> int:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return 0
    if v < 18:   return 0
    if v < 22:   return 1
    if v < 30:   return 2
    return 3


def _stratified_splits(
    dataset: List[dict],
    *,
    n_folds: int,
    seed: int,
    mpap_lookup: Optional[Dict[str, float]] = None,
) -> List[Tuple[List[int], List[int]]]:
    """Build mPAP-bucket × label stratified k-fold split for a fresh seed."""
    labels = np.array([int(d["label"]) for d in dataset])
    if mpap_lookup is not None:
        buckets = np.array([_bucketize_mpap(mpap_lookup.get(d["patient_id"]))
                            for d in dataset])
        strata = buckets * 2 + labels
    else:
        strata = labels
    unique, counts = np.unique(strata, return_counts=True)
    rare = set(unique[counts < n_folds].tolist())
    if rare:
        strata = np.array([s if s not in rare else int(s % 2) for s in strata])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(np.arange(len(dataset)), strata))


# ─────────────────────────── cross-val ───────────────────────────
def cross_validate_plus(
    dataset: List[dict],
    config: dict,
    *,
    n_folds: int = 5,
    device: str = "cuda",
    splits: Optional[List[dict]] = None,
    mpap_lookup: Optional[Dict[str, float]] = None,
    use_focal: bool = False,
    node_drop_p: float = 0.0,
    mpap_aux: bool = False,
    mpap_aux_weight: float = 0.1,
    use_youden: bool = False,
    repeats: int = 1,
    seed_base: int = 42,
) -> dict:
    """Run k-fold CV, optionally repeated, with preset splits or stratified.

    dataset: list of entries {graph, label, patient_id} (output of load_cache).
    splits:  optional list of {"train": [case_id,...], "val": [case_id,...]}.
             Used only when repeats==1; otherwise fresh stratified splits per repeat.
    repeats: number of distinct random seeds (each gives a new k-fold partition).
    """
    from models import build_model  # shim injected by train_plus.py

    patient_to_idx = {d["patient_id"]: i for i, d in enumerate(dataset)}

    if repeats == 1 and splits:
        logger.info("using preset splits: %d folds", len(splits))
        fold_indices_per_repeat = [[]]
        for s in splits:
            tr = [patient_to_idx[c] for c in s["train"] if c in patient_to_idx]
            va = [patient_to_idx[c] for c in s["val"] if c in patient_to_idx]
            fold_indices_per_repeat[0].append((tr, va))
    else:
        if splits and repeats > 1:
            logger.info("[note] preset splits ignored: repeats=%d > 1 → "
                        "regenerating stratified splits per seed", repeats)
        fold_indices_per_repeat = []
        for r in range(repeats):
            seed = seed_base + r
            fi = _stratified_splits(dataset, n_folds=n_folds, seed=seed,
                                    mpap_lookup=mpap_lookup)
            logger.info("repeat %d/%d  seed=%d  → %d folds",
                        r + 1, repeats, seed, len(fi))
            fold_indices_per_repeat.append(fi)

    batch_size = config["training"].get("batch_size", 8)
    epochs = config["training"].get("epochs", 200)

    fold_metrics: List[dict] = []
    flat_folds = [(r_idx, fold_idx, tr, va)
                  for r_idx, folds in enumerate(fold_indices_per_repeat)
                  for fold_idx, (tr, va) in enumerate(folds)]
    for global_fold_idx, (r_idx, fold, tr_idx, va_idx) in enumerate(flat_folds):
        logger.info("=== repeat %d / fold %d/%d  (train=%d  val=%d) ===",
                    r_idx, fold + 1, len(fold_indices_per_repeat[r_idx]),
                    len(tr_idx), len(va_idx))
        train_data = [dataset[i] for i in tr_idx]
        val_data = [dataset[i] for i in va_idx]

        # mPAP lookup by fold-local index for attaching into Data objects.
        if mpap_aux and mpap_lookup is not None:
            mpap_train = {
                i: mpap_lookup.get(d["patient_id"]) for i, d in enumerate(train_data)
            }
            mpap_val = {
                i: mpap_lookup.get(d["patient_id"]) for i, d in enumerate(val_data)
            }
        else:
            mpap_train = mpap_val = None

        mcfg = dict(config["model"])
        mcfg["in_channels"] = train_data[0]["graph"].x.shape[1]
        mcfg["out_channels"] = 2
        model = build_model(mcfg)
        hidden = mcfg.get("hidden_channels", 128)

        trainer = TrainerPlus(
            model, config["training"], device,
            use_focal=use_focal,
            node_drop_p=node_drop_p,
            mpap_aux=mpap_aux,
            mpap_aux_weight=mpap_aux_weight,
            hidden_dim=hidden,
        )
        save_dir = os.path.join(
            config.get("output_dir", "./outputs"),
            f"repeat_{r_idx}_fold_{fold}"
            if len(fold_indices_per_repeat) > 1 else f"fold_{fold}",
        )
        trainer.train(
            [d["graph"] for d in train_data],
            [d["graph"] for d in val_data],
            epochs=epochs,
            batch_size=batch_size,
            save_dir=save_dir,
            mpap_train=mpap_train,
            mpap_val=mpap_val,
        )
        val_metrics = trainer.evaluate(
            [d["graph"] for d in val_data],
            batch_size=batch_size,
            return_probs=use_youden,
        )
        val_metrics["fold"] = fold
        val_metrics["repeat"] = r_idx

        if use_youden:
            y_true = val_metrics.pop("_y_true")
            y_prob = val_metrics.pop("_y_prob")
            # Preserve argmax-0.5 numbers under *_argmax keys (Sprint 3 convention).
            argmax_keys = ("accuracy", "f1", "precision", "recall",
                           "sensitivity", "specificity", "threshold")
            for k in argmax_keys:
                val_metrics[f"{k}_argmax"] = val_metrics[k]
            t = _youden_threshold(y_true, y_prob)
            youden = _metrics_at_threshold(y_true, y_prob, threshold=t)
            # AUC is threshold-free → already correct, don't overwrite.
            for k in argmax_keys:
                val_metrics[k] = youden[k]
            logger.info("fold %d: Youden threshold=%.3f  sens=%.3f  spec=%.3f",
                        fold, t, val_metrics["sensitivity"], val_metrics["specificity"])
        fold_metrics.append(val_metrics)
        logger.info("fold %d: %s", fold, val_metrics)

    agg = {}
    for k in ("auc", "accuracy", "f1", "precision", "recall",
              "sensitivity", "specificity"):
        vals = [m[k] for m in fold_metrics]
        agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return {
        "folds": fold_metrics,
        "mean": agg,
        "repeats": len(fold_indices_per_repeat),
        "n_folds_per_repeat": len(fold_indices_per_repeat[0]) if fold_indices_per_repeat else 0,
    }


# ─────────────────────────── radiomics-only baseline ───────────────────────────
class RadiomicsMLP(nn.Module):
    """Tiny MLP head used for the radiomics_only ablation."""

    def __init__(self, in_dim: int, hidden: int = 128, out_dim: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _stack_features(entries: List[dict], feat_keys=("vascular", "airway")) -> np.ndarray:
    """Concat vascular+airway scalar dicts in a deterministic order → (N, D) array."""
    rows: List[List[float]] = []
    key_order: Optional[List[str]] = None
    for e in entries:
        f = e.get("features", {}) or {}
        flat: Dict[str, float] = {}
        for k in feat_keys:
            sub = f.get(k, {}) or {}
            for kk, vv in sub.items():
                if isinstance(vv, (int, float)):
                    flat[f"{k}.{kk}"] = float(vv)
        if key_order is None:
            key_order = sorted(flat.keys())
        rows.append([flat.get(k, 0.0) for k in key_order])
    arr = np.asarray(rows, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def cross_validate_radiomics(
    dataset: List[dict],
    config: dict,
    *,
    n_folds: int = 5,
    device: str = "cuda",
    mpap_lookup: Optional[Dict[str, float]] = None,
    use_focal: bool = False,
    use_youden: bool = False,
    repeats: int = 1,
    seed_base: int = 42,
) -> dict:
    """Radiomics-only CV: tiny MLP on the cached vascular+airway scalars."""
    feats = _stack_features(dataset)
    labels = np.array([int(d["label"]) for d in dataset])
    in_dim = feats.shape[1]
    hidden = int(config.get("model", {}).get("hidden_channels", 128))
    epochs = int(config["training"].get("epochs", 200))
    lr = float(config["training"].get("lr", 1e-3))
    wd = float(config["training"].get("weight_decay", 5e-4))
    patience = int(config["training"].get("patience", 30))
    bs = int(config["training"].get("batch_size", 16))

    fold_indices_per_repeat = [
        _stratified_splits(dataset, n_folds=n_folds, seed=seed_base + r,
                           mpap_lookup=mpap_lookup)
        for r in range(repeats)
    ]
    fold_metrics: List[dict] = []
    for r_idx, folds in enumerate(fold_indices_per_repeat):
        for fold, (tr_idx, va_idx) in enumerate(folds):
            x_tr = torch.tensor(feats[tr_idx], dtype=torch.float, device=device)
            x_va = torch.tensor(feats[va_idx], dtype=torch.float, device=device)
            y_tr = torch.tensor(labels[tr_idx], dtype=torch.long, device=device)
            y_va = torch.tensor(labels[va_idx], dtype=torch.long, device=device)

            model = RadiomicsMLP(in_dim, hidden=hidden, out_dim=2).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
            if use_focal:
                alpha = _class_balanced_alpha(y_tr.cpu().tolist())
                loss_fn: nn.Module = FocalLoss(alpha=alpha, gamma=2.0).to(device)
            else:
                cnt = np.bincount(y_tr.cpu().numpy(), minlength=2).astype(float)
                cnt = np.clip(cnt, 1.0, None)
                w = cnt.sum() / (2 * cnt)
                loss_fn = nn.CrossEntropyLoss(
                    weight=torch.tensor(w, dtype=torch.float, device=device))

            best_auc, best_state, stale = -1.0, None, 0
            for ep in range(1, epochs + 1):
                model.train()
                perm = torch.randperm(x_tr.size(0))
                for i in range(0, x_tr.size(0), bs):
                    idx = perm[i:i + bs]
                    if idx.numel() < 2:
                        continue
                    opt.zero_grad()
                    out = model(x_tr[idx])
                    loss = loss_fn(out, y_tr[idx])
                    loss.backward()
                    opt.step()
                sch.step()

                model.eval()
                with torch.no_grad():
                    p = F.softmax(model(x_va), dim=1)[:, 1].cpu().numpy()
                auc = (roc_auc_score(y_va.cpu().numpy(), p)
                       if len(np.unique(y_va.cpu().numpy())) > 1 else 0.0)
                if auc > best_auc:
                    best_auc, best_state, stale = auc, deepcopy(model.state_dict()), 0
                else:
                    stale += 1
                if stale >= patience:
                    break

            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                p = F.softmax(model(x_va), dim=1)[:, 1].cpu().numpy()
            y_true = y_va.cpu().numpy()
            m = _metrics_at_threshold(y_true, p, threshold=0.5)
            m["fold"] = fold
            m["repeat"] = r_idx

            if use_youden:
                argmax_keys = ("accuracy", "f1", "precision", "recall",
                               "sensitivity", "specificity", "threshold")
                for k in argmax_keys:
                    m[f"{k}_argmax"] = m[k]
                t = _youden_threshold(y_true, p)
                yo = _metrics_at_threshold(y_true, p, threshold=t)
                for k in argmax_keys:
                    m[k] = yo[k]
                logger.info("[radiomics] r%d f%d Youden=%.3f sens=%.3f spec=%.3f auc=%.3f",
                            r_idx, fold, t, m["sensitivity"], m["specificity"], m["auc"])
            else:
                logger.info("[radiomics] r%d f%d auc=%.3f sens=%.3f spec=%.3f",
                            r_idx, fold, m["auc"], m["sensitivity"], m["specificity"])
            fold_metrics.append(m)

    agg = {}
    for k in ("auc", "accuracy", "f1", "precision", "recall",
              "sensitivity", "specificity"):
        vals = [fm[k] for fm in fold_metrics]
        agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return {
        "folds": fold_metrics,
        "mean": agg,
        "repeats": len(fold_indices_per_repeat),
        "n_folds_per_repeat": len(fold_indices_per_repeat[0]) if fold_indices_per_repeat else 0,
        "in_dim": in_dim,
    }
