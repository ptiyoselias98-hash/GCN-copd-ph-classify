"""Training loop, evaluation, cross-validation for graph classification."""
from __future__ import annotations

import json
import logging
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
from sklearn.model_selection import StratifiedKFold, train_test_split

logger = logging.getLogger(__name__)


def split_dataset(
    dataset: List[dict],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Stratified train/val/test split."""
    labels = np.array([d["label"] for d in dataset])
    idx = np.arange(len(dataset))

    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_ratio, random_state=seed, stratify=labels
    )
    rel_val = val_ratio / max(1.0 - test_ratio, 1e-6)
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=rel_val,
        random_state=seed,
        stratify=labels[idx_trainval],
    )

    return (
        [dataset[i] for i in idx_train],
        [dataset[i] for i in idx_val],
        [dataset[i] for i in idx_test],
    )


def _compute_class_weights(labels: List[int], num_classes: int = 2) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    inv = counts.sum() / (num_classes * counts)
    return torch.tensor(inv, dtype=torch.float)


class Trainer:
    """Minimal Trainer for PyG graph classification."""

    def __init__(self, model: nn.Module, cfg: dict, device: str = "cuda"):
        self.model = model.to(device)
        self.device = device
        self.cfg = cfg
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.get("lr", 1e-3),
            weight_decay=cfg.get("weight_decay", 5e-4),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, cfg.get("epochs", 200))
        )
        self.class_weights: Optional[torch.Tensor] = None

    # ----- internals -----
    def _forward(self, batch):
        x, edge_index = batch.x.to(self.device), batch.edge_index.to(self.device)
        batch_idx = batch.batch.to(self.device) if hasattr(batch, "batch") else None
        return self.model(x, edge_index, batch_idx)

    def _loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        w = self.class_weights.to(self.device) if self.class_weights is not None else None
        return F.cross_entropy(logits, y, weight=w)

    # ----- public -----
    def train(
        self,
        train_loader,
        val_loader,
        epochs: int = 200,
        save_dir: str = "./outputs",
    ) -> dict:
        os.makedirs(save_dir, exist_ok=True)

        train_labels = [int(d.y.item()) for d in train_loader.dataset]
        self.class_weights = _compute_class_weights(train_labels)
        logger.info("class weights: %s", self.class_weights.tolist())

        best_auc = -1.0
        best_state = None
        patience = self.cfg.get("patience", 30)
        stale = 0
        history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "val_auc": [], "val_f1": [], "val_acc": [],
        }

        for epoch in range(1, epochs + 1):
            self.model.train()
            tot, n = 0.0, 0
            for batch in train_loader:
                self.optimizer.zero_grad()
                logits, _, _ = self._forward(batch)
                y = batch.y.to(self.device).view(-1)
                loss = self._loss(logits, y)
                loss.backward()
                self.optimizer.step()
                tot += loss.item() * y.size(0)
                n += y.size(0)
            self.scheduler.step()
            train_loss = tot / max(n, 1)

            val = self.evaluate(val_loader)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val["loss"])
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
                    "epoch %3d | train_loss=%.4f | val_auc=%.4f f1=%.4f acc=%.4f%s",
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
    def evaluate(self, loader) -> Dict[str, float]:
        self.model.eval()
        probs: List[np.ndarray] = []
        preds: List[int] = []
        ys: List[int] = []
        losses: List[float] = []
        for batch in loader:
            logits, _, _ = self._forward(batch)
            y = batch.y.to(self.device).view(-1)
            loss = F.cross_entropy(logits, y)
            losses.append(float(loss.item()))
            p = F.softmax(logits, dim=1).detach().cpu().numpy()
            probs.append(p[:, 1])
            preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            ys.extend(y.cpu().numpy().tolist())
        y_true = np.array(ys)
        y_pred = np.array(preds)
        y_prob = np.concatenate(probs) if probs else np.zeros_like(y_true, dtype=float)

        metrics = {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
            "f1": float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) else 0.0,
            "precision": float(precision_score(y_true, y_pred, zero_division=0)) if len(y_true) else 0.0,
            "recall": float(recall_score(y_true, y_pred, zero_division=0)) if len(y_true) else 0.0,
        }
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
        except Exception:  # noqa: BLE001
            metrics["auc"] = 0.0
        return metrics

    @torch.no_grad()
    def extract_embeddings(self, loader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval()
        embs, labels, preds = [], [], []
        for batch in loader:
            _, emb, _ = self._forward(batch)
            embs.append(emb.detach().cpu().numpy())
            labels.extend(batch.y.view(-1).cpu().numpy().tolist())
            preds.extend([0] * batch.y.view(-1).size(0))
        return np.concatenate(embs, axis=0), np.array(labels), np.array(preds)


def cross_validate(
    dataset: List[dict],
    config: dict,
    n_folds: int = 5,
    device: str = "cuda",
) -> dict:
    """Stratified k-fold CV."""
    from torch_geometric.loader import DataLoader
    from models import build_model

    labels = np.array([d["label"] for d in dataset])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_metrics: List[dict] = []
    batch_size = config["training"].get("batch_size", 8)

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(dataset)), labels)):
        logger.info("=== fold %d/%d ===", fold + 1, n_folds)
        train_data = [dataset[i] for i in train_idx]
        val_data = [dataset[i] for i in val_idx]

        train_loader = DataLoader([d["graph"] for d in train_data], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader([d["graph"] for d in val_data], batch_size=batch_size)

        mcfg = dict(config["model"])
        mcfg["in_channels"] = train_data[0]["graph"].x.shape[1]
        mcfg["out_channels"] = 2
        model = build_model(mcfg)

        trainer = Trainer(model, config["training"], device)
        save_dir = os.path.join(config.get("output_dir", "./outputs"), f"fold_{fold}")
        trainer.train(train_loader, val_loader,
                      epochs=config["training"].get("epochs", 200),
                      save_dir=save_dir)
        val_metrics = trainer.evaluate(val_loader)
        val_metrics["fold"] = fold
        fold_metrics.append(val_metrics)
        logger.info("fold %d: %s", fold, val_metrics)

    agg = {}
    for k in ("auc", "accuracy", "f1", "precision", "recall"):
        vals = [m[k] for m in fold_metrics]
        agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return {"folds": fold_metrics, "mean": agg}
