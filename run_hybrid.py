"""Sprint 1 — 5-fold CV training, three modes: radiomics_only / gcn_only / hybrid.

Loads cached graph data (cache/*.pkl) + commercial CT radiomics
(data/copd_ph_radiomics.csv). Only patients present in BOTH sources are used.

Run on remote server:
    source /home/imss/miniconda3/etc/profile.d/conda.sh
    conda activate pulmonary_bv5_py39
    CUDA_VISIBLE_DEVICES=0 python run_hybrid.py \
        --cache_dir ./cache \
        --radiomics ./data/copd_ph_radiomics.csv \
        --splits "/home/imss/cw/COPDnonPH COPD-PH /data/splits/folds" \
        --labels "/home/imss/cw/COPDnonPH COPD-PH /data/tables/labels.csv" \
        --output_dir ./outputs/sprint1_hybrid --epochs 300
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hybrid_gcn import HybridGCN
from utils.graph_builder import normalize_graph_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("sprint1")


# ======================================================================
# Data loading + matching
# ======================================================================

def case_to_pinyin(case_id: str) -> str:
    """Extract pinyin segment from `{nonph,ph}_{pinyin}_{idcard}_...`."""
    parts = case_id.split("_")
    if len(parts) >= 3 and parts[0] in ("nonph", "ph"):
        return parts[1].lower()
    return case_id.lower()


def load_labels(path: str) -> dict[str, int]:
    labels = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        id_key = "case_id" if "case_id" in (reader.fieldnames or []) else "patient_id"
        for row in reader:
            labels[row[id_key]] = int(row["label"])
    return labels


def load_splits(splits_dir: str, n_folds: int = 5):
    folds = []
    for k in range(1, n_folds + 1):
        fd = Path(splits_dir) / f"fold_{k}"
        tr = [c.strip() for c in (fd / "train.txt").read_text().splitlines() if c.strip()]
        va = [c.strip() for c in (fd / "val.txt").read_text().splitlines() if c.strip()]
        folds.append((tr, va))
    return folds


def load_radiomics(path: str) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path)
    df["patient_id"] = df["patient_id"].astype(str).str.strip().str.lower()
    feat_cols = [c for c in df.columns if c not in ("patient_id", "label")]
    # NaN fill with column median (robust, computed here; later sanitized again)
    df[feat_cols] = df[feat_cols].apply(lambda s: s.fillna(s.median()))
    logger.info("radiomics: %d patients × %d features", len(df), len(feat_cols))
    return df, feat_cols


def build_dataset(cache_dir: str, labels: dict, radiomics_df: pd.DataFrame,
                  feat_cols: list[str]) -> list[dict]:
    pinyin_to_rad = {
        str(pid).strip().lower(): row_vals.astype(np.float32)
        for pid, row_vals in zip(
            radiomics_df["patient_id"].values,
            radiomics_df[feat_cols].values,
        )
    }

    dataset = []
    missing_cache = missing_rad = 0
    for case_id, label in labels.items():
        pkl = Path(cache_dir) / f"{case_id}.pkl"
        if not pkl.exists():
            missing_cache += 1
            continue
        py = case_to_pinyin(case_id)
        rad = pinyin_to_rad.get(py)
        if rad is None:
            missing_rad += 1
            continue
        with pkl.open("rb") as f:
            entry = pickle.load(f)
        entry["patient_id"] = case_id
        entry["radiomics"] = rad
        entry["label"] = int(label)
        dataset.append(entry)

    logger.info("matched: %d  (cache-miss: %d, radiomics-miss: %d)",
                len(dataset), missing_cache, missing_rad)
    pos = sum(1 for e in dataset if e["label"] == 1)
    logger.info("class dist: pos=%d  neg=%d", pos, len(dataset) - pos)
    return dataset


# ======================================================================
# Data objects
# ======================================================================

def attach_radiomics(graph: Data, rad: np.ndarray) -> Data:
    """Return a new Data with `radiomics` as shape (1, D) so batching stacks to (B, D)."""
    g = graph.clone() if hasattr(graph, "clone") else Data(**{k: v for k, v in graph})
    g.radiomics = torch.from_numpy(rad).float().unsqueeze(0)
    return g


def make_loaders(dataset, train_ids, val_ids, batch_size, radiomics_dim):
    id2entry = {e["patient_id"]: e for e in dataset}
    tr = [id2entry[i] for i in train_ids if i in id2entry]
    va = [id2entry[i] for i in val_ids if i in id2entry]

    tr_graphs = normalize_graph_features([e["graph"] for e in tr])
    va_graphs = normalize_graph_features([e["graph"] for e in va])

    # train-set statistics for radiomics standardization
    rad_tr = np.stack([e["radiomics"] for e in tr])
    rad_tr = np.nan_to_num(rad_tr, nan=0.0, posinf=0.0, neginf=0.0)
    mu = rad_tr.mean(axis=0)
    sd = rad_tr.std(axis=0) + 1e-6

    def std(v):
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        return ((v - mu) / sd).astype(np.float32)

    tr_items = [attach_radiomics(g, std(e["radiomics"])) for g, e in zip(tr_graphs, tr)]
    va_items = [attach_radiomics(g, std(e["radiomics"])) for g, e in zip(va_graphs, va)]

    # label tensor lives on graph.y already
    tl = DataLoader(tr_items, batch_size=batch_size, shuffle=True)
    vl = DataLoader(va_items, batch_size=batch_size)
    tr_labels = np.array([int(e["label"]) for e in tr])
    return tl, vl, tr_labels


# ======================================================================
# Train / eval
# ======================================================================

def full_metrics(y_true, y_pred, y_prob):
    m = {
        "AUC": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    m["Specificity"] = float(tn / max(tn + fp, 1))
    return m


def train_one_fold(mode, radiomics_dim, tl, vl, tr_labels, device, epochs, lr, wd,
                   patience=40):
    model = HybridGCN(
        gcn_in=12, gcn_hidden=64, radiomics_dim=radiomics_dim,
        out_channels=2, num_layers=3, dropout=0.3, mode=mode,
    ).to(device)

    # class-weighted CE
    n1 = int((tr_labels == 1).sum()); n0 = int((tr_labels == 0).sum())
    w = torch.tensor([max(n1, 1) / max(n0, 1), 1.0], dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_auc = -1.0; best_state = None; bad = 0
    for epoch in range(epochs):
        model.train()
        for batch in tl:
            batch = batch.to(device)
            logits, _, _ = model(batch.x, batch.edge_index, batch.batch,
                                 getattr(batch, "radiomics", None))
            loss = F.cross_entropy(logits, batch.y.view(-1), weight=w)
            opt.zero_grad(); loss.backward(); opt.step()

        # val
        model.eval(); yt, yp, pr = [], [], []
        with torch.no_grad():
            for batch in vl:
                batch = batch.to(device)
                logits, _, _ = model(batch.x, batch.edge_index, batch.batch,
                                     getattr(batch, "radiomics", None))
                prob = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                pred = logits.argmax(dim=1).cpu().numpy()
                y = batch.y.view(-1).cpu().numpy()
                yt += y.tolist(); yp += pred.tolist(); pr += prob.tolist()
        auc = roc_auc_score(yt, pr) if len(set(yt)) > 1 else 0.0

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # final metrics on val
    model.eval(); yt, yp, pr = [], [], []
    with torch.no_grad():
        for batch in vl:
            batch = batch.to(device)
            logits, _, _ = model(batch.x, batch.edge_index, batch.batch,
                                 getattr(batch, "radiomics", None))
            prob = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pred = logits.argmax(dim=1).cpu().numpy()
            y = batch.y.view(-1).cpu().numpy()
            yt += y.tolist(); yp += pred.tolist(); pr += prob.tolist()

    m = full_metrics(np.array(yt), np.array(yp), np.array(pr))
    return m, yt, pr


# ======================================================================
# Main
# ======================================================================

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", default="./cache")
    p.add_argument("--radiomics", default="./data/copd_ph_radiomics.csv")
    p.add_argument("--labels", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--output_dir", default="./outputs/sprint1_hybrid")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--modes", default="radiomics_only,gcn_only,hybrid")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device=%s", device)

    # --- load data
    labels = load_labels(args.labels)
    folds = load_splits(args.splits)
    rad_df, feat_cols = load_radiomics(args.radiomics)
    dataset = build_dataset(args.cache_dir, labels, rad_df, feat_cols)
    id_set = {e["patient_id"] for e in dataset}
    logger.info("cohort for Sprint 1: %d patients", len(id_set))

    # filter fold ids to matched cohort
    filtered_folds = []
    for tr, va in folds:
        ftr = [i for i in tr if i in id_set]
        fva = [i for i in va if i in id_set]
        filtered_folds.append((ftr, fva))
        logger.info("fold: train=%d  val=%d", len(ftr), len(fva))

    results = {}
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        logger.info("=== MODE: %s ===", mode)
        fold_metrics, yts, prs = [], [], []
        for k, (tr_ids, va_ids) in enumerate(filtered_folds, 1):
            tl, vl, tr_lab = make_loaders(dataset, tr_ids, va_ids,
                                          args.batch_size, len(feat_cols))
            m, yt, pr = train_one_fold(mode, len(feat_cols), tl, vl, tr_lab,
                                       device, args.epochs, args.lr, args.wd)
            logger.info("  fold %d AUC=%.4f Acc=%.4f F1=%.4f", k, m["AUC"],
                        m["Accuracy"], m["F1"])
            fold_metrics.append(m); yts += yt; prs += pr

        keys = list(fold_metrics[0].keys())
        mean = {k: float(np.mean([fm[k] for fm in fold_metrics])) for k in keys}
        std = {k: float(np.std([fm[k] for fm in fold_metrics])) for k in keys}
        pooled_auc = float(roc_auc_score(yts, prs)) if len(set(yts)) > 1 else 0.0
        results[mode] = {
            "folds": fold_metrics,
            "mean": mean,
            "std": std,
            "pooled_AUC": pooled_auc,
        }
        logger.info("  mean AUC=%.4f±%.4f  pooled=%.4f", mean["AUC"], std["AUC"], pooled_auc)

    with (out / "sprint1_results.json").open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("saved %s", out / "sprint1_results.json")

    # summary table
    print("\n=== SPRINT 1 SUMMARY ===")
    print(f"{'mode':<18} {'AUC':>14} {'Acc':>14} {'F1':>14} {'Spec':>14} {'pooled_AUC':>12}")
    for mode, r in results.items():
        m, s = r["mean"], r["std"]
        print(f"{mode:<18} {m['AUC']:.4f}±{s['AUC']:.4f} "
              f"{m['Accuracy']:.4f}±{s['Accuracy']:.4f} "
              f"{m['F1']:.4f}±{s['F1']:.4f} "
              f"{m['Specificity']:.4f}±{s['Specificity']:.4f} "
              f"{r['pooled_AUC']:>12.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
