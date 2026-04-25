"""Sprint 6 — Expanded 282-patient dataset + all improvements.

Combines multiple improvement directions:
  Dir 1: Expanded 282-patient dataset (170 PH + 112 nonPH)
  Dir 2: Graph augmentation (--augment: edge_drop, feature_mask, subgraph)
  Dir 4: PA/Ao ratio as extra global feature (--pa_ao_json)
  Dir 5: Model improvements (--residual, --jk)

Two experimental arms:
  arm_a: gcn_only on all 282 patients (baseline + enhanced features)
  arm_b: full 3-mode on original ~100-patient subset (comparison baseline)

Uses focal loss + Youden threshold (Sprint 3 best setup).

Run on server:
    cd '/home/imss/cw/GCN copdnoph copdph'
    python run_sprint6.py --arm arm_a \
        --cache_dir ./cache \
        --radiomics ./data/copd_ph_radiomics.csv \
        --labels ./data/labels_expanded_282.csv \
        --splits ./data/splits_expanded_282 \
        --output_dir ./outputs/sprint6_arm_a \
        --epochs 300 --batch_size 8 \
        --pa_ao_json ./data/ct_pa_ao_measurements_v2.json \
        --augment edge_drop,feature_mask \
        --residual
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
import torch


# === ROUND 10 — Gradient Reversal Layer (GRL) ===
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def _grad_reverse(x, lambd):
    return _GradReverse.apply(x, lambd)


class _ProtocolAdvHead(torch.nn.Module):
    def __init__(self, embed_dim, hidden=64):
        super().__init__()
        self.fc1 = torch.nn.Linear(embed_dim, hidden)
        self.fc2 = torch.nn.Linear(hidden, 2)

    def forward(self, z, lambd):
        r = _grad_reverse(z, lambd)
        return self.fc2(torch.nn.functional.relu(self.fc1(r)))
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.metrics import roc_auc_score, roc_curve
from torch_geometric.loader import DataLoader

from hybrid_gcn import HybridGCN
from enhance_features import augment_graph, GLOBAL_FEATURE_DIM, BASELINE_IN_DIM
from utils.graph_builder import normalize_graph_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("sprint6")


# ──────────────── Graph Augmentation (Direction 2) ────────────────
def augment_edge_drop(data, p=0.05):
    """Randomly drop edges with probability p during training."""
    if data.edge_index.size(1) == 0:
        return data
    mask = torch.rand(data.edge_index.size(1)) > p
    data.edge_index = data.edge_index[:, mask]
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        data.edge_attr = data.edge_attr[mask]
    return data


def augment_feature_mask(data, p=0.1):
    """Randomly mask (zero out) node feature dimensions with probability p."""
    if data.x.size(0) == 0:
        return data
    mask = torch.rand_like(data.x) > p
    data.x = data.x * mask.float()
    return data


def augment_subgraph(data, keep_ratio=0.8):
    """Keep a random connected subgraph by sampling nodes."""
    n = data.x.size(0)
    if n <= 3:
        return data
    k = max(3, int(n * keep_ratio))
    perm = torch.randperm(n)[:k]
    perm = perm.sort().values

    # Remap edges
    mask = torch.zeros(n, dtype=torch.bool)
    mask[perm] = True
    remap = torch.full((n,), -1, dtype=torch.long)
    remap[perm] = torch.arange(k)

    if data.edge_index.size(1) > 0:
        edge_mask = mask[data.edge_index[0]] & mask[data.edge_index[1]]
        new_ei = remap[data.edge_index[:, edge_mask]]
    else:
        new_ei = torch.zeros((2, 0), dtype=torch.long)

    data.x = data.x[perm]
    data.edge_index = new_ei
    if hasattr(data, "pos") and data.pos is not None:
        data.pos = data.pos[perm]
    if hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_index.size(1) > 0:
        data.edge_attr = data.edge_attr[edge_mask]
    data.num_nodes = k
    return data


def apply_augmentation(data, aug_types):
    """Apply a chain of augmentations to a PyG Data object (training only)."""
    for aug in aug_types:
        if aug == "edge_drop":
            data = augment_edge_drop(data, p=0.05)
        elif aug == "feature_mask":
            data = augment_feature_mask(data, p=0.1)
        elif aug == "subgraph":
            data = augment_subgraph(data, keep_ratio=0.8)
    return data


# ──────────────── PA/Ao Loading (Direction 4) ────────────────
def load_pa_ao(json_path: str) -> dict:
    """Load PA/Ao measurements JSON, return {case_id: pa_ao_ratio}."""
    with open(json_path) as f:
        data = json.load(f)
    measurements = data.get("measurements", {})
    result = {}
    for case_id, m in measurements.items():
        ratio = m.get("pa_ao_ratio")
        if ratio is not None and np.isfinite(ratio):
            result[case_id] = float(ratio)
    logger.info("Loaded PA/Ao ratios for %d cases", len(result))
    return result


# ──────────────── Metrics ────────────────
def full_metrics(y_true, y_pred, y_prob):
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score)
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


# ──────────────── Losses ────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = float(gamma)

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def build_cb_weights(n0, n1, beta=0.9999):
    samples = np.array([max(n0, 1), max(n1, 1)], dtype=np.float64)
    eff = (1.0 - np.power(beta, samples)) / (1.0 - beta)
    w = 1.0 / eff
    w = w / w.sum() * 2.0
    return torch.tensor(w, dtype=torch.float32)


def youden_threshold(y_true, y_score):
    if len(set(y_true)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, y_score)
    j = tpr - fpr
    best = int(np.argmax(j))
    t = float(thr[best])
    return min(max(t, 1e-4), 1.0 - 1e-4)


# ──────────────── Data loading ────────────────
def case_to_pinyin(case_id: str) -> str:
    parts = case_id.split("_")
    if len(parts) >= 3 and parts[0] in ("nonph", "ph"):
        return parts[1].lower()
    return case_id.lower()


def load_labels(path: str) -> dict:
    labels = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        id_key = "case_id" if "case_id" in (reader.fieldnames or []) else "patient_id"
        for row in reader:
            labels[row[id_key]] = int(row["label"])
    return labels


def load_splits_folder(splits_dir: str, n_folds: int = 5):
    folds = []
    for k in range(1, n_folds + 1):
        fd = Path(splits_dir) / f"fold_{k}"
        tr = [c.strip() for c in (fd / "train.txt").read_text().splitlines() if c.strip()]
        va = [c.strip() for c in (fd / "val.txt").read_text().splitlines() if c.strip()]
        folds.append((tr, va))
    return folds


def load_radiomics(path: str):
    import pandas as pd
    df = pd.read_csv(path)
    df["patient_id"] = df["patient_id"].astype(str).str.strip().str.lower()
    feat_cols = [c for c in df.columns if c not in ("patient_id", "label")]
    df[feat_cols] = df[feat_cols].apply(lambda s: s.fillna(s.median()))
    return df, feat_cols


def _first_col(df, needles):
    for c in df.columns:
        s = str(c)
        if all(n in s for n in needles):
            return c
    return None


def _get(full, col):
    if col is None or full is None:
        return None
    try:
        return float(full[col])
    except (TypeError, ValueError, KeyError):
        return None


def build_dataset_expanded(cache_dir, labels, rad_df, feat_cols, *, enhanced, require_radiomics,
                           keep_full_node_dim=False):
    """Build dataset from cache. If require_radiomics=False, patients without
    radiomics get zero-filled radiomics vectors (for gcn_only mode)."""
    import pandas as pd

    pinyin_to_rad = {
        str(pid).strip().lower(): row_vals.astype(np.float32)
        for pid, row_vals in zip(rad_df["patient_id"].values, rad_df[feat_cols].values)
    }

    # Enhancement column lookups
    pinyin_to_full = {
        str(pid).strip().lower(): row
        for pid, row in rad_df.set_index("patient_id").iterrows()
    }
    col_total = _first_col(rad_df, ["肺血管容积"])
    col_fractal = _first_col(rad_df, ["肺血管分形维度"]) or _first_col(rad_df, ["分形维度"])
    col_art_dens = _first_col(rad_df, ["动脉平均密度"])
    col_vein_dens = _first_col(rad_df, ["静脉平均密度"])
    col_vein_bv5 = _first_col(rad_df, ["静脉BV5"])
    col_vein_branches = _first_col(rad_df, ["静脉血管分支数量"])
    col_bv5_ratio = _first_col(rad_df, ["bv5_ratio"])
    col_av_ratio = _first_col(rad_df, ["artery_vein_vol_ratio"])
    col_total_bv5 = _first_col(rad_df, ["肺血管BV5"])
    col_lung_std = _first_col(rad_df, ["左右肺密度标准差"])
    col_vein_bv10 = _first_col(rad_df, ["静脉BV10"])
    col_total_branches = _first_col(rad_df, ["肺血管血管分支数量"]) \
        or _first_col(rad_df, ["肺血管分支数量"])
    col_tort = _first_col(rad_df, ["肺血管弯曲度"])

    rad_dim = len(feat_cols)
    zero_rad = np.zeros(rad_dim, dtype=np.float32)

    dataset = []
    missing_cache = 0
    missing_rad = 0
    has_rad_count = 0

    for case_id, label in labels.items():
        pkl = Path(cache_dir) / f"{case_id}.pkl"
        if not pkl.exists():
            missing_cache += 1
            continue

        py = case_to_pinyin(case_id)
        rad = pinyin_to_rad.get(py)
        has_rad = rad is not None

        if require_radiomics and not has_rad:
            missing_rad += 1
            continue

        if not has_rad:
            rad = zero_rad.copy()
        else:
            has_rad_count += 1

        with pkl.open("rb") as f:
            entry = pickle.load(f)

        g = entry["graph"]

        # Align feature dimensions — old caches may have 15D, new have 12D (or 13D for tri-flat)
        # Truncate to BASELINE_IN_DIM (12) unless keep_full_node_dim (tri-flat 13D with struct_id)
        if not keep_full_node_dim and g.x.size(1) > BASELINE_IN_DIM:
            g.x = g.x[:, :BASELINE_IN_DIM]

        # Normalize edge_attr dimension (old=4D, new=3D) — truncate to min
        if hasattr(g, "edge_attr") and g.edge_attr is not None and g.edge_attr.size(1) > 3:
            g.edge_attr = g.edge_attr[:, :3]

        # Remove extra attributes that only exist in some caches
        for attr in ("airway_ratio", "artery_ratio", "vein_ratio", "node_type"):
            if hasattr(g, attr):
                delattr(g, attr)

        if enhanced:
            full = pinyin_to_full.get(py) if has_rad else None
            pipeline_vol = entry.get("features", {}).get("vascular", {}).get(
                "total_vessel_volume_ml"
            )
            g = augment_graph(
                g,
                commercial_total_vol_ml=_get(full, col_total),
                commercial_fractal_dim=_get(full, col_fractal),
                commercial_artery_density=_get(full, col_art_dens),
                commercial_vein_density=_get(full, col_vein_dens),
                pipeline_total_vol_ml=float(pipeline_vol) if pipeline_vol else None,
                commercial_vein_bv5=_get(full, col_vein_bv5),
                commercial_vein_branch_count=_get(full, col_vein_branches),
                commercial_bv5_ratio=_get(full, col_bv5_ratio),
                commercial_artery_vein_vol_ratio=_get(full, col_av_ratio),
                commercial_total_bv5=_get(full, col_total_bv5),
                commercial_lung_density_std=_get(full, col_lung_std),
                commercial_vein_bv10=_get(full, col_vein_bv10),
                commercial_total_branch_count=_get(full, col_total_branches),
                commercial_vessel_tortuosity=_get(full, col_tort),
            )

        dataset.append({
            "patient_id": case_id,
            "graph": g,
            "radiomics": rad,
            "label": int(label),
            "has_radiomics": has_rad,
        })

    pos = sum(1 for e in dataset if e["label"] == 1)
    logger.info("[%s] matched=%d (pos=%d neg=%d) has_rad=%d cache_miss=%d rad_miss=%d",
                "enhanced" if enhanced else "baseline",
                len(dataset), pos, len(dataset) - pos,
                has_rad_count, missing_cache, missing_rad)
    return dataset


# ──────────────── Globals subset ────────────────
LOCAL4_IDX = [5, 7, 10, 11]


def apply_globals_keep(dataset, keep):
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
        return int(g0.global_features.size(-1))
    if keep == "local4":
        idx = torch.tensor(LOCAL4_IDX, dtype=torch.long)
        for e in dataset:
            g = e["graph"]
            g.global_features = g.global_features.index_select(-1, idx)
        return len(LOCAL4_IDX)
    raise ValueError(f"unknown globals_keep={keep}")


# ──────────────── DataLoader ────────────────
def make_loaders(dataset, train_ids, val_ids, batch_size, aug_types=None, num_workers=0):
    id2e = {e["patient_id"]: e for e in dataset}
    tr = [id2e[i] for i in train_ids if i in id2e]
    va = [id2e[i] for i in val_ids if i in id2e]

    tr_graphs = normalize_graph_features([e["graph"].clone() for e in tr])
    va_graphs = normalize_graph_features([e["graph"].clone() for e in va])

    rad_tr = np.stack([e["radiomics"] for e in tr])
    rad_tr = np.nan_to_num(rad_tr, nan=0.0, posinf=0.0, neginf=0.0)
    mu = rad_tr.mean(axis=0)
    sd = rad_tr.std(axis=0) + 1e-6

    def std(v):
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        return ((v - mu) / sd).astype(np.float32)

    def _wrap(g, rad_vec, src, is_train=False):
        g.radiomics = torch.from_numpy(rad_vec).float().unsqueeze(0)
        gf = getattr(src, "global_features", None)
        if gf is not None:
            g.global_features = gf.clone() if hasattr(gf, "clone") else gf
        # Apply augmentation to training graphs
        if is_train and aug_types:
            g = apply_augmentation(g, aug_types)
        return g

    tr_items = [_wrap(g, std(e["radiomics"]), e["graph"], is_train=True)
                for g, e in zip(tr_graphs, tr)]
    va_items = [_wrap(g, std(e["radiomics"]), e["graph"], is_train=False)
                for g, e in zip(va_graphs, va)]

    tl = DataLoader(tr_items, batch_size=batch_size, shuffle=True,
                    drop_last=len(tr_items) > batch_size,
                    num_workers=num_workers, persistent_workers=(num_workers > 0))
    vl = DataLoader(va_items, batch_size=batch_size,
                    num_workers=num_workers, persistent_workers=(num_workers > 0))
    tr_labels = np.array([int(e["label"]) for e in tr])
    return tl, vl, tr_labels


# ──────────────── Training ────────────────
def train_fold(mode, radiomics_dim, gcn_in, tl, vl, trl, device, epochs, lr, wd,
               patience=40, global_dim=0, residual=False, jk="none", adv_lambda=0.0):
    model = HybridGCN(gcn_in=gcn_in, gcn_hidden=64, radiomics_dim=radiomics_dim,
                      num_layers=3, dropout=0.3, mode=mode,
                      global_dim=global_dim, residual=residual, jk=jk).to(device)
    adv_head = _ProtocolAdvHead(64).to(device) if adv_lambda > 0 else None
    adv_opt = torch.optim.Adam(adv_head.parameters(), lr=lr) if adv_head is not None else None
    adv_criterion = torch.nn.CrossEntropyLoss()
    n1 = int((trl == 1).sum())
    n0 = int((trl == 0).sum())
    w = build_cb_weights(n0, n1).to(device)
    criterion = FocalLoss(alpha=w, gamma=2.0)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_auc = -1.0
    best_state = None
    bad = 0

    for epoch in range(epochs):
        model.train()
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
            # ROUND 11 FIX: GRL coef=1.0 (not λ — that goes on the loss);
            # adversary on nonPH-only to match within-nonPH evaluation.
            if adv_head is not None and hasattr(batch, "is_contrast"):
                is_c_full = batch.is_contrast.view(-1).to(logits.device)
                y_full = batch.y.view(-1).to(logits.device)
                nph_mask = (y_full == 0)
                if nph_mask.sum() >= 2 and is_c_full[nph_mask].unique().numel() == 2:
                    z_nph = z_proj[nph_mask]
                    c_nph = is_c_full[nph_mask]
                    adv_logits = adv_head(z_nph, 1.0)
                    adv_loss_value = adv_criterion(adv_logits, c_nph)
                    loss = loss + adv_lambda * adv_loss_value
                    # Diagnostic: track adversary's protocol AUC on this batch
                    with torch.no_grad():
                        adv_prob = torch.softmax(adv_logits.detach(), dim=1)[:, 1].cpu().numpy()
                        try:
                            from sklearn.metrics import roc_auc_score as _ras
                            _adv_auc = _ras(c_nph.cpu().numpy(), adv_prob)
                        except Exception:
                            _adv_auc = float("nan")
                        if not hasattr(adv_head, "_auc_log"):
                            adv_head._auc_log = []
                        adv_head._auc_log.append(_adv_auc)
            opt.zero_grad()
            if adv_opt is not None:
                adv_opt.zero_grad()
            loss.backward()
            opt.step()
            if adv_opt is not None:
                adv_opt.step()

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
        if adv_head is not None and getattr(adv_head, "_auc_log", []):
            _recent = adv_head._auc_log[-50:]
            _recent = [v for v in _recent if v == v]  # filter NaN
            if _recent:
                logger.info("    [adv] epoch=%d batch_auc_mean=%.3f n_batches=%d",
                            epoch, sum(_recent)/len(_recent), len(_recent))
            adv_head._auc_log = []  # reset for next epoch
        if auc > best_auc:
            best_auc = auc
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final eval with Youden threshold + embedding extraction
    model.eval()
    yt, pr = [], []
    embs = []
    with torch.no_grad():
        for batch in vl:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch,
                        getattr(batch, "radiomics", None),
                        global_features=getattr(batch, "global_features", None))
            if len(out) == 4:
                logits, z_proj, _, _ = out
            elif len(out) == 3:
                logits, z_proj, _ = out
            else:
                logits = out[0]
                z_proj = torch.zeros(logits.size(0), 1, device=logits.device)
            prob = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            yt += batch.y.view(-1).cpu().numpy().tolist()
            pr += prob.tolist()
            embs.append(z_proj.detach().cpu().numpy())
    if embs:
        import numpy as _np_emb
        _embs_concat = _np_emb.concatenate(embs, axis=0)
    else:
        import numpy as _np_emb
        _embs_concat = _np_emb.zeros((0, 1))

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
    return out, yt, pr, _embs_concat


def run_all(dataset, folds, device, args, radiomics_dim, gcn_in, global_dim, modes,
            aug_types=None, residual=False, jk="none", repeats=1):
    out = {}
    for mode in modes:
        logger.info("=== mode=%s gcn_in=%d global_dim=%d aug=%s res=%s jk=%s repeats=%d ===",
                    mode, gcn_in, global_dim, aug_types, residual, jk, repeats)

        # For repeated CV (Direction 3): train multiple times with different seeds
        # and ensemble soft-vote the predictions
        all_repeat_metrics = []
        all_repeat_yts = []
        all_repeat_prs = []

        for rep in range(repeats):
            if repeats > 1:
                rep_seed = args.seed + rep * 1000
                torch.manual_seed(rep_seed)
                np.random.seed(rep_seed)
                logger.info("  repeat %d/%d (seed=%d)", rep + 1, repeats, rep_seed)

            fold_metrics, yts, prs = [], [], []
            for k, (tr, va) in enumerate(folds, 1):
                tl, vl, trl = make_loaders(dataset, tr, va, args.batch_size,
                                           aug_types=aug_types,
                                           num_workers=getattr(args, "num_workers", 0))
                m, yt, pr, embs = train_fold(mode, radiomics_dim, gcn_in, tl, vl, trl,
                                       device, args.epochs, args.lr, args.wd,
                                       global_dim=global_dim, residual=residual, jk=jk,
                                       adv_lambda=getattr(args, 'adv_lambda', 0.0))
                _emb_save_dir = getattr(args, "_emb_save_dir", None)
                if _emb_save_dir:
                    import numpy as _np_es
                    import os as _os_es
                    _os_es.makedirs(_emb_save_dir, exist_ok=True)
                    _np_es.savez(
                        _os_es.path.join(_emb_save_dir, f"emb_{mode}_rep{rep+1}_fold{k}.npz"),
                        embeddings=embs, y_true=_np_es.asarray(yt),
                    )
                logger.info("  fold %d thr=%.3f AUC=%.4f Acc=%.4f Spec=%.4f F1=%.4f",
                            k, m["threshold"], m["AUC"], m["Accuracy"],
                            m["Specificity"], m["F1"])
                fold_metrics.append(m)
                yts += yt
                prs += pr

            all_repeat_metrics.append(fold_metrics)
            all_repeat_yts.append(yts)
            all_repeat_prs.append(prs)

        # If repeats > 1: ensemble by averaging predictions across repeats
        if repeats > 1:
            # Average probabilities across repeats for each patient
            avg_prs = np.mean(all_repeat_prs, axis=0).tolist()
            yts = all_repeat_yts[0]  # same patients, same order

            # Recompute metrics from ensemble predictions
            y_true = np.array(yts)
            y_score = np.array(avg_prs)
            thr = youden_threshold(yts, avg_prs)
            y_pred = (y_score >= thr).astype(int)
            ensemble_metrics = full_metrics(y_true, y_pred, y_score)
            ensemble_metrics["threshold"] = float(thr)

            # Also compute per-repeat stats for reference
            all_folds_flat = [m for rep_m in all_repeat_metrics for m in rep_m]
            keys = [k for k in all_folds_flat[0].keys() if k != "threshold"]
            mean = {k: float(np.mean([fm[k] for fm in all_folds_flat])) for k in keys}
            std = {k: float(np.std([fm[k] for fm in all_folds_flat])) for k in keys}
            pooled = float(roc_auc_score(yts, avg_prs)) if len(set(yts)) > 1 else 0.0

            out[mode] = {
                "folds": all_folds_flat,
                "mean": mean,
                "std": std,
                "pooled_AUC": pooled,
                "ensemble_metrics": ensemble_metrics,
                "n_repeats": repeats,
                "ensemble_y_true": list(map(int, yts)),
                "ensemble_y_score": list(map(float, avg_prs)),
                "per_repeat_y_score": [list(map(float, pr)) for pr in all_repeat_prs],
            }
            logger.info("  ensemble AUC=%.4f Spec=%.4f F1=%.4f",
                        ensemble_metrics["AUC"], ensemble_metrics["Specificity"],
                        ensemble_metrics["F1"])
        else:
            fold_metrics = all_repeat_metrics[0]
            yts = all_repeat_yts[0]
            prs = all_repeat_prs[0]
            keys = [k for k in fold_metrics[0].keys() if k != "threshold"]
            mean = {k: float(np.mean([fm[k] for fm in fold_metrics])) for k in keys}
            std = {k: float(np.std([fm[k] for fm in fold_metrics])) for k in keys}
            pooled = float(roc_auc_score(yts, prs)) if len(set(yts)) > 1 else 0.0
            out[mode] = {"folds": fold_metrics, "mean": mean, "std": std,
                         "pooled_AUC": pooled,
                         "ensemble_y_true": list(map(int, yts)),
                         "ensemble_y_score": list(map(float, prs))}
    return out


# ──────────────── Main ────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=["arm_a", "arm_b", "arm_c"],
                   help="arm_a=gcn_only on 282, arm_b=3-mode on ~100")
    p.add_argument("--cache_dir", default="./cache")
    p.add_argument("--radiomics", default="./data/copd_ph_radiomics.csv")
    p.add_argument("--labels", default="./data/labels_expanded_282.csv")
    p.add_argument("--splits", default="./data/splits_expanded_282")
    p.add_argument("--output_dir", default="./outputs/sprint6")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--globals_keep", default="local4",
                   choices=["all", "local4", "none"])
    # Direction 2: graph augmentation
    p.add_argument("--augment", default="",
                   help="Comma-separated augmentation types: edge_drop,feature_mask,subgraph")
    # Direction 4: PA/Ao ratio
    p.add_argument("--pa_ao_json", default="",
                   help="Path to ct_pa_ao_measurements_v2.json for PA/Ao ratio feature")
    # Direction 5: model improvements
    p.add_argument("--residual", action="store_true",
                   help="Add residual connections in GCN layers")
    # Direction 3: repeated CV + ensemble
    p.add_argument("--lung_features_csv", default="",
                   help="CSV with case_id + 13 lung scalars (arm_c lung globals)")
    p.add_argument("--keep_full_node_dim", action="store_true",
                   help="Do NOT truncate node features to 12D (enables tri-flat 13D input)")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader num_workers (default 0)")
    p.add_argument("--skip_enhanced", action="store_true",
                   help="Only run baseline feature set (skip enhanced radiomics augmentation)")
    p.add_argument("--repeats", type=int, default=1,
                   help="Number of repeated CV runs for ensemble (default 1)")
    p.add_argument("--adv_lambda", type=float, default=0.0,
                   help="GRL lambda for protocol-adversarial head (0=off)")
    p.add_argument("--protocol_csv", type=str, default=None,
                   help="case_id,protocol CSV for adversarial targets")
    p.add_argument("--dump_embeddings", action="store_true",
                   help="Save per-fold penultimate embeddings as .npz under output_dir/embeddings/")
    args = p.parse_args()
    _proto_map = {}
    if getattr(args, 'protocol_csv', None):
        import csv as _csv_adv
        with open(args.protocol_csv, encoding='utf-8') as _f:
            for _r in _csv_adv.DictReader(_f):
                _proto_map[_r['case_id']] = int(_r.get('protocol', 'plain_scan') == 'contrast')
        print(f'[ROUND10 GRL] loaded {len(_proto_map)} protocol labels')
    if args.dump_embeddings:
        import os as _os_main
        args._emb_save_dir = _os_main.path.join(args.output_dir, 'embeddings')
    else:
        args._emb_save_dir = None

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("[config] arm=%s device=%s epochs=%d", args.arm, device, args.epochs)

    labels = load_labels(args.labels)
    folds = load_splits_folder(args.splits)
    rad_df, feat_cols = load_radiomics(args.radiomics)
    radiomics_dim = len(feat_cols)

    # Parse augmentation types
    aug_types = [a.strip() for a in args.augment.split(",") if a.strip()] if args.augment else []
    logger.info("[config] augment=%s pa_ao=%s residual=%s", aug_types, bool(args.pa_ao_json), args.residual)

    # Load PA/Ao ratios if provided
    pa_ao_ratios = {}
    if args.pa_ao_json:
        pa_ao_ratios = load_pa_ao(args.pa_ao_json)

    all_results = {"_config": {
        "arm": args.arm,
        "globals_keep": args.globals_keep,
        "epochs": args.epochs,
        "lr": args.lr,
        "wd": args.wd,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "total_labels": len(labels),
        "augment": aug_types,
        "pa_ao": bool(args.pa_ao_json),
        "residual": args.residual,
        "repeats": args.repeats,
    }}

    if args.arm == "arm_a":
        # gcn_only on all 282 patients (no radiomics required)
        modes = ["gcn_only"]
        require_rad = False
    elif args.arm == "arm_c":
        # arm_c: gcn_only on 282 + tri-flat 13D + lung scalar globals
        modes = ["gcn_only"]
        require_rad = False
    else:
        # arm_b: full 3-mode on patients with radiomics only
        modes = ["radiomics_only", "gcn_only", "hybrid"]
        require_rad = True

    # Load lung features (arm_c or on-demand)
    lung_feats = {}
    LUNG_COLS = ["lung_vol_mL", "mean_HU", "std_HU", "HU_p5", "HU_p25",
                 "HU_p50", "HU_p75", "HU_p95", "LAA_950_frac", "LAA_910_frac",
                 "LAA_856_frac", "largest_comp_frac", "n_components"]
    if args.lung_features_csv:
        import pandas as _pd
        _lf_df = _pd.read_csv(args.lung_features_csv)
        _means = _lf_df[LUNG_COLS].median(numeric_only=True).to_dict()
        for _, _row in _lf_df.iterrows():
            _cid = str(_row["case_id"]).strip()
            _vec = []
            for _c in LUNG_COLS:
                _v = _row.get(_c)
                if _v is None or (isinstance(_v, float) and np.isnan(_v)):
                    _v = _means.get(_c, 0.0)
                _vec.append(float(_v))
            lung_feats[_cid] = np.array(_vec, dtype=np.float32)
        # z-score normalize lung features across cohort
        if lung_feats:
            _mat = np.stack(list(lung_feats.values()))
            _mu = _mat.mean(axis=0)
            _sd = _mat.std(axis=0) + 1e-6
            for _k in list(lung_feats.keys()):
                lung_feats[_k] = (lung_feats[_k] - _mu) / _sd
            logger.info("Loaded lung features for %d cases (z-scored, dim=%d)",
                        len(lung_feats), len(LUNG_COLS))

    # For tri-flat (arm_b_triflat, arm_c) skip the enhanced pass (radiomics assertions fail on 13D)
    phases = ["baseline", "enhanced"] if not args.skip_enhanced else ["baseline"]
    for fs in phases:
        enhanced = fs == "enhanced"
        dataset = build_dataset_expanded(
            args.cache_dir, labels, rad_df, feat_cols,
            enhanced=enhanced, require_radiomics=require_rad,
            keep_full_node_dim=args.keep_full_node_dim,
        )
        if not dataset:
            logger.warning("[%s] empty dataset, skipping", fs)
            continue

        id_set = {e["patient_id"] for e in dataset}
        filtered = [([i for i in tr if i in id_set], [i for i in va if i in id_set])
                    for tr, va in folds]
        if _proto_map:
            for _entry in dataset:
                _cid = _entry.get("patient_id")
                _g = _entry.get("graph")
                if _g is not None and _cid is not None:
                    _g.is_contrast = torch.tensor([_proto_map.get(_cid, 0)], dtype=torch.long)
        gcn_in = dataset[0]["graph"].x.size(1)

        if enhanced:
            global_dim = apply_globals_keep(dataset, args.globals_keep)
            logger.info("[%s] globals_keep=%s -> global_dim=%d", fs, args.globals_keep, global_dim)
        else:
            global_dim = 0

        # Direction 4: Inject PA/Ao ratio as extra global feature
        if pa_ao_ratios and enhanced:
            miss = 0
            for e in dataset:
                g = e["graph"]
                ratio = pa_ao_ratios.get(e["patient_id"], None)
                pa_val = float(ratio) if ratio is not None else 0.0
                if ratio is None:
                    miss += 1
                pa_tensor = torch.tensor([[pa_val]], dtype=torch.float32)
                if hasattr(g, "global_features") and g.global_features is not None:
                    g.global_features = torch.cat([g.global_features, pa_tensor], dim=1)
                else:
                    g.global_features = pa_tensor
            global_dim += 1
            logger.info("[%s] PA/Ao ratio injected (miss=%d/%d) -> global_dim=%d",
                        fs, miss, len(dataset), global_dim)

        # arm_c: inject 13-d lung scalar features as extra globals (baseline OR enhanced)
        if lung_feats:
            miss_lf = 0
            lung_dim = len(LUNG_COLS)
            for e in dataset:
                g = e["graph"]
                v = lung_feats.get(e["patient_id"])
                if v is None:
                    miss_lf += 1
                    v = np.zeros(lung_dim, dtype=np.float32)
                lf_tensor = torch.tensor(v.reshape(1, -1), dtype=torch.float32)
                if hasattr(g, "global_features") and g.global_features is not None:
                    g.global_features = torch.cat([g.global_features, lf_tensor], dim=1)
                else:
                    g.global_features = lf_tensor
            global_dim += lung_dim
            logger.info("[%s] lung features injected (miss=%d/%d, +%d dims) -> global_dim=%d",
                        fs, miss_lf, len(dataset), lung_dim, global_dim)

        all_results[fs] = run_all(dataset, filtered, device, args,
                                  radiomics_dim, gcn_in, global_dim, modes,
                                  aug_types=aug_types, residual=args.residual,
                                  repeats=args.repeats)

    # Save
    json_path = out_dir / "sprint6_results.json"
    with json_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("saved %s", json_path)

    # Summary
    print(f"\n{'='*120}")
    print(f"SPRINT 6 SUMMARY (arm={args.arm})")
    print("=" * 120)
    keys = ["AUC", "Accuracy", "Precision", "Sensitivity", "F1", "Specificity"]
    for fs, modes_dict in all_results.items():
        if fs == "_config":
            continue
        for mode, r in modes_dict.items():
            m, s = r["mean"], r["std"]
            print(f"{fs:<10} {mode:<16} "
                  + " ".join(f"{m[k]:.4f}+/-{s[k]:.4f}" for k in keys)
                  + f"  pooled={r['pooled_AUC']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
