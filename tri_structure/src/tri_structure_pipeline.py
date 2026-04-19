#!/usr/bin/env python3
"""
tri_structure_pipeline.py — Unified classification + clustering pipeline.

The core idea: classification and clustering are NOT separate tasks on
separate models.  They share the SAME graph embedding produced by the
TriStructureGCN with cross-structure attention fusion.

Pipeline:
  1. Load cache → partition into (artery, vein, airway) subgraphs
  2. Train TriStructureGCN with classification loss (+ optional mPAP aux)
  3. Extract shared embeddings z_fused for ALL patients
  4. Run clustering on z_fused → discover structural phenotypes
  5. Analyse attention weights → which structure drives each patient's classification
  6. Cross-reference clusters with mPAP severity → validate phenotypes

Outputs:
  - cv_results.json           (classification metrics, 5-fold × 3 repeats)
  - shared_embeddings.npz     (z_fused + attention weights for all patients)
  - cluster_analysis.json     (clustering on shared embeddings)
  - attention_profiles.json   (per-patient structure attention distribution)

Usage (Claude Code):
    python tri_structure_pipeline.py \\
        --cache_dir ./cache \\
        --labels ./data/labels_gold.csv \\
        --mpap ./data/mpap_lookup_gold.json \\
        --output_dir ./outputs/tri_structure

    # Quick test with fewer epochs:
    python tri_structure_pipeline.py \\
        --cache_dir ./cache --labels ./data/labels_gold.csv \\
        --epochs 50 --repeats 1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, roc_curve,
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score,
)
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from graph_partitioner import (
    load_labels, load_mpap, load_tri_structure_dataset, normalize_per_structure,
    signature_feature_names,
)
from models import TriStructureGCN, DualStructureGCN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("tri_pipeline")


# ═══════════════════════════════════════════════════════════════════
# Training utilities
# ═══════════════════════════════════════════════════════════════════

def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    best = np.where(j == j.max())[0]
    bi = best[np.argmin(np.abs(thr[best] - 0.5))]
    return float(max(0.0, min(1.0, thr[bi])))


def compute_metrics(y_true, y_prob, threshold=0.5):
    preds = (y_prob >= threshold).astype(int)
    tn = int(((y_true == 0) & (preds == 0)).sum())
    fp = int(((y_true == 0) & (preds == 1)).sum())
    spec = float(tn / max(tn + fp, 1))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = 0.0
    return {
        "auc": auc,
        "accuracy": float(accuracy_score(y_true, preds)),
        "sensitivity": float(recall_score(y_true, preds, zero_division=0)),
        "specificity": spec,
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "threshold": float(threshold),
    }


def node_drop_augment(graph, p=0.1):
    """Zero out random nodes' features as augmentation."""
    if p <= 0 or graph.x.shape[0] < 4:
        return graph
    new = graph.clone() if hasattr(graph, 'clone') else deepcopy(graph)
    mask = (torch.rand(graph.x.shape[0]) > p).float().unsqueeze(1)
    new.x = graph.x * mask.to(graph.x.device)
    return new


# ═══════════════════════════════════════════════════════════════════
# Single fold training
# ═══════════════════════════════════════════════════════════════════

def train_one_fold(
    model: nn.Module,
    train_data: List[dict],
    val_data: List[dict],
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    device: str = "cpu",
    use_mpap_aux: bool = False,
    mpap_weight: float = 0.1,
    node_drop_p: float = 0.1,
    patience: int = 30,
    use_airway: bool = True,
) -> dict:
    """
    Train one fold.  Returns val predictions, embeddings, attention weights.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    # Class-balanced weights
    labels_arr = np.array([d["label"] for d in train_data])
    counts = np.bincount(labels_arr, minlength=2).astype(float)
    counts = np.clip(counts, 1, None)
    w = counts.sum() / (2 * counts)
    class_w = torch.tensor(w, dtype=torch.float, device=device)
    ce_loss = nn.CrossEntropyLoss(weight=class_w)
    mse_loss = nn.MSELoss()

    best_auc = -1.0
    best_state = None
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        np.random.shuffle(train_data)

        for sample in train_data:
            art = sample["artery"].to(device)
            ven = sample["vein"].to(device)
            label_t = torch.tensor([sample["label"]], dtype=torch.long, device=device)

            # Augmentation
            if node_drop_p > 0:
                art = node_drop_augment(art, node_drop_p)
                ven = node_drop_augment(ven, node_drop_p)

            if use_airway:
                aw = sample["airway"].to(device)
                out = model(art, ven, aw)
            else:
                out = model(art, ven)

            loss = ce_loss(out["logits"], label_t)

            if use_mpap_aux and "mpap_pred" in out and sample.get("mpap") is not None:
                mpap_true = torch.tensor([[sample["mpap"]]], dtype=torch.float, device=device)
                loss = loss + mpap_weight * mse_loss(out["mpap_pred"], mpap_true)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validate every 5 epochs
        if epoch % 5 == 0 or epoch == epochs:
            val_result = evaluate(model, val_data, device, use_airway)
            auc = val_result["metrics"]["auc"]

            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1

            if stale > patience // 5:
                break

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)

    # Final eval on val set with Youden threshold
    val_result = evaluate(model, val_data, device, use_airway)
    return val_result


def evaluate(model, data, device, use_airway=True):
    """Evaluate model, return predictions + embeddings + attention weights."""
    model.eval()
    y_true, y_prob = [], []
    embeddings = []
    attn_weights_list = []
    case_ids = []

    with torch.no_grad():
        for sample in data:
            art = sample["artery"].to(device)
            ven = sample["vein"].to(device)

            if use_airway:
                aw = sample["airway"].to(device)
                out = model(art, ven, aw)
            else:
                out = model(art, ven)

            prob = F.softmax(out["logits"], dim=1)[0, 1].item()
            y_true.append(sample["label"])
            y_prob.append(prob)
            embeddings.append(out["embedding"].cpu().numpy())
            attn_weights_list.append(out["attn_weights"].cpu().numpy())
            case_ids.append(sample.get("case_id", ""))

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    thresh = youden_threshold(y_true, y_prob)
    metrics = compute_metrics(y_true, y_prob, thresh)

    return {
        "metrics": metrics,
        "y_true": y_true,
        "y_prob": y_prob,
        "embeddings": np.concatenate(embeddings, axis=0),
        "attn_weights": np.concatenate(attn_weights_list, axis=0),
        "case_ids": case_ids,
    }


# ═══════════════════════════════════════════════════════════════════
# Cross-validation
# ═══════════════════════════════════════════════════════════════════

def run_cv(
    dataset: List[dict],
    *,
    n_folds: int = 5,
    repeats: int = 3,
    epochs: int = 200,
    lr: float = 1e-3,
    hidden: int = 64,
    n_layers: int = 3,
    dropout: float = 0.3,
    device: str = "cpu",
    use_mpap_aux: bool = False,
    use_airway: bool = True,
    seed_base: int = 42,
    pool_mode: str = "mean",
):
    """
    Repeated stratified k-fold CV.

    Returns:
      - Aggregated classification metrics (mean ± std)
      - Full embedding matrix (N × D) from held-out predictions
      - Per-patient attention weights
    """
    label_arr = np.array([d["label"] for d in dataset])
    n = len(dataset)

    # Detect feature dims
    in_dim_a = dataset[0]["artery"].x.shape[1]
    in_dim_v = dataset[0]["vein"].x.shape[1]
    in_dim_w = dataset[0]["airway"].x.shape[1] if use_airway else 0

    # Accumulate across all folds
    all_fold_metrics = []
    # For shared embedding: collect val-set embeddings indexed by patient
    embed_dim = hidden
    global_embeddings = np.zeros((n, embed_dim))
    global_attn = np.zeros((n, 3 if use_airway else 2))
    global_y_prob = np.zeros(n)
    embed_counts = np.zeros(n)  # average across repeats

    for rep in range(repeats):
        seed = seed_base + rep
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        for fold, (train_idx, val_idx) in enumerate(skf.split(range(n), label_arr)):
            log.info("Rep %d/%d  Fold %d/%d", rep + 1, repeats, fold + 1, n_folds)

            train_data = [dataset[i] for i in train_idx]
            val_data = [dataset[i] for i in val_idx]

            if use_airway:
                model = TriStructureGCN(
                    in_dim_artery=in_dim_a, in_dim_vein=in_dim_v, in_dim_airway=in_dim_w,
                    hidden=hidden, n_layers=n_layers, dropout=dropout,
                    use_mpap_aux=use_mpap_aux,
                    pool=pool_mode,
                )
            else:
                model = DualStructureGCN(
                    in_dim_artery=in_dim_a, in_dim_vein=in_dim_v,
                    hidden=hidden, n_layers=n_layers, dropout=dropout,
                    use_mpap_aux=use_mpap_aux,
                )

            fold_result = train_one_fold(
                model, train_data, val_data,
                epochs=epochs, lr=lr, device=device,
                use_mpap_aux=use_mpap_aux, use_airway=use_airway,
            )

            all_fold_metrics.append(fold_result["metrics"])

            # Accumulate embeddings for shared analysis
            for i, vi in enumerate(val_idx):
                global_embeddings[vi] += fold_result["embeddings"][i]
                global_attn[vi] += fold_result["attn_weights"][i]
                global_y_prob[vi] += fold_result["y_prob"][i]
                embed_counts[vi] += 1

            m = fold_result["metrics"]
            log.info("  AUC=%.3f  Sens=%.3f  Spec=%.3f  F1=%.3f",
                     m["auc"], m["sensitivity"], m["specificity"], m["f1"])

    # Average embeddings across repeats
    mask = embed_counts > 0
    global_embeddings[mask] /= embed_counts[mask, None]
    global_attn[mask] /= embed_counts[mask, None]
    global_y_prob[mask] /= embed_counts[mask]

    # Aggregate metrics
    keys = ["auc", "accuracy", "sensitivity", "specificity", "f1", "precision"]
    mean_metrics = {}
    for k in keys:
        vals = [m[k] for m in all_fold_metrics]
        mean_metrics[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    return {
        "fold_metrics": all_fold_metrics,
        "mean_metrics": mean_metrics,
        "embeddings": global_embeddings,
        "attn_weights": global_attn,
        "y_prob": global_y_prob,
    }


# ═══════════════════════════════════════════════════════════════════
# Shared embedding analysis: clustering + phenotype discovery
# ═══════════════════════════════════════════════════════════════════

def _cluster_on_view(
    X: np.ndarray,
    labels: np.ndarray,
    mpap_values: np.ndarray,
) -> dict:
    """
    Run the standard suite of clustering algorithms on a feature matrix X.

    Returns a dict keyed by algorithm name (e.g. ``kmeans_k2``) with silhouette,
    ARI/NMI against class labels, cluster sizes, per-cluster mPAP stats, and
    ARI against mPAP tertiles when mPAP is available.
    """
    n = len(labels)
    out: dict = {}
    for name, model in [
        ("kmeans_k2", KMeans(n_clusters=2, n_init=20, random_state=42)),
        ("kmeans_k3", KMeans(n_clusters=3, n_init=20, random_state=42)),
        ("kmeans_k4", KMeans(n_clusters=4, n_init=20, random_state=42)),
        ("gmm_k2", GaussianMixture(n_components=2, n_init=5, random_state=42)),
        ("gmm_k3", GaussianMixture(n_components=3, n_init=5, random_state=42)),
    ]:
        try:
            cl = model.fit_predict(X)
            sil = float(silhouette_score(X, cl)) if len(set(cl)) > 1 else 0.0
            ari = float(adjusted_rand_score(labels, cl))
            nmi = float(normalized_mutual_info_score(labels, cl))

            entry: dict = {
                "silhouette": sil, "ari_label": ari, "nmi_label": nmi,
                "cluster_sizes": [int((cl == c).sum()) for c in sorted(set(cl))],
            }

            mpap_valid = ~np.isnan(mpap_values)
            if mpap_valid.sum() > 5:
                cluster_mpap = {}
                for c in sorted(set(cl)):
                    m = (cl == c) & mpap_valid
                    if m.sum() > 0:
                        cluster_mpap[str(c)] = {
                            "mean_mpap": float(np.mean(mpap_values[m])),
                            "std_mpap": float(np.std(mpap_values[m])),
                            "n": int(m.sum()),
                            "ph_rate": float((labels[cl == c] == 1).mean()),
                        }
                entry["cluster_mpap"] = cluster_mpap

                q33 = np.percentile(mpap_values[mpap_valid], 33.3)
                q66 = np.percentile(mpap_values[mpap_valid], 66.6)
                tert = np.zeros(n, dtype=int)
                tert[mpap_values > q33] = 1
                tert[mpap_values > q66] = 2
                entry["ari_mpap_tertile"] = float(
                    adjusted_rand_score(tert[mpap_valid], cl[mpap_valid])
                )

            out[name] = entry
        except Exception as e:
            out[name] = {"error": str(e)}
    return out


def analyse_shared_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    attn_weights: np.ndarray,
    mpap_values: np.ndarray,
    case_ids: list,
    structure_names: list = ["artery", "vein", "airway"],
    signatures: Optional[np.ndarray] = None,
) -> dict:
    """
    Analyse the shared embedding space produced by TriStructureGCN.

    Three analyses:
      1. Clustering on z_fused → phenotype discovery
         When ``signatures`` is provided, clustering is run on three views:
         ``embedding`` (z_fused only), ``signature`` (interpretable graph
         signatures only), and ``hybrid`` (standardised concat of both).
      2. Attention weight profiles → structure importance per patient
      3. Cross-reference clusters with mPAP → validate biological meaning
    """
    results: dict = {}

    # ── 1. Clustering on shared embeddings ──
    if signatures is None:
        results["clustering"] = _cluster_on_view(embeddings, labels, mpap_values)
    else:
        sig_std = StandardScaler().fit_transform(signatures)
        emb_std = StandardScaler().fit_transform(embeddings)
        hybrid = np.concatenate([emb_std, sig_std], axis=1)

        views = {
            "embedding": embeddings,
            "signature": sig_std,
            "hybrid": hybrid,
        }
        clustering_views = {
            view_name: _cluster_on_view(X, labels, mpap_values)
            for view_name, X in views.items()
        }
        results["clustering_views"] = clustering_views
        # Keep backwards-compatible ``clustering`` pointing at embedding view.
        results["clustering"] = clustering_views["embedding"]
        results["signature_feature_names"] = signature_feature_names()

    # ── 2. Attention profiles ──
    # For each patient, which structure did the model attend to most?
    dominant_structure = np.argmax(attn_weights, axis=1)  # 0=artery, 1=vein, 2=airway

    # Attention profile by class
    attn_by_class = {}
    for cls_label in [0, 1]:
        mask = labels == cls_label
        if mask.sum() > 0:
            mean_attn = attn_weights[mask].mean(axis=0)
            std_attn = attn_weights[mask].std(axis=0)
            attn_by_class[str(cls_label)] = {
                name: {"mean": float(mean_attn[i]), "std": float(std_attn[i])}
                for i, name in enumerate(structure_names[:attn_weights.shape[1]])
            }

    results["attention_profiles"] = {
        "by_class": attn_by_class,
        "dominant_structure_counts": {
            name: int((dominant_structure == i).sum())
            for i, name in enumerate(structure_names[:attn_weights.shape[1]])
        },
    }

    # ── 3. Structure dominance × mPAP correlation ──
    mpap_valid = ~np.isnan(mpap_values)
    if mpap_valid.sum() > 10:
        for i, name in enumerate(structure_names[:attn_weights.shape[1]]):
            corr = np.corrcoef(attn_weights[mpap_valid, i], mpap_values[mpap_valid])[0, 1]
            results["attention_profiles"][f"{name}_mpap_correlation"] = float(corr)

    return results


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache_dir", required=True, help="Path to cache/ with per-patient .pkl files")
    p.add_argument("--labels", required=True, help="labels_gold.csv")
    p.add_argument("--mpap", default=None, help="mpap_lookup_gold.json")
    p.add_argument("--output_dir", default="./outputs/tri_structure")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--device", default="auto")
    p.add_argument("--mpap_aux", action="store_true", help="Enable mPAP regression auxiliary")
    p.add_argument("--no_airway", action="store_true", help="Skip airway (artery+vein only)")
    p.add_argument("--pool_mode", choices=["mean", "attn", "add"], default="mean",
                   help="Per-structure pooling mode (mean/attn/add).")
    p.add_argument("--use_signature", action="store_true",
                   help="Also run clustering on interpretable graph signatures "
                        "and on a hybrid [embedding, signature] view.")
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    use_airway = not args.no_airway

    log.info("Device: %s  |  Airway: %s  |  mPAP aux: %s  |  Pool: %s  |  Signature: %s",
             device, use_airway, args.mpap_aux, args.pool_mode, args.use_signature)

    # ── Load data ──
    labels = load_labels(Path(args.labels))
    mpap_lookup = load_mpap(Path(args.mpap)) if args.mpap else {}

    dataset = load_tri_structure_dataset(Path(args.cache_dir), labels, mpap_lookup)
    if len(dataset) < 10:
        log.error("Too few cases: %d", len(dataset))
        sys.exit(1)

    dataset = normalize_per_structure(dataset, structures=("artery", "vein", "airway") if use_airway else ("artery", "vein"))

    log.info("Dataset: %d cases (%d PH, %d non-PH)",
             len(dataset),
             sum(1 for d in dataset if d["label"] == 1),
             sum(1 for d in dataset if d["label"] == 0))

    # ── Run CV with shared embedding extraction ──
    cv_result = run_cv(
        dataset,
        n_folds=args.n_folds, repeats=args.repeats, epochs=args.epochs,
        lr=args.lr, hidden=args.hidden, n_layers=args.n_layers,
        dropout=args.dropout, device=device,
        use_mpap_aux=args.mpap_aux, use_airway=use_airway,
        pool_mode=args.pool_mode,
    )

    # ── Analyse shared embeddings ──
    case_ids = [d["case_id"] for d in dataset]
    label_arr = np.array([d["label"] for d in dataset])
    mpap_arr = np.array([d.get("mpap", np.nan) or np.nan for d in dataset])

    struct_names = ["artery", "vein", "airway"] if use_airway else ["artery", "vein"]

    signatures_arr: Optional[np.ndarray] = None
    if args.use_signature:
        missing = [d.get("case_id", "?") for d in dataset if "signature" not in d]
        if missing:
            log.warning("Signature missing for %d cases (first=%s) — skipping signature view.",
                        len(missing), missing[0] if missing else None)
        else:
            signatures_arr = np.stack([d["signature"] for d in dataset]).astype(np.float32)
            log.info("Signature matrix: %s", signatures_arr.shape)

    analysis = analyse_shared_embeddings(
        embeddings=cv_result["embeddings"],
        labels=label_arr,
        attn_weights=cv_result["attn_weights"],
        mpap_values=mpap_arr,
        case_ids=case_ids,
        structure_names=struct_names,
        signatures=signatures_arr,
    )

    # ── Save everything ──
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Classification results
    cv_path = Path(args.output_dir) / "cv_results.json"
    with cv_path.open("w") as f:
        json.dump({
            "experiment": "tri_structure_pipeline",
            "config": {
                "hidden": args.hidden, "n_layers": args.n_layers,
                "dropout": args.dropout, "epochs": args.epochs,
                "n_folds": args.n_folds, "repeats": args.repeats,
                "use_airway": use_airway, "mpap_aux": args.mpap_aux,
                "pool_mode": args.pool_mode,
                "use_signature": args.use_signature,
            },
            "n_cases": len(dataset),
            "mean_metrics": cv_result["mean_metrics"],
            "fold_metrics": cv_result["fold_metrics"],
        }, f, indent=2)

    # 2. Shared embeddings (for external visualisation)
    save_payload = {
        "embeddings": cv_result["embeddings"],
        "attn_weights": cv_result["attn_weights"],
        "y_prob": cv_result["y_prob"],
        "labels": label_arr,
        "mpap": mpap_arr,
        "case_ids": np.array(case_ids),
    }
    if signatures_arr is not None:
        save_payload["signatures"] = signatures_arr
        save_payload["signature_feature_names"] = np.array(signature_feature_names())
    np.savez(Path(args.output_dir) / "shared_embeddings.npz", **save_payload)

    # 3. Clustering + attention analysis
    analysis_path = Path(args.output_dir) / "cluster_analysis.json"
    with analysis_path.open("w") as f:
        json.dump(analysis, f, indent=2, default=str)

    log.info("All outputs saved to %s", args.output_dir)

    # ── Print summary ──
    print("\n" + "=" * 65)
    print("TRI-STRUCTURE GCN — CLASSIFICATION (shared embedding)")
    print("=" * 65)
    for k in ["auc", "sensitivity", "specificity", "f1", "accuracy", "precision"]:
        m = cv_result["mean_metrics"][k]
        print(f"  {k:14s}: {m['mean']:.4f} ± {m['std']:.4f}")

    print("\n" + "=" * 65)
    print("SHARED EMBEDDING CLUSTERING")
    print("=" * 65)
    views = analysis.get("clustering_views", {"embedding": analysis["clustering"]})
    for view_name, view_clusters in views.items():
        print(f"-- view: {view_name} --")
        for name, cr in view_clusters.items():
            if "ari_label" in cr:
                print(f"  {name:14s}: ARI(label)={cr['ari_label']:+.4f}  "
                      f"NMI={cr['nmi_label']:.4f}  Sil={cr['silhouette']:.4f}  "
                      f"sizes={cr['cluster_sizes']}")

    print("\n" + "=" * 65)
    print("ATTENTION PROFILES (which structure matters?)")
    print("=" * 65)
    for cls_name, cls_label in [("COPD (non-PH)", "0"), ("COPD-PH", "1")]:
        attn = analysis["attention_profiles"]["by_class"].get(cls_label, {})
        parts = []
        for sname in struct_names:
            if sname in attn:
                parts.append(f"{sname}={attn[sname]['mean']:.3f}")
        print(f"  {cls_name:16s}: {', '.join(parts)}")

    for sname in struct_names:
        key = f"{sname}_mpap_correlation"
        if key in analysis["attention_profiles"]:
            print(f"  {sname:16s} attention × mPAP: r = {analysis['attention_profiles'][key]:.3f}")


if __name__ == "__main__":
    main()
