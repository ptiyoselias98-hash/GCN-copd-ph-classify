"""Sprint 2 — 5-fold CV comparing 12D baseline vs 16D enhanced node features.

Runs three modes × two feature sets = 6 configs and writes one JSON.
Pulls commercial radiomics from data/copd_ph_radiomics.csv for per-patient
enhancement constants (density / fractal_dim / total vessel volume).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# reuse Sprint 1 helpers
from run_hybrid import (
    attach_radiomics, case_to_pinyin, full_metrics, load_labels,
    load_radiomics, load_splits, train_one_fold,
)
from enhance_features import augment_graph
from hybrid_gcn import HybridGCN  # noqa: F401 imported via train_one_fold
from utils.graph_builder import normalize_graph_features
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("sprint2")


# ---- column name fuzzy lookup (radiomics CSV has original Chinese names) ----
def _first_col(df: pd.DataFrame, needles: list[str]) -> str | None:
    for c in df.columns:
        s = str(c)
        if all(n in s for n in needles):
            return c
    return None


def build_dataset_v2(cache_dir, labels, rad_df, feat_cols, *, enhanced: bool):
    """Load cache + radiomics; optionally replace graph with 16D enhanced."""
    # pinyin -> (full row for enhancement, feature vector for hybrid)
    pinyin_to_full = {
        str(pid).strip().lower(): row
        for pid, row in rad_df.set_index("patient_id").iterrows()
    }
    pinyin_to_rad = {
        str(pid).strip().lower(): row_vals.astype(np.float32)
        for pid, row_vals in zip(
            rad_df["patient_id"].values,
            rad_df[feat_cols].values,
        )
    }

    col_total = _first_col(rad_df, ["肺血管容积"])
    col_fractal = _first_col(rad_df, ["肺血管分形维度"]) or _first_col(rad_df, ["分形维度"])
    col_art_dens = _first_col(rad_df, ["动脉平均密度"])
    col_vein_dens = _first_col(rad_df, ["静脉平均密度"])

    logger.info("enhancement cols: total=%r fractal=%r art_dens=%r vein_dens=%r",
                col_total, col_fractal, col_art_dens, col_vein_dens)

    dataset = []
    for case_id, label in labels.items():
        pkl = Path(cache_dir) / f"{case_id}.pkl"
        if not pkl.exists():
            continue
        py = case_to_pinyin(case_id)
        rad = pinyin_to_rad.get(py)
        if rad is None:
            continue
        with pkl.open("rb") as f:
            entry = pickle.load(f)
        g = entry["graph"]
        if enhanced:
            full = pinyin_to_full.get(py)
            pipeline_vol = entry.get("features", {}).get("vascular", {}).get(
                "total_vessel_volume_ml"
            )
            g = augment_graph(
                g,
                commercial_total_vol_ml=float(full[col_total]) if col_total else None,
                commercial_fractal_dim=float(full[col_fractal]) if col_fractal else None,
                commercial_artery_density=float(full[col_art_dens]) if col_art_dens else None,
                commercial_vein_density=float(full[col_vein_dens]) if col_vein_dens else None,
                pipeline_total_vol_ml=float(pipeline_vol) if pipeline_vol else None,
            )
        dataset.append({
            "patient_id": case_id,
            "graph": g,
            "radiomics": rad,
            "label": int(label),
        })
    pos = sum(1 for e in dataset if e["label"] == 1)
    logger.info("[%s] matched=%d pos=%d neg=%d node_dim=%d",
                "enhanced" if enhanced else "baseline",
                len(dataset), pos, len(dataset) - pos,
                dataset[0]["graph"].x.size(1) if dataset else -1)
    return dataset


def make_loaders(dataset, train_ids, val_ids, batch_size):
    id2e = {e["patient_id"]: e for e in dataset}
    tr = [id2e[i] for i in train_ids if i in id2e]
    va = [id2e[i] for i in val_ids if i in id2e]

    tr_graphs = normalize_graph_features([e["graph"].clone() for e in tr])
    va_graphs = normalize_graph_features([e["graph"].clone() for e in va])

    rad_tr = np.stack([e["radiomics"] for e in tr])
    rad_tr = np.nan_to_num(rad_tr, nan=0.0, posinf=0.0, neginf=0.0)
    mu = rad_tr.mean(axis=0); sd = rad_tr.std(axis=0) + 1e-6

    def std(v):
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        return ((v - mu) / sd).astype(np.float32)

    tr_items = [attach_radiomics(g, std(e["radiomics"])) for g, e in zip(tr_graphs, tr)]
    va_items = [attach_radiomics(g, std(e["radiomics"])) for g, e in zip(va_graphs, va)]

    tl = DataLoader(tr_items, batch_size=batch_size, shuffle=True)
    vl = DataLoader(va_items, batch_size=batch_size)
    tr_labels = np.array([int(e["label"]) for e in tr])
    return tl, vl, tr_labels


def run_all(dataset, folds, device, args, radiomics_dim, gcn_in):
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    out = {}
    for mode in modes:
        logger.info("=== mode=%s  gcn_in=%d ===", mode, gcn_in)
        fold_metrics, yts, prs = [], [], []
        for k, (tr, va) in enumerate(folds, 1):
            tl, vl, trl = make_loaders(dataset, tr, va, args.batch_size)
            # train_one_fold uses hardcoded gcn_in=12 — patch by wrapping
            m, yt, pr = _train_fold(mode, radiomics_dim, gcn_in, tl, vl, trl,
                                    device, args.epochs, args.lr, args.wd)
            logger.info("  fold %d AUC=%.4f Acc=%.4f F1=%.4f Prec=%.4f Sens=%.4f Spec=%.4f",
                        k, m["AUC"], m["Accuracy"], m["F1"], m["Precision"],
                        m["Sensitivity"], m["Specificity"])
            fold_metrics.append(m); yts += yt; prs += pr

        keys = list(fold_metrics[0].keys())
        mean = {k: float(np.mean([fm[k] for fm in fold_metrics])) for k in keys}
        std = {k: float(np.std([fm[k] for fm in fold_metrics])) for k in keys}
        pooled = float(roc_auc_score(yts, prs)) if len(set(yts)) > 1 else 0.0
        out[mode] = {"folds": fold_metrics, "mean": mean, "std": std,
                     "pooled_AUC": pooled}
    return out


def _train_fold(mode, radiomics_dim, gcn_in, tl, vl, trl, device, epochs, lr, wd,
                patience=40):
    import torch.nn.functional as F
    model = HybridGCN(gcn_in=gcn_in, gcn_hidden=64, radiomics_dim=radiomics_dim,
                      num_layers=3, dropout=0.3, mode=mode).to(device)
    n1 = int((trl == 1).sum()); n0 = int((trl == 0).sum())
    w = torch.tensor([max(n1, 1) / max(n0, 1), 1.0], dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_auc = -1.0; best_state = None; bad = 0
    for _ in range(epochs):
        model.train()
        for batch in tl:
            batch = batch.to(device)
            logits, _, _ = model(batch.x, batch.edge_index, batch.batch,
                                 getattr(batch, "radiomics", None))
            loss = F.cross_entropy(logits, batch.y.view(-1), weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
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
            best_auc = auc; bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", default="./cache")
    p.add_argument("--radiomics", default="./data/copd_ph_radiomics.csv")
    p.add_argument("--labels", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--output_dir", default="./outputs/sprint2_enhanced")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--modes", default="radiomics_only,gcn_only,hybrid")
    p.add_argument("--feature_sets", default="baseline,enhanced")
    args = p.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    labels = load_labels(args.labels)
    folds = load_splits(args.splits)
    rad_df, feat_cols = load_radiomics(args.radiomics)

    all_results = {}
    for fs in [s.strip() for s in args.feature_sets.split(",") if s.strip()]:
        enhanced = fs == "enhanced"
        dataset = build_dataset_v2(args.cache_dir, labels, rad_df, feat_cols,
                                   enhanced=enhanced)
        id_set = {e["patient_id"] for e in dataset}
        filtered = [([i for i in tr if i in id_set], [i for i in va if i in id_set])
                    for tr, va in folds]
        gcn_in = 16 if enhanced else 12
        all_results[fs] = run_all(dataset, filtered, device, args,
                                  len(feat_cols), gcn_in)

    with (out / "sprint2_results.json").open("w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("saved %s", out / "sprint2_results.json")

    # summary
    print(f"\n{'='*110}")
    print("SPRINT 2 SUMMARY (feat_set / mode → AUC Acc Prec Sens F1 Spec)")
    print("="*110)
    keys = ["AUC", "Accuracy", "Precision", "Sensitivity", "F1", "Specificity"]
    for fs, modes in all_results.items():
        for mode, r in modes.items():
            m, s = r["mean"], r["std"]
            print(f"{fs:<10} {mode:<16} "
                  + " ".join(f"{m[k]:.4f}±{s[k]:.4f}" for k in keys)
                  + f"  pooled={r['pooled_AUC']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
