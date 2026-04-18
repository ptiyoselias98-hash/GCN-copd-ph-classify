"""Three-part interpretability visualization for Sprint 2 hybrid-GCN.

Produces in outputs/viz/:
  1. group_stats.png         — PH vs non-PH distributions of 4 enhancement features.
  2. vessel_tree_samples.png — 3D vessel graphs of one PH + one non-PH case,
                               nodes coloured by local degree.
  3. saliency_trees.png      — same two cases with node colour = input-gradient
                               saliency from a fold-1 enhanced-hybrid model.

Run remotely:
    CUDA_VISIBLE_DEVICES=0 python visualize.py \
        --cache_dir ./cache \
        --radiomics ./data/copd_ph_radiomics.csv \
        --labels "/home/imss/cw/COPDnonPH COPD-PH /data/tables/labels.csv" \
        --splits "/home/imss/cw/COPDnonPH COPD-PH /data/splits/folds" \
        --output_dir outputs/viz --epochs 120
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
import pandas as pd
import torch
import torch.nn.functional as F
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from enhance_features import augment_graph
from hybrid_gcn import HybridGCN
from run_hybrid import (
    attach_radiomics, case_to_pinyin, load_labels, load_radiomics,
    load_splits, make_loaders,
)
from run_sprint2 import _first_col
from utils.graph_builder import normalize_graph_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("viz")


# ----------------------------------------------------------------------
# Data prep
# ----------------------------------------------------------------------

def build_enhanced_dataset(cache_dir, labels, rad_df, feat_cols):
    import pickle
    col_total = _first_col(rad_df, ["肺血管容积"])
    col_fractal = _first_col(rad_df, ["肺血管分形维度"]) or _first_col(rad_df, ["分形维度"])
    col_art_dens = _first_col(rad_df, ["动脉平均密度"])
    col_vein_dens = _first_col(rad_df, ["静脉平均密度"])

    pinyin_to_row = {
        str(pid).strip().lower(): row
        for pid, row in zip(rad_df["patient_id"].values, rad_df.to_dict("records"))
    }
    pinyin_to_rad = {
        str(pid).strip().lower(): vals.astype(np.float32)
        for pid, vals in zip(rad_df["patient_id"].values, rad_df[feat_cols].values)
    }

    dataset = []
    for case_id, label in labels.items():
        pkl = Path(cache_dir) / f"{case_id}.pkl"
        if not pkl.exists():
            continue
        py = case_to_pinyin(case_id)
        rad = pinyin_to_rad.get(py)
        row = pinyin_to_row.get(py)
        if rad is None or row is None:
            continue
        with pkl.open("rb") as f:
            entry = pickle.load(f)
        entry["patient_id"] = case_id
        entry["radiomics"] = rad
        entry["label"] = int(label)
        g = entry["graph"]
        pipeline_vol = entry.get("features", {}).get("vascular", {}).get("total_vessel_volume_ml")
        g_aug = augment_graph(
            g,
            commercial_total_vol_ml=float(row[col_total]) if col_total else None,
            commercial_fractal_dim=float(row[col_fractal]) if col_fractal else None,
            commercial_artery_density=float(row[col_art_dens]) if col_art_dens else None,
            commercial_vein_density=float(row[col_vein_dens]) if col_vein_dens else None,
            pipeline_total_vol_ml=float(pipeline_vol) if pipeline_vol else None,
        )
        entry["graph"] = g_aug
        entry["row"] = row
        dataset.append(entry)
    return dataset, (col_total, col_fractal, col_art_dens, col_vein_dens)


# ----------------------------------------------------------------------
# 1. Group statistics
# ----------------------------------------------------------------------

def plot_group_stats(rad_df, labels, out_path, cols):
    col_total, col_fractal, col_art_dens, col_vein_dens = cols
    label_map = {case_to_pinyin(k): v for k, v in labels.items()}
    rad_df = rad_df.copy()
    rad_df["_label"] = rad_df["patient_id"].map(label_map)
    df = rad_df.dropna(subset=["_label"]).copy()

    pretty = {
        col_total: "Total vessel volume (ml)",
        col_fractal: "Fractal dimension",
        col_art_dens: "Artery density (HU)",
        col_vein_dens: "Vein density (HU)",
    }
    cols_used = [c for c in [col_total, col_fractal, col_art_dens, col_vein_dens] if c]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, col in zip(axes, cols_used):
        grp0 = df.loc[df["_label"] == 0, col].dropna().values
        grp1 = df.loc[df["_label"] == 1, col].dropna().values
        bp = ax.boxplot([grp0, grp1], labels=["non-PH", "PH"], widths=0.55,
                        patch_artist=True, showmeans=True)
        for patch, c in zip(bp["boxes"], ["#4C9AFF", "#E5484D"]):
            patch.set_facecolor(c); patch.set_alpha(0.55)
        # scatter overlay
        for i, grp in enumerate([grp0, grp1], start=1):
            jitter = np.random.default_rng(0).normal(0, 0.04, size=len(grp))
            ax.scatter(np.full_like(grp, i) + jitter, grp, s=10, alpha=0.55,
                       color=["#0A3D91", "#8B1A1D"][i - 1])
        try:
            from scipy import stats
            _, p = stats.mannwhitneyu(grp0, grp1, alternative="two-sided")
            ax.set_title(f"{pretty[col]}\nMann-Whitney p={p:.3g}", fontsize=10)
        except Exception:
            ax.set_title(pretty[col], fontsize=10)
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.suptitle("PH vs non-PH — commercial vessel features (n={})".format(len(df)),
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out_path)


# ----------------------------------------------------------------------
# 2 & 3. 3D vessel tree rendering
# ----------------------------------------------------------------------

def render_tree(ax, pos, edge_index, node_color, title, cmap="viridis",
                colorbar_label=None):
    xs, ys, zs = pos[:, 0], pos[:, 1], pos[:, 2]
    src, dst = edge_index[0], edge_index[1]
    segs = np.stack([pos[src], pos[dst]], axis=1)  # (E, 2, 3)
    lc = Line3DCollection(segs, colors="#888888", linewidths=0.6, alpha=0.6)
    ax.add_collection3d(lc)

    nc = node_color
    lo, hi = np.percentile(nc, [2, 98]) if nc.size else (0, 1)
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    sc = ax.scatter(xs, ys, zs, c=nc, cmap=cmap, s=14, vmin=lo, vmax=hi,
                    edgecolors="none")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=18, azim=-60)
    # equal-ish axes
    for setter, arr in [(ax.set_xlim, xs), (ax.set_ylim, ys), (ax.set_zlim, zs)]:
        if arr.size:
            setter(arr.min(), arr.max())
    if colorbar_label:
        cb = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.08)
        cb.set_label(colorbar_label, fontsize=9)
    return sc


def pick_samples(dataset):
    ph = next((e for e in dataset if e["label"] == 1), None)
    nonph = next((e for e in dataset if e["label"] == 0), None)
    return ph, nonph


def plot_vessel_samples(ph, nonph, out_path):
    fig = plt.figure(figsize=(13, 6))
    for i, (entry, name) in enumerate([(nonph, "non-PH"), (ph, "PH")], start=1):
        g = entry["graph"]
        pos = g.pos.cpu().numpy()
        ei = g.edge_index.cpu().numpy()
        # node degree
        deg = np.bincount(ei[0], minlength=pos.shape[0])
        ax = fig.add_subplot(1, 2, i, projection="3d")
        render_tree(ax, pos, ei, deg, f"{name}: {entry['patient_id']}  (nodes={pos.shape[0]})",
                    cmap="viridis", colorbar_label="node degree")
    fig.suptitle("3D pulmonary vessel graphs (nodes coloured by branching degree)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out_path)


# ----------------------------------------------------------------------
# Saliency
# ----------------------------------------------------------------------

def train_quick_hybrid(dataset, train_ids, val_ids, radiomics_dim, device,
                       epochs=120, lr=1e-3, wd=5e-4):
    tl, vl, tr_labels = make_loaders(dataset, train_ids, val_ids,
                                     batch_size=8, radiomics_dim=radiomics_dim)
    # node_dim from first item
    first = next(iter(tl))
    node_dim = first.x.shape[1]
    model = HybridGCN(gcn_in=node_dim, gcn_hidden=64, radiomics_dim=radiomics_dim,
                      out_channels=2, num_layers=3, dropout=0.3, mode="hybrid").to(device)
    n1 = int((tr_labels == 1).sum()); n0 = int((tr_labels == 0).sum())
    w = torch.tensor([max(n1, 1) / max(n0, 1), 1.0], dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

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
            from sklearn.metrics import roc_auc_score
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
            logger.info("epoch %d  val AUC=%.4f", epoch, auc)
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info("best fold-1 val AUC=%.4f", best_auc)
    return model


def saliency_for_entry(model, entry, device):
    """Input-gradient saliency: |∂p(PH) / ∂x_node| averaged over features."""
    from torch_geometric.data import Batch
    g = entry["graph"]
    g = g.clone()
    g = normalize_graph_features([g])[0]
    g = attach_radiomics(g, entry["_rad_std"])
    batch = Batch.from_data_list([g]).to(device)
    batch.x.requires_grad_(True)
    model.eval()
    logits, _, _ = model(batch.x, batch.edge_index, batch.batch,
                         getattr(batch, "radiomics", None))
    prob_ph = F.softmax(logits, dim=1)[:, 1].sum()
    prob_ph.backward()
    grad = batch.x.grad.detach().abs().sum(dim=1).cpu().numpy()
    return grad


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


def plot_saliency_samples(ph, nonph, out_path):
    fig = plt.figure(figsize=(13, 6))
    for i, (entry, name) in enumerate([(nonph, "non-PH"), (ph, "PH")], start=1):
        g = entry["graph"]
        pos = g.pos.cpu().numpy()
        ei = g.edge_index.cpu().numpy()
        sal = entry["_saliency"]
        ax = fig.add_subplot(1, 2, i, projection="3d")
        render_tree(ax, pos, ei, sal, f"{name}: {entry['patient_id']}",
                    cmap="magma", colorbar_label="|∇ p(PH) / ∇ x_node|")
    fig.suptitle("Per-node saliency from enhanced-hybrid GCN (fold-1)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out_path)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", default="./cache")
    p.add_argument("--radiomics", default="./data/copd_ph_radiomics.csv")
    p.add_argument("--labels", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--output_dir", default="outputs/viz")
    p.add_argument("--epochs", type=int, default=120)
    args = p.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", device)

    labels = load_labels(args.labels)
    rad_df, feat_cols = load_radiomics(args.radiomics)
    dataset, cols = build_enhanced_dataset(args.cache_dir, labels, rad_df, feat_cols)
    logger.info("enhanced dataset: %d patients", len(dataset))

    # 1. group stats
    plot_group_stats(rad_df, labels, out / "group_stats.png", cols)

    # pick samples (first PH / non-PH)
    ph, nonph = pick_samples(dataset)
    assert ph is not None and nonph is not None

    # 2. vessel tree (degree coloring)
    plot_vessel_samples(ph, nonph, out / "vessel_tree_samples.png")

    # 3. saliency — train fold-1 quick, then compute saliency for the two samples
    folds = load_splits(args.splits)
    tr_ids, va_ids = folds[0]
    rad_dim = len(feat_cols)
    model = train_quick_hybrid(dataset, tr_ids, va_ids, rad_dim, device,
                               epochs=args.epochs)

    std_fn = compute_radiomics_standardizer(dataset, tr_ids)
    for entry in (ph, nonph):
        entry["_rad_std"] = std_fn(entry["radiomics"])
        entry["_saliency"] = saliency_for_entry(model, entry, device)

    plot_saliency_samples(ph, nonph, out / "saliency_trees.png")

    logger.info("done — outputs in %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
