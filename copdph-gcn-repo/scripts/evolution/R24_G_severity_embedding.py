"""R24.G — mPAP-anchored severity embedding (renamed from "Disease Progression Space").

Local CPU. 78-feature morph → 32D latent via:
  (a) Contrastive SSL with feature-dropout augmentation (positive pair = same case
      with different 5%-dropout views; negatives = different cases in batch)
  (b) PCA-32 baseline (mandatory side-by-side per round-3 codex)

Anchors: mPAP-stratified PH centroids (mild <25 / mod 25-35 / severe ≥35).
OOF assignment: 5-fold CV — train SSL/PCA on train fold, project val to latent,
compute Mahalanobis (Ledoit-Wolf shrinkage) to nearest anchor, normalise to
[0,1] severity-percentile.

Validation: OOF severity-percentile vs measured mPAP Spearman ρ — gate ρ≥+0.50,
SSL must beat PCA-32 by ≥0.05 ρ.

Output:
  outputs/r24/r24g_ssl_d32_oof_severity.csv
  outputs/r24/r24g_pca32_baseline_oof.csv
  outputs/r24/r24g_validation.json
  outputs/figures/fig_r24g_progression_space.png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent
COHORT = ROOT / "outputs" / "r24" / "cohort_locked_table.csv"
MORPH = ROOT / "outputs" / "r20" / "morph_unified301.csv"
OUT = ROOT / "outputs" / "r24"
FIG = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

ARTIFACTS = {"airway_n_terminals", "airway_term_per_node",
             "artery_lap_eig0", "artery_n_terminals", "artery_term_per_node",
             "vein_lap_eig0", "vein_n_terminals", "vein_term_per_node"}


class SSLEncoder(nn.Module):
    def __init__(self, in_dim, latent_dim=32, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, latent_dim))

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=-1)


def feature_dropout_aug(X, p=0.05, rng=None):
    rng = rng or np.random.default_rng()
    mask = rng.random(X.shape) > p
    return X * mask


def train_ssl(X_train, latent_dim=32, epochs=100, batch=64, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    enc = SSLEncoder(X_train.shape[1], latent_dim).cpu()
    opt = torch.optim.Adam(enc.parameters(), lr=lr)
    n = X_train.shape[0]
    for ep in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch):
            b = idx[s:s+batch]
            if len(b) < 2: continue
            x_a = feature_dropout_aug(X_train[b], 0.05, rng)
            x_b = feature_dropout_aug(X_train[b], 0.05, rng)
            xa = torch.tensor(x_a, dtype=torch.float32)
            xb = torch.tensor(x_b, dtype=torch.float32)
            za = enc(xa); zb = enc(xb)
            # InfoNCE
            logits = za @ zb.T / 0.5
            labels = torch.arange(len(b))
            loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
            opt.zero_grad(); loss.backward(); opt.step()
    enc.eval()
    return enc


def project(enc, X):
    with torch.no_grad():
        return enc(torch.tensor(X, dtype=torch.float32)).numpy()


def compute_severity_percentile(Z_train, mpap_train, Z_val, label_train,
                                  mpap_unknown_severe_proxy=35.0):
    """For each val case: Mahalanobis distance to mPAP-stratified PH anchors."""
    # Define anchors using TRAIN-fold PH cases with measured mPAP
    ph_train_mpap = (label_train == 1) & np.isfinite(mpap_train)
    if ph_train_mpap.sum() < 6:
        # Fallback: single PH centroid
        ph_idx = label_train == 1
        if ph_idx.sum() < 3:
            return np.full(len(Z_val), 0.5)
        centroid = Z_train[ph_idx].mean(0)
        cov = LedoitWolf().fit(Z_train[ph_idx]).covariance_
        try:
            inv = np.linalg.pinv(cov)
        except Exception:
            inv = np.eye(Z_train.shape[1])
        diffs = Z_val - centroid
        dists = np.sqrt(np.einsum("ij,jk,ik->i", diffs, inv, diffs).clip(0))
        # Inverse: smaller dist → higher severity-percentile
        return 1.0 - (dists - dists.min()) / (dists.max() - dists.min() + 1e-9)
    anchors = []
    anchor_mpaps = []
    mp = mpap_train[ph_train_mpap]; zp = Z_train[ph_train_mpap]
    for lo, hi in [(0, 25), (25, 35), (35, 100)]:
        m = (mp >= lo) & (mp < hi)
        if m.sum() >= 2:
            anchors.append(zp[m].mean(0))
            anchor_mpaps.append(mp[m].mean())
    if not anchors:
        anchors = [zp.mean(0)]
        anchor_mpaps = [mp.mean()]
    anchors = np.array(anchors)
    anchor_mpaps = np.array(anchor_mpaps)
    # Mahalanobis using shrinkage cov of all PH train points
    cov = LedoitWolf().fit(zp).covariance_
    try:
        inv = np.linalg.pinv(cov)
    except Exception:
        inv = np.eye(Z_train.shape[1])
    severity_score = np.zeros(len(Z_val))
    for i, z in enumerate(Z_val):
        # Weight anchors by inverse Mahalanobis distance, weighted-average mPAP
        diffs = anchors - z
        dists = np.sqrt(np.einsum("ij,jk,ik->i", diffs, inv, diffs).clip(0))
        weights = 1.0 / (dists + 1e-6)
        severity_score[i] = (weights * anchor_mpaps).sum() / weights.sum()
    # Convert to [0, 1] percentile rank within val fold
    if severity_score.std() < 1e-9:
        return np.full(len(Z_val), 0.5)
    return (severity_score - severity_score.min()) / (severity_score.max() - severity_score.min() + 1e-9)


def kfold_oof(X, y_label, mpap, fold_id, method="ssl", latent_dim=32):
    """5-fold OOF severity-percentile assignment."""
    oof = np.full(len(X), np.nan)
    valid_folds = sorted(set(fold_id[fold_id > 0].tolist()))
    for f in valid_folds:
        tr = (fold_id != f) & (fold_id > 0)
        va = fold_id == f
        sc = RobustScaler().fit(X[tr])
        X_tr = sc.transform(X[tr]); X_va = sc.transform(X[va])
        if method == "ssl":
            enc = train_ssl(X_tr, latent_dim=latent_dim, epochs=80, batch=32, seed=42 + f)
            Z_tr = project(enc, X_tr); Z_va = project(enc, X_va)
        elif method == "pca":
            pca = PCA(n_components=min(latent_dim, X_tr.shape[0] - 1, X_tr.shape[1])).fit(X_tr)
            Z_tr = pca.transform(X_tr); Z_va = pca.transform(X_va)
        oof[va] = compute_severity_percentile(Z_tr, mpap[tr], Z_va, y_label[tr])
    return oof


def main():
    cohort = pd.read_csv(COHORT)
    morph = pd.read_csv(MORPH)
    morph_no_label = morph.drop(columns=["label"], errors="ignore")
    df = cohort.merge(morph_no_label, on="case_id", how="inner")
    df = df[df["is_contrast_only_subset"]].reset_index(drop=True)
    print(f"R24.G within-contrast n={len(df)}")
    feat_cols = [c for c in morph.columns
                 if c not in ("case_id", "label", "source_cache")
                 and c not in ARTIFACTS]
    X = df[feat_cols].fillna(0).values.astype(float)
    y = df["label"].astype(int).values
    mpap = df["measured_mpap"].astype(float).values
    fold_id = df["fold_id"].astype(int).values

    print("Training SSL d=32...")
    oof_ssl = kfold_oof(X, y, mpap, fold_id, "ssl", latent_dim=32)
    print("Training PCA-32 baseline...")
    oof_pca = kfold_oof(X, y, mpap, fold_id, "pca", latent_dim=32)

    # Save CSVs
    df_ssl = df[["case_id", "label", "measured_mpap"]].copy()
    df_ssl["severity_pct"] = oof_ssl
    df_pca = df_ssl.copy(); df_pca["severity_pct"] = oof_pca
    df_ssl.to_csv(OUT / "r24g_ssl_d32_oof_severity.csv", index=False)
    df_pca.to_csv(OUT / "r24g_pca32_baseline_oof.csv", index=False)

    # Validation against measured mPAP
    val_ssl = df_ssl.dropna(subset=["measured_mpap"])
    val_pca = df_pca.dropna(subset=["measured_mpap"])
    rho_ssl, p_ssl = spearmanr(val_ssl["severity_pct"], val_ssl["measured_mpap"]) if len(val_ssl) >= 5 else (np.nan, np.nan)
    rho_pca, p_pca = spearmanr(val_pca["severity_pct"], val_pca["measured_mpap"]) if len(val_pca) >= 5 else (np.nan, np.nan)
    out = {
        "n_within_contrast": int(len(df)),
        "n_mpap_resolved": int(len(val_ssl)),
        "ssl_d32": {"oof_rho_vs_mpap": float(rho_ssl), "p": float(p_ssl)},
        "pca_d32": {"oof_rho_vs_mpap": float(rho_pca), "p": float(p_pca)},
        "ssl_minus_pca": float(rho_ssl - rho_pca),
        "gate_oof_rho_ge_0.50": bool(rho_ssl >= 0.50),
        "gate_ssl_beats_pca_by_0.05": bool(rho_ssl - rho_pca >= 0.05),
    }
    (OUT / "r24g_validation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Figure: SSL vs PCA OOF-percentile vs mPAP scatter
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (df_p, name, rho, p) in zip(axes,
                                          [(val_ssl, "SSL contrastive d=32", rho_ssl, p_ssl),
                                           (val_pca, "PCA-32 baseline", rho_pca, p_pca)]):
        sc = ax.scatter(df_p["measured_mpap"], df_p["severity_pct"],
                        c=df_p["label"], cmap="RdBu_r", s=30, alpha=0.7,
                        edgecolors="black", linewidths=0.4)
        ax.set_xlabel("measured mPAP (mmHg)")
        ax.set_ylabel("OOF severity-percentile")
        ax.set_title(f"{name}\nSpearman ρ = {rho:+.3f}, p = {p:.2g}, n = {len(df_p)}", fontsize=11)
        ax.grid(alpha=0.3)
    fig.suptitle(f"R24.G — mPAP-anchored severity embedding (within-contrast n=190)\n"
                 f"Q1 continuous trajectory + Q3 early identification + Q5 risk score; "
                 f"gate ρ≥0.50 SSL: {'PASS' if out['gate_oof_rho_ge_0.50'] else 'FAIL'}; "
                 f"SSL beats PCA by ≥0.05: {'PASS' if out['gate_ssl_beats_pca_by_0.05'] else 'FAIL'}",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out_fig = FIG / "fig_r24g_progression_space.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close()

    print(json.dumps(out, indent=2))
    print(f"saved {out_fig}")


if __name__ == "__main__":
    main()
