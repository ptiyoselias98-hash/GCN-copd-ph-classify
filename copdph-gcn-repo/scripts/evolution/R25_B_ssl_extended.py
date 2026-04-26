"""R25.B — Re-run R24.G SSL on extended 148D feature set (morph + lung + TDA).

Goal: improve OOF severity ρ_mPAP from R24.G's +0.252 toward ρ≥0.50 gate.

Output:
  outputs/r24/r25b_ssl_extended_oof.csv
  outputs/r24/r25b_pca_extended_oof.csv
  outputs/r24/r25b_validation.json
  outputs/figures/fig_r25b_ssl_extended.png
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
EXT = ROOT / "outputs" / "r24" / "extended_features_212.csv"
OUT = ROOT / "outputs" / "r24"
FIG = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)


class SSLEnc(nn.Module):
    def __init__(self, in_dim, latent=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 192), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(192, 192), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(192, latent))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def feat_dropout(X, p=0.05, rng=None):
    rng = rng or np.random.default_rng()
    return X * (rng.random(X.shape) > p)


def train_ssl(X, latent=32, epochs=120, batch=64, lr=1e-3, seed=42):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    enc = SSLEnc(X.shape[1], latent).cpu()
    opt = torch.optim.Adam(enc.parameters(), lr=lr)
    n = X.shape[0]
    for _ in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch):
            b = idx[s:s+batch]
            if len(b) < 2: continue
            xa = torch.tensor(feat_dropout(X[b], 0.05, rng), dtype=torch.float32)
            xb = torch.tensor(feat_dropout(X[b], 0.05, rng), dtype=torch.float32)
            za = enc(xa); zb = enc(xb)
            logits = za @ zb.T / 0.5
            labels = torch.arange(len(b))
            loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
            opt.zero_grad(); loss.backward(); opt.step()
    enc.eval(); return enc


def project(enc, X):
    with torch.no_grad():
        return enc(torch.tensor(X, dtype=torch.float32)).numpy()


def severity_pct(Z_tr, mpap_tr, Z_va, label_tr):
    ph_mp = (label_tr == 1) & np.isfinite(mpap_tr)
    if ph_mp.sum() < 6:
        ph = label_tr == 1
        if ph.sum() < 3: return np.full(len(Z_va), 0.5)
        ctr = Z_tr[ph].mean(0)
        cov = LedoitWolf().fit(Z_tr[ph]).covariance_
        try: inv = np.linalg.pinv(cov)
        except Exception: inv = np.eye(Z_tr.shape[1])
        d = np.sqrt(np.einsum("ij,jk,ik->i", Z_va - ctr, inv, Z_va - ctr).clip(0))
        return 1 - (d - d.min()) / (d.max() - d.min() + 1e-9)
    anchors, anchor_mp = [], []
    mp = mpap_tr[ph_mp]; zp = Z_tr[ph_mp]
    for lo, hi in [(0, 25), (25, 35), (35, 100)]:
        m = (mp >= lo) & (mp < hi)
        if m.sum() >= 2:
            anchors.append(zp[m].mean(0)); anchor_mp.append(mp[m].mean())
    if not anchors:
        anchors = [zp.mean(0)]; anchor_mp = [mp.mean()]
    anchors = np.array(anchors); anchor_mp = np.array(anchor_mp)
    cov = LedoitWolf().fit(zp).covariance_
    try: inv = np.linalg.pinv(cov)
    except Exception: inv = np.eye(Z_tr.shape[1])
    score = np.zeros(len(Z_va))
    for i, z in enumerate(Z_va):
        d = np.sqrt(np.einsum("ij,jk,ik->i", anchors - z, inv, anchors - z).clip(0))
        w = 1 / (d + 1e-6)
        score[i] = (w * anchor_mp).sum() / w.sum()
    if score.std() < 1e-9: return np.full(len(Z_va), 0.5)
    return (score - score.min()) / (score.max() - score.min() + 1e-9)


def kfold_oof(X, y, mp, fold, method="ssl", latent=32):
    oof = np.full(len(X), np.nan)
    for f in sorted(set(fold[fold > 0].tolist())):
        tr = (fold != f) & (fold > 0); va = fold == f
        sc = RobustScaler().fit(X[tr])
        Xtr = sc.transform(X[tr]); Xva = sc.transform(X[va])
        if method == "ssl":
            enc = train_ssl(Xtr, latent=latent, epochs=100, batch=32, seed=42 + f)
            Ztr, Zva = project(enc, Xtr), project(enc, Xva)
        else:
            pca = PCA(n_components=min(latent, Xtr.shape[0]-1, Xtr.shape[1])).fit(Xtr)
            Ztr, Zva = pca.transform(Xtr), pca.transform(Xva)
        oof[va] = severity_pct(Ztr, mp[tr], Zva, y[tr])
    return oof


def main():
    df = pd.read_csv(EXT)
    df = df[df["is_contrast_only_subset"]].reset_index(drop=True)
    feat_cols = [c for c in df.columns if c not in
                 ("case_id", "label", "protocol", "is_contrast_only_subset",
                  "measured_mpap", "measured_mpap_flag", "fold_id")
                 and pd.api.types.is_numeric_dtype(df[c])]
    print(f"R25.B within-contrast n={len(df)}, numeric features={len(feat_cols)}")
    X = df[feat_cols].fillna(0).values.astype(float)
    y = df["label"].astype(int).values
    mp = df["measured_mpap"].astype(float).values
    fold = df["fold_id"].astype(int).values

    print("SSL d=32 extended..."); oof_ssl = kfold_oof(X, y, mp, fold, "ssl", 32)
    print("PCA-32 baseline..."); oof_pca = kfold_oof(X, y, mp, fold, "pca", 32)

    df_ssl = df[["case_id", "label", "measured_mpap"]].copy()
    df_ssl["severity_pct"] = oof_ssl
    df_pca = df_ssl.copy(); df_pca["severity_pct"] = oof_pca
    df_ssl.to_csv(OUT / "r25b_ssl_extended_oof.csv", index=False)
    df_pca.to_csv(OUT / "r25b_pca_extended_oof.csv", index=False)

    val_ssl = df_ssl.dropna(subset=["measured_mpap"])
    val_pca = df_pca.dropna(subset=["measured_mpap"])
    rho_ssl, p_ssl = spearmanr(val_ssl.severity_pct, val_ssl.measured_mpap)
    rho_pca, p_pca = spearmanr(val_pca.severity_pct, val_pca.measured_mpap)
    out = {
        "n_within_contrast": int(len(df)),
        "n_features": len(feat_cols),
        "n_mpap_resolved": int(len(val_ssl)),
        "ssl_rho": float(rho_ssl), "ssl_p": float(p_ssl),
        "pca_rho": float(rho_pca), "pca_p": float(p_pca),
        "ssl_minus_pca": float(rho_ssl - rho_pca),
        "gate_rho_ge_0.50": bool(rho_ssl >= 0.50),
        "gate_ssl_beats_pca_0.05": bool(rho_ssl - rho_pca >= 0.05),
        "improvement_vs_R24G_78D": float(rho_ssl - 0.252),
    }
    (OUT / "r25b_validation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (df_p, name, rho, p) in zip(axes,
        [(val_ssl, f"SSL d=32 extended ({len(feat_cols)} feats)", rho_ssl, p_ssl),
         (val_pca, "PCA-32 baseline", rho_pca, p_pca)]):
        ax.scatter(df_p.measured_mpap, df_p.severity_pct, c=df_p.label,
                   cmap="RdBu_r", s=30, alpha=0.7, edgecolors="black", linewidths=0.4)
        ax.set_xlabel("measured mPAP (mmHg)"); ax.set_ylabel("OOF severity-percentile")
        ax.set_title(f"{name}\nρ = {rho:+.3f}, p = {p:.2g}, n = {len(df_p)}", fontsize=11)
        ax.grid(alpha=0.3)
    fig.suptitle(f"R25.B — Extended 148D SSL (morph+lung+TDA) within-contrast n={len(df)}\n"
                 f"R24.G 78D ρ=+0.252 → R25.B 148D ρ={rho_ssl:+.3f}; gate ρ≥0.50: "
                 f"{'PASS ✓' if out['gate_rho_ge_0.50'] else 'FAIL ✗'}", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG / "fig_r25b_ssl_extended.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
