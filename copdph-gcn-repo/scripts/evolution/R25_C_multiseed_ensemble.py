"""R25.C — Multi-seed SSL ensemble: average OOF severity-percentile across 5 seeds.

Goal: bump R25.B ρ=+0.486 over the ρ≥0.50 gate via noise reduction.

Output:
  outputs/r24/r25c_ensemble_oof.csv
  outputs/r24/r25c_validation.json
  outputs/figures/fig_r25c_ensemble.png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.preprocessing import RobustScaler
from sklearn.covariance import LedoitWolf
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent
EXT = ROOT / "outputs" / "r24" / "extended_features_212.csv"
OUT = ROOT / "outputs" / "r24"
FIG = ROOT / "outputs" / "figures"


class SSLEnc(nn.Module):
    def __init__(self, in_dim, latent=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 192), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(192, 192), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(192, latent))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def feat_dropout(X, p, rng):
    return X * (rng.random(X.shape) > p)


def train_ssl(X, latent, epochs, batch, lr, seed):
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
            za, zb = enc(xa), enc(xb)
            logits = za @ zb.T / 0.5
            labels = torch.arange(len(b))
            loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
            opt.zero_grad(); loss.backward(); opt.step()
    enc.eval(); return enc


def project(enc, X):
    with torch.no_grad():
        return enc(torch.tensor(X, dtype=torch.float32)).numpy()


def severity(Z_tr, mp_tr, Z_va, label_tr):
    ph_mp = (label_tr == 1) & np.isfinite(mp_tr)
    if ph_mp.sum() < 6:
        return np.full(len(Z_va), 0.5)
    anchors, anchor_mp = [], []
    mp = mp_tr[ph_mp]; zp = Z_tr[ph_mp]
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
    s = np.zeros(len(Z_va))
    for i, z in enumerate(Z_va):
        d = np.sqrt(np.einsum("ij,jk,ik->i", anchors - z, inv, anchors - z).clip(0))
        w = 1 / (d + 1e-6); s[i] = (w * anchor_mp).sum() / w.sum()
    if s.std() < 1e-9: return np.full(len(Z_va), 0.5)
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


def kfold_one_seed(X, y, mp, fold, seed, latent=32):
    oof = np.full(len(X), np.nan)
    for f in sorted(set(fold[fold > 0].tolist())):
        tr = (fold != f) & (fold > 0); va = fold == f
        sc = RobustScaler().fit(X[tr])
        Xtr = sc.transform(X[tr]); Xva = sc.transform(X[va])
        enc = train_ssl(Xtr, latent=latent, epochs=100, batch=32, lr=1e-3, seed=seed * 100 + f)
        Ztr, Zva = project(enc, Xtr), project(enc, Xva)
        oof[va] = severity(Ztr, mp[tr], Zva, y[tr])
    return oof


def main():
    df = pd.read_csv(EXT)
    df = df[df["is_contrast_only_subset"]].reset_index(drop=True)
    feat_cols = [c for c in df.columns if c not in
                 ("case_id", "label", "protocol", "is_contrast_only_subset",
                  "measured_mpap", "measured_mpap_flag", "fold_id")
                 and pd.api.types.is_numeric_dtype(df[c])]
    print(f"R25.C n={len(df)}, features={len(feat_cols)}")
    X = df[feat_cols].fillna(0).values.astype(float)
    y = df["label"].astype(int).values
    mp = df["measured_mpap"].astype(float).values
    fold = df["fold_id"].astype(int).values

    seeds = [42, 43, 44, 45, 46]
    seed_oofs = []
    seed_rhos = []
    for s in seeds:
        print(f"  seed {s}...")
        oof_s = kfold_one_seed(X, y, mp, fold, s, latent=32)
        seed_oofs.append(oof_s)
        valid = ~np.isnan(mp)
        rho, _ = spearmanr(oof_s[valid], mp[valid])
        seed_rhos.append(float(rho))
        print(f"    seed={s}: ρ={rho:+.3f}")

    # Ensemble: mean OOF severity across seeds
    ens_oof = np.nanmean(np.column_stack(seed_oofs), axis=1)
    valid = ~np.isnan(mp)
    rho_ens, p_ens = spearmanr(ens_oof[valid], mp[valid])
    out = {
        "n": int(len(df)),
        "n_features": len(feat_cols),
        "n_mpap_resolved": int(valid.sum()),
        "per_seed_rho": seed_rhos,
        "per_seed_mean": float(np.mean(seed_rhos)),
        "per_seed_std": float(np.std(seed_rhos)),
        "ensemble_rho": float(rho_ens),
        "ensemble_p": float(p_ens),
        "improvement_over_R25B_single_seed": float(rho_ens - 0.486),
        "gate_rho_ge_0.50": bool(rho_ens >= 0.50),
    }
    df_out = df[["case_id", "label", "measured_mpap"]].copy()
    df_out["severity_pct_ensemble"] = ens_oof
    for i, s in enumerate(seeds):
        df_out[f"sev_seed_{s}"] = seed_oofs[i]
    df_out.to_csv(OUT / "r25c_ensemble_oof.csv", index=False)
    (OUT / "r25c_validation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    val_df = df_out.dropna(subset=["measured_mpap"])
    ax = axes[0]
    ax.scatter(val_df["measured_mpap"], val_df["severity_pct_ensemble"],
               c=val_df["label"], cmap="RdBu_r", s=30, alpha=0.7,
               edgecolors="black", linewidths=0.4)
    ax.set_xlabel("measured mPAP (mmHg)"); ax.set_ylabel("OOF ensemble severity-percentile")
    ax.set_title(f"5-seed SSL ensemble\nρ = {rho_ens:+.3f}, p = {p_ens:.2g}, n = {valid.sum()}")
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.bar(range(len(seeds)), seed_rhos, color=["#3b82f6"]*len(seeds), alpha=0.8,
           edgecolor="black")
    ax.axhline(rho_ens, color="red", linestyle="--", lw=2, label=f"ensemble ρ={rho_ens:+.3f}")
    ax.axhline(0.50, color="orange", linestyle=":", lw=1, label="gate ρ=0.50")
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f"seed={s}" for s in seeds])
    ax.set_ylabel("Per-seed OOF ρ vs mPAP")
    ax.set_title(f"Single-seed range [{min(seed_rhos):.3f}, {max(seed_rhos):.3f}], "
                 f"ensemble {rho_ens:+.3f}")
    ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle(f"R25.C — Multi-seed SSL ensemble (5 seeds × 145 features)\n"
                 f"R24.G 78D ρ=+0.252 → R25.B 145D 1-seed ρ=+0.486 → R25.C 145D ens ρ={rho_ens:+.3f}; "
                 f"gate ρ≥0.50: {'PASS ✓' if out['gate_rho_ge_0.50'] else 'FAIL ✗'}", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG / "fig_r25c_ensemble.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
