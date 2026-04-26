"""R26.A — Modality ablation: isolate parenchyma vs TDA vs morph contribution.

5-seed SSL ensemble (proven R25.C method) on 4 feature subsets:
  morph-only (78D)
  morph + lung (95D)
  morph + TDA (96D)
  all (145D — R25.C baseline)

Output:
  outputs/r24/r26a_modality_ablation.csv
  outputs/r24/r26a_validation.json
  outputs/figures/fig_r26a_modality_ablation.png
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
    if ph_mp.sum() < 6: return np.full(len(Z_va), 0.5)
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


def kfold_one_seed(X, y, mp, fold, seed):
    oof = np.full(len(X), np.nan)
    for f in sorted(set(fold[fold > 0].tolist())):
        tr = (fold != f) & (fold > 0); va = fold == f
        sc = RobustScaler().fit(X[tr])
        Xtr = sc.transform(X[tr]); Xva = sc.transform(X[va])
        enc = train_ssl(Xtr, latent=32, epochs=100, batch=32, lr=1e-3, seed=seed * 100 + f)
        Ztr, Zva = project(enc, Xtr), project(enc, Xva)
        oof[va] = severity(Ztr, mp[tr], Zva, y[tr])
    return oof


def ensemble_rho(X, y, mp, fold, seeds=(42, 43, 44, 45, 46)):
    oofs = [kfold_one_seed(X, y, mp, fold, s) for s in seeds]
    ens = np.nanmean(np.column_stack(oofs), axis=1)
    valid = ~np.isnan(mp)
    rho, p = spearmanr(ens[valid], mp[valid])
    per_seed = []
    for o in oofs:
        r, _ = spearmanr(o[valid], mp[valid])
        per_seed.append(float(r))
    return float(rho), float(p), per_seed


def main():
    df = pd.read_csv(EXT)
    df = df[df["is_contrast_only_subset"]].reset_index(drop=True)

    feat_cols = [c for c in df.columns if c not in
                 ("case_id", "label", "protocol", "is_contrast_only_subset",
                  "measured_mpap", "measured_mpap_flag", "fold_id")
                 and pd.api.types.is_numeric_dtype(df[c])]
    morph_cols = [c for c in feat_cols if c.startswith(("artery_", "vein_", "airway_"))
                   and "_pers" not in c]
    tda_cols = [c for c in feat_cols if "_pers" in c]
    lung_cols = [c for c in feat_cols if c not in morph_cols and c not in tda_cols]
    print(f"Feature breakdown: morph={len(morph_cols)} TDA={len(tda_cols)} lung={len(lung_cols)} = {len(morph_cols)+len(tda_cols)+len(lung_cols)} total")

    sets = {
        "morph_only": morph_cols,
        "morph_lung": morph_cols + lung_cols,
        "morph_tda": morph_cols + tda_cols,
        "all_three": morph_cols + lung_cols + tda_cols,
    }
    y = df["label"].astype(int).values
    mp = df["measured_mpap"].astype(float).values
    fold = df["fold_id"].astype(int).values

    rows = []
    for name, cols in sets.items():
        X = df[cols].fillna(0).values.astype(float)
        print(f"  {name}: dim={X.shape[1]} ...")
        rho, p, per_seed = ensemble_rho(X, y, mp, fold)
        rows.append({"feature_set": name, "n_features": X.shape[1],
                     "ensemble_rho": rho, "ensemble_p": p,
                     "per_seed_min": float(min(per_seed)),
                     "per_seed_max": float(max(per_seed)),
                     "per_seed_mean": float(np.mean(per_seed)),
                     "per_seed_std": float(np.std(per_seed)),
                     "gate_rho_ge_0.50": bool(rho >= 0.50)})
        print(f"    ρ={rho:+.3f} p={p:.2g} per-seed=[{min(per_seed):.3f}, {max(per_seed):.3f}]")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "r26a_modality_ablation.csv", index=False)
    val = {"feature_set_results": rows,
           "n_within_contrast": int(len(df)),
           "n_mpap_resolved": int(np.isfinite(mp).sum())}
    (OUT / "r26a_validation.json").write_text(json.dumps(val, indent=2), encoding="utf-8")

    # Figure: bar chart of ensemble rho per modality
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(sets.keys())
    rhos = [r["ensemble_rho"] for r in rows]
    colors = ["#94a3b8" if r < 0.50 else "#10b981" for r in rhos]
    bars = ax.bar(range(len(names)), rhos, color=colors, edgecolor="black", alpha=0.85)
    for b, r in zip(bars, rows):
        ax.errorbar(b.get_x() + b.get_width()/2, r["ensemble_rho"],
                     yerr=[[r["ensemble_rho"] - r["per_seed_min"]],
                           [r["per_seed_max"] - r["ensemble_rho"]]],
                     fmt="o", c="black", capsize=5, lw=1.5, markersize=6)
    ax.axhline(0.50, color="orange", linestyle="--", lw=1.5, label="gate ρ=0.50")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Ensemble OOF Spearman ρ vs measured mPAP")
    ax.set_title("R26.A — Modality ablation (5-seed SSL ensemble)\n"
                 "Q2 isolate parenchyma + TDA + airway contributions; bar = ensemble ρ; error = per-seed range",
                 fontsize=12)
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    for i, r in enumerate(rows):
        ax.text(i, r["ensemble_rho"] + 0.02, f"{r['ensemble_rho']:+.3f}\n(n={r['n_features']}D)",
                ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG / "fig_r26a_modality_ablation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved figure")


if __name__ == "__main__":
    main()
