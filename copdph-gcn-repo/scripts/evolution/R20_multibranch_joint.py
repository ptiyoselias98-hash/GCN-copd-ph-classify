"""R20 — Multi-branch joint model (artery + vein + airway + parenchyma).

4 modality branches on per-structure feature vectors:
  - artery branch:  44 features (R17 morph)
  - vein branch:    44 features (R17 morph)
  - airway branch:  44 features (R17 morph)
  - parenchyma branch: ~40 features (R16 lung_features_v2)

Each branch: MLP 2-layer → 64-D embedding.
Fusion: multi-head self-attention over [a, v, w, p] tokens → fused 64-D.
Heads:
  - disease: 2-class softmax (PH vs nonPH)
  - endotype: 5-stage mPAP regression (continuous, MSE)
  - protocol-adv: 2-class softmax + GRL (gradient reversal) on encoder

Loss: BCE_disease + λ_endo * MSE_mpap + λ_adv * GRL_protocol_CE.

Run on remote GPU 1 (parallel to GPU 0 DDPM).
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path.cwd()
MORPH = ROOT / "outputs" / "r17" / "per_structure_morphometrics.csv"
LUNG = ROOT / "outputs" / "lung_features_v2.csv"
LABELS = ROOT / "data" / "labels_extended_382.csv"
PROTO = ROOT / "data" / "case_protocol_extended.csv"
SEG_FAILS = ROOT / "outputs" / "r13" / "seg_failures_real.json"
MPAP = ROOT / "data" / "mpap_lookup_gold.json"
OUT = ROOT / "outputs" / "r20"
OUT.mkdir(parents=True, exist_ok=True)


# ---- GRL ----
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd); return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grl(x, lambd): return _GradReverse.apply(x, lambd)


class BranchEncoder(nn.Module):
    def __init__(self, in_dim, hidden=128, out=64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden); self.fc2 = nn.Linear(hidden, out)
        self.bn = nn.BatchNorm1d(hidden); self.drop = nn.Dropout(0.3)
    def forward(self, x):
        x = F.silu(self.bn(self.fc1(x))); x = self.drop(x)
        return self.fc2(x)


class MultibranchJoint(nn.Module):
    def __init__(self, dims, n_heads=4, fused_dim=64):
        super().__init__()
        self.branches = nn.ModuleDict({k: BranchEncoder(d, out=fused_dim) for k, d in dims.items()})
        self.attn = nn.MultiheadAttention(fused_dim, num_heads=n_heads, batch_first=True)
        self.fuse = nn.Linear(fused_dim, fused_dim)
        self.disease_head = nn.Linear(fused_dim, 2)
        self.mpap_head = nn.Linear(fused_dim, 1)
        self.adv_head = nn.Sequential(nn.Linear(fused_dim, 32), nn.ReLU(), nn.Linear(32, 2))
    def forward(self, batch_x: dict, adv_lambda=0.0):
        tokens = [self.branches[k](batch_x[k]) for k in batch_x]
        T = torch.stack(tokens, dim=1)  # (B, n_modal, fused_dim)
        attn_out, attn_w = self.attn(T, T, T, need_weights=True)
        z = self.fuse(attn_out.mean(dim=1))
        disease_logits = self.disease_head(z)
        mpap_pred = self.mpap_head(z).squeeze(-1)
        adv_in = grl(z, adv_lambda)
        adv_logits = self.adv_head(adv_in)
        return {"disease": disease_logits, "mpap": mpap_pred,
                "adv": adv_logits, "z": z, "attn": attn_w}


class JointDataset(Dataset):
    def __init__(self, df, feature_groups):
        self.df = df.reset_index(drop=True); self.fg = feature_groups
        self.x_cache = {k: torch.from_numpy(df[v].values.astype(np.float32))
                         for k, v in feature_groups.items()}
        self.y_disease = torch.from_numpy(df["label"].values.astype(np.int64))
        self.y_mpap = torch.from_numpy(df["mpap"].values.astype(np.float32))
        self.y_adv = torch.from_numpy(df["is_contrast"].values.astype(np.int64))
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        return ({k: v[i] for k, v in self.x_cache.items()},
                self.y_disease[i], self.y_mpap[i], self.y_adv[i])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_endo", type=float, default=0.1)
    p.add_argument("--lambda_adv", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--within_contrast", action="store_true",
                   help="Restrict to contrast cohort (no protocol confound)")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    morph = pd.read_csv(MORPH); lung = pd.read_csv(LUNG)
    lab = pd.read_csv(LABELS); pro = pd.read_csv(PROTO)
    df = lab.merge(pro[["case_id", "protocol"]], on="case_id") \
        .merge(morph, on="case_id", suffixes=("", "_dup1")) \
        .merge(lung, on="case_id", how="left", suffixes=("", "_dup2"))
    df = df.loc[:, ~df.columns.str.contains("_dup")].copy()
    fails = set()
    if SEG_FAILS.exists():
        sf = json.loads(SEG_FAILS.read_text(encoding="utf-8"))
        fails = {r["case_id"] for r in sf.get("real_fails", []) + sf.get("lung_anomaly", [])}
    df = df[~df["case_id"].isin(fails)].copy()
    mpap = json.loads(MPAP.read_text(encoding="utf-8"))
    df["mpap"] = df["case_id"].map(mpap)
    df.loc[df["protocol"].str.lower() == "plain_scan", "mpap"] = 5.0
    df.loc[(df["label"] == 0) & (df["protocol"].str.lower() == "contrast"), "mpap"] = 15.0
    df["is_contrast"] = (df["protocol"].str.lower() == "contrast").astype(int)
    df = df.dropna(subset=["mpap"]).copy()
    if args.within_contrast:
        df = df[df["protocol"].str.lower() == "contrast"].copy()
        print(f"WITHIN-CONTRAST RESTRICTION: n={len(df)} (PH={int((df['label']==1).sum())} nonPH={int((df['label']==0).sum())})")

    feat_groups = {
        "artery": [c for c in df.columns if c.startswith("artery_") and pd.api.types.is_numeric_dtype(df[c]) and "persH" not in c and c != "artery_vol_mL"],
        "vein": [c for c in df.columns if c.startswith("vein_") and pd.api.types.is_numeric_dtype(df[c]) and "persH" not in c and c != "vein_vol_mL"],
        "airway": [c for c in df.columns if c.startswith("airway_") and pd.api.types.is_numeric_dtype(df[c]) and "persH" not in c and c != "airway_vol_mL"],
        "paren": [c for c in df.columns if c.startswith(("paren_", "whole_")) and pd.api.types.is_numeric_dtype(df[c])],
    }
    for k, v in feat_groups.items():
        print(f"  {k}: {len(v)} features")
    df = df.dropna(subset=sum(feat_groups.values(), [])).copy()
    print(f"final cohort n={len(df)}")

    # Standardize features
    from sklearn.preprocessing import StandardScaler
    sc_dict = {}
    for k, cols in feat_groups.items():
        sc = StandardScaler(); df[cols] = sc.fit_transform(df[cols].values)
        sc_dict[k] = sc

    # 5-fold disease CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    aucs = []
    fold_results = []
    for k_fold, (tr_idx, te_idx) in enumerate(skf.split(df, df["label"])):
        tr_df = df.iloc[tr_idx]; te_df = df.iloc[te_idx]
        tr_ds = JointDataset(tr_df, feat_groups); te_ds = JointDataset(te_df, feat_groups)
        tl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        vl = DataLoader(te_ds, batch_size=args.batch_size)

        dims = {k: len(v) for k, v in feat_groups.items()}
        model = MultibranchJoint(dims).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        crit_disease = nn.CrossEntropyLoss()
        crit_mpap = nn.MSELoss()
        crit_adv = nn.CrossEntropyLoss()
        best_auc = 0
        for ep in range(args.epochs):
            model.train(); train_losses = []
            for x_dict, yd, ym, ya in tl:
                x_dict = {k: v.to(device) for k, v in x_dict.items()}
                yd = yd.to(device); ym = ym.to(device); ya = ya.to(device)
                opt.zero_grad()
                out = model(x_dict, adv_lambda=args.lambda_adv)
                loss = crit_disease(out["disease"], yd) \
                     + args.lambda_endo * crit_mpap(out["mpap"], ym) \
                     + args.lambda_adv * crit_adv(out["adv"], ya)
                loss.backward(); opt.step()
                train_losses.append(loss.item())
            # Val
            model.eval(); ys, probs = [], []
            with torch.no_grad():
                for x_dict, yd, ym, ya in vl:
                    x_dict = {k: v.to(device) for k, v in x_dict.items()}
                    out = model(x_dict)
                    p_disease = F.softmax(out["disease"], dim=1)[:, 1].cpu().numpy()
                    ys += yd.numpy().tolist(); probs += p_disease.tolist()
            auc = roc_auc_score(ys, probs) if len(set(ys)) > 1 else 0.0
            if auc > best_auc: best_auc = auc
        aucs.append(best_auc)
        fold_results.append({"fold": k_fold + 1, "best_auc": float(best_auc),
                             "n_train": int(len(tr_df)), "n_val": int(len(te_df))})
        print(f"  fold {k_fold+1}: best_auc={best_auc:.4f}")

    summary = {
        "config": vars(args),
        "n_total": int(len(df)),
        "feature_dims": {k: len(v) for k, v in feat_groups.items()},
        "fold_aucs": [float(a) for a in aucs],
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        "fold_results": fold_results,
    }
    (OUT / "r20_results.json").write_text(json.dumps(summary, indent=2),
                                             encoding="utf-8")
    print(f"\n[done] mean AUC = {summary['mean_auc']:.4f} ± {summary['std_auc']:.4f}")
    print(f"saved → {OUT}/r20_results.json")


if __name__ == "__main__":
    main()
