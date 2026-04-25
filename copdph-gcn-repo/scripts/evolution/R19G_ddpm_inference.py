"""R19.G — DDPM inference: per-case anomaly score on PH vs nonPH.

After R19 training completes, load `outputs/r19/ddpm_state_dict.pt` and
score each test patch's reconstruction NLL. Per-case mean NLL serves as
anomaly score; ROC of nonPH (training distribution) vs PH (anomalous)
quantifies whether the diffusion model captures PH-specific parenchyma
deviation.

Usage on remote:
    CUDA_VISIBLE_DEVICES=0 python _R19G_ddpm_inference.py \\
        --checkpoint outputs/r19/ddpm_state_dict.pt \\
        --n_eval_patches 16

Outputs:
  outputs/r19/ddpm_anomaly_scores.csv  (case_id, label, mean_nll, n_patches)
  outputs/r19/ddpm_anomaly_eval.{json,md}
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu

# Reuse model definitions from R19_diffusion_train.py
import sys
sys.path.insert(0, str(Path.cwd()))
from _R19_DDPM import TinyUNet3D, LungPatchDataset

ROOT = Path.cwd()
NII_NEW100 = ROOT / "nii-new100"
NII_LEGACY = Path("/home/imss/cw/GCN copdnoph copdph") / "nii-unified-282"
LABELS = ROOT / "data" / "labels_extended_382.csv"
OUT = ROOT / "outputs" / "r19"


def score_case(model, case_dir, device, n_patches=16, T=1000, betas_device=None):
    """Score one case: mean reconstruction NLL across n_patches sampled patches."""
    import nibabel as nib
    ct_p = case_dir / "ct.nii.gz"; lung_p = case_dir / "lung.nii.gz"
    if not ct_p.exists() or not lung_p.exists():
        return None
    ct = nib.load(str(ct_p)).get_fdata().astype(np.float32)
    lung = nib.load(str(lung_p)).get_fdata() > 0.5
    if ct.shape != lung.shape:
        sl = tuple(slice(0, min(ct.shape[d], lung.shape[d])) for d in range(3))
        ct = ct[sl]; lung = lung[sl]
    coords = np.argwhere(lung)
    if coords.size == 0:
        return None
    rng = np.random.default_rng(hash(case_dir.name) & 0xffffffff)
    nlls = []
    for _ in range(n_patches):
        center = coords[rng.integers(0, len(coords))]
        s = 16  # patch_size//2
        z = (max(0, center[0]-s), min(ct.shape[0], center[0]+s))
        y = (max(0, center[1]-s), min(ct.shape[1], center[1]+s))
        x = (max(0, center[2]-s), min(ct.shape[2], center[2]+s))
        patch = ct[z[0]:z[1], y[0]:y[1], x[0]:x[1]]
        pad = [(0, 32 - patch.shape[d]) for d in range(3)]
        patch = np.pad(patch, pad, mode="constant", constant_values=-1024)
        patch = np.clip(patch, -1024, 0)
        patch = 2.0 * (patch - (-1024)) / 1024.0 - 1.0
        x_clean = torch.from_numpy(patch[None, None]).float().to(device)
        # Add noise at random t and predict — NLL = mean((noise_pred - noise)^2)
        t = torch.tensor([T // 2], device=device).long()
        noise = torch.randn_like(x_clean)
        ab = betas_device[t][None, None, None, None]
        x_noisy = ab.sqrt() * x_clean + (1 - ab).sqrt() * noise
        with torch.no_grad():
            pred = model(x_noisy, t)
            nll = F.mse_loss(pred, noise).item()
        nlls.append(nll)
    return float(np.mean(nlls)), int(len(nlls))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/r19/ddpm_state_dict.pt")
    p.add_argument("--n_eval_patches", type=int, default=16)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    import pandas as pd
    device = torch.device(args.device)
    model = TinyUNet3D(ch_base=32).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    T = 1000
    betas = torch.linspace(1e-4, 0.02, T, device=device)
    alphas = 1 - betas; alpha_bars = torch.cumprod(alphas, dim=0)

    labels_df = pd.read_csv(LABELS)
    rows = []
    for _, lr in labels_df.iterrows():
        cid = lr["case_id"]; label = int(lr["label"])
        # Try new100 first then legacy
        cd = NII_NEW100 / cid if (NII_NEW100 / cid).is_dir() else NII_LEGACY / cid
        if not cd.is_dir(): continue
        try:
            r = score_case(model, cd, device, args.n_eval_patches, T, alpha_bars)
        except Exception as e:
            r = None
        if r is None: continue
        nll, n = r
        rows.append({"case_id": cid, "label": label, "mean_nll": nll, "n_patches": n})
        if len(rows) % 25 == 0: print(f"  ...{len(rows)} done")

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ddpm_anomaly_scores.csv", index=False)
    print(f"saved {len(df)} rows → ddpm_anomaly_scores.csv")

    # Eval — anomaly AUC PH vs nonPH on lung-window-only nonPH-trained model
    out = {"n": int(len(df)),
           "n_ph": int((df["label"] == 1).sum()),
           "n_nonph": int((df["label"] == 0).sum())}
    if df["label"].nunique() == 2 and len(df) >= 30:
        y = df["label"].values; nll = df["mean_nll"].values
        try:
            auc = roc_auc_score(y, nll)
            out["anomaly_auc"] = float(auc)
        except Exception: out["anomaly_auc"] = float("nan")
        ph_nll = df.loc[df["label"] == 1, "mean_nll"].values
        np_nll = df.loc[df["label"] == 0, "mean_nll"].values
        try:
            u, p = mannwhitneyu(ph_nll, np_nll, alternative="greater")
            out["mwu_ph_gt_nonph_p"] = float(p)
        except Exception: out["mwu_ph_gt_nonph_p"] = None
        out["ph_mean_nll"] = float(ph_nll.mean())
        out["nonph_mean_nll"] = float(np_nll.mean())
        out["delta_nll"] = float(ph_nll.mean() - np_nll.mean())

    (OUT / "ddpm_anomaly_eval.json").write_text(json.dumps(out, indent=2),
                                                   encoding="utf-8")
    md = ["# R19.G — DDPM anomaly evaluation",
          "",
          f"Model: nonPH-trained DDPM (R19) at checkpoint {args.checkpoint}.",
          f"Inference: per-case mean MSE-noise-prediction NLL on {args.n_eval_patches} random lung patches at t=T/2.",
          "",
          f"**Cohort**: n={out['n']} (PH={out.get('n_ph',0)}, nonPH={out.get('n_nonph',0)})",
          ""]
    if "anomaly_auc" in out:
        md += [f"## Anomaly AUC (PH vs nonPH)",
               "",
               f"- AUC = **{out['anomaly_auc']:.3f}**",
               f"- PH mean NLL = {out['ph_mean_nll']:.4f}, nonPH mean NLL = {out['nonph_mean_nll']:.4f}",
               f"- Δ = {out['delta_nll']:+.4f}",
               f"- MWU one-sided (PH > nonPH) p = {out.get('mwu_ph_gt_nonph_p', 'NA'):.3g}" if isinstance(out.get('mwu_ph_gt_nonph_p'), float) else "",
               "",
               "Higher NLL = more 'unusual' by the nonPH-learned distribution.",
               "AUC > 0.6 with PH > nonPH NLL means the model captures",
               "PH-specific parenchyma anomalies."]
    (OUT / "ddpm_anomaly_eval.md").write_text("\n".join(md), encoding="utf-8")
    print(f"saved → {OUT}/ddpm_anomaly_eval.{{json,md}}")
    if "anomaly_auc" in out:
        print(f"\nAnomaly AUC = {out['anomaly_auc']:.3f} (PH vs nonPH)")


if __name__ == "__main__":
    main()
