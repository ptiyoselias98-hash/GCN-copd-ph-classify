"""R20.A — Local CPU DDPM inference on legacy 282 cohort.

Loads the R19-trained DDPM checkpoint and scores all 282 legacy cases
(170 PH + 112 nonPH) locally on CPU. Closes R18 must-fix #1 (DDPM eval
PH-vs-nonPH AUC) without needing legacy NIfTIs scp'd to remote.

Reuses the model definition from R19_diffusion_train.py.

Usage (local Windows):
    python scripts/evolution/R20A_local_ddpm_inference.py \\
        --checkpoint copdph-gcn-repo/outputs/r19/ddpm_state_dict.pt \\
        --n_eval_patches 8

Output: outputs/r20/ddpm_anomaly_legacy.csv (282 rows) +
        outputs/r20/ddpm_anomaly_legacy_eval.{json,md}
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent.parent
NII_LOCAL = Path(r"E:\桌面文件\nii格式图\nii-unified-282")
LABELS = ROOT / "data" / "labels_extended_382.csv"
OUT = ROOT / "outputs" / "r20"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(ROOT / "outputs" / "r19" / "ddpm_state_dict.pt"))
    p.add_argument("--n_eval_patches", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    import torch
    import torch.nn.functional as F
    import nibabel as nib
    import pandas as pd

    # Lazy-import model definition from the training script
    sys.path.insert(0, str(ROOT / "scripts" / "evolution"))
    from R19_diffusion_train import TinyUNet3D

    device = torch.device("cpu")
    model = TinyUNet3D(ch_base=32).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"loaded model from {args.checkpoint}; device=cpu")

    T = 1000
    betas = torch.linspace(1e-4, 0.02, T, device=device)
    alphas = 1 - betas; alpha_bars = torch.cumprod(alphas, dim=0)

    labels_df = pd.read_csv(LABELS)
    rows = []
    for i, lr in enumerate(labels_df.iterrows()):
        _, lr = lr
        cid = lr["case_id"]; label = int(lr["label"])
        cd = NII_LOCAL / cid
        if not cd.is_dir(): continue
        ct_p = cd / "ct.nii.gz"; lung_p = cd / "lung.nii.gz"
        if not (ct_p.exists() and lung_p.exists()): continue
        try:
            t0 = time.time()
            ct = nib.load(str(ct_p)).get_fdata().astype(np.float32)
            lung = nib.load(str(lung_p)).get_fdata() > -2000  # HU-sentinel: foreground != -2048
            if ct.shape != lung.shape:
                sl = tuple(slice(0, min(ct.shape[d], lung.shape[d])) for d in range(3))
                ct = ct[sl]; lung = lung[sl]
            coords = np.argwhere(lung)
            if coords.size == 0:
                continue
            rng = np.random.default_rng(hash(cid) & 0xffffffff)
            nlls = []
            for _ in range(args.n_eval_patches):
                center = coords[rng.integers(0, len(coords))]
                s = 16
                z = (max(0, center[0]-s), min(ct.shape[0], center[0]+s))
                y = (max(0, center[1]-s), min(ct.shape[1], center[1]+s))
                x = (max(0, center[2]-s), min(ct.shape[2], center[2]+s))
                patch = ct[z[0]:z[1], y[0]:y[1], x[0]:x[1]]
                pad = [(0, 32 - patch.shape[d]) for d in range(3)]
                patch = np.pad(patch, pad, mode="constant", constant_values=-1024)
                patch = np.clip(patch, -1024, 0)
                patch = 2.0 * (patch - (-1024)) / 1024.0 - 1.0
                x_clean = torch.from_numpy(patch[None, None]).float()
                t = torch.tensor([T // 2]).long()
                noise = torch.randn_like(x_clean)
                ab = alpha_bars[t][None, None, None, None]
                x_noisy = ab.sqrt() * x_clean + (1 - ab).sqrt() * noise
                with torch.no_grad():
                    pred = model(x_noisy, t)
                    nll = F.mse_loss(pred, noise).item()
                nlls.append(nll)
            mean_nll = float(np.mean(nlls))
            rows.append({"case_id": cid, "label": label, "mean_nll": mean_nll,
                         "n_patches": len(nlls), "wall_seconds": round(time.time()-t0, 2)})
            if (i + 1) % 25 == 0:
                print(f"  ...{len(rows)}/{i+1} | last case wall={time.time()-t0:.1f}s")
        except Exception as e:
            print(f"  [fail] {cid}: {str(e)[:120]}")
        if args.limit > 0 and len(rows) >= args.limit:
            break

    df = pd.DataFrame(rows)
    out_csv = OUT / "ddpm_anomaly_legacy.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nsaved {len(df)} rows → {out_csv}")

    out = {"n_total": int(len(df)),
           "n_ph": int((df["label"] == 1).sum()),
           "n_nonph": int((df["label"] == 0).sum())}
    if df["label"].nunique() == 2:
        from sklearn.metrics import roc_auc_score
        from scipy.stats import mannwhitneyu
        y = df["label"].values; nll = df["mean_nll"].values
        try: out["anomaly_auc"] = float(roc_auc_score(y, nll))
        except Exception: out["anomaly_auc"] = float("nan")
        ph_nll = df.loc[df["label"] == 1, "mean_nll"].values
        np_nll = df.loc[df["label"] == 0, "mean_nll"].values
        try:
            u, p = mannwhitneyu(ph_nll, np_nll, alternative="greater")
            out["mwu_ph_gt_nonph_p"] = float(p)
        except Exception: out["mwu_ph_gt_nonph_p"] = None
        out["ph_mean_nll"] = float(ph_nll.mean())
        out["nonph_mean_nll"] = float(np_nll.mean())
    (OUT / "ddpm_anomaly_legacy_eval.json").write_text(json.dumps(out, indent=2),
                                                         encoding="utf-8")
    md = ["# R20.A — DDPM anomaly evaluation on legacy 282 cohort (CPU)",
          "",
          f"Trained: R19 DDPM (nonPH-only on new100 plain-scan).",
          f"Inference: 8 random 32³ lung patches per case at t=T/2.",
          "",
          f"**Cohort**: n={out['n_total']} (PH={out.get('n_ph',0)}, nonPH={out.get('n_nonph',0)})",
          ""]
    if "anomaly_auc" in out:
        md += [f"## Anomaly AUC (PH > nonPH)",
               "",
               f"- AUC = **{out['anomaly_auc']:.3f}**",
               f"- PH mean NLL = {out['ph_mean_nll']:.4f}, nonPH mean NLL = {out['nonph_mean_nll']:.4f}",
               f"- Δ = {out['ph_mean_nll'] - out['nonph_mean_nll']:+.4f}",
               f"- MWU one-sided p = {out.get('mwu_ph_gt_nonph_p', 'NA')}"]
    (OUT / "ddpm_anomaly_legacy_eval.md").write_text("\n".join(md), encoding="utf-8")
    print(f"saved → {OUT}/ddpm_anomaly_legacy_eval.{{json,md}}")
    if "anomaly_auc" in out:
        print(f"\nAnomaly AUC = {out['anomaly_auc']:.3f}")


if __name__ == "__main__":
    main()
