"""R19 — Lung parenchyma DDPM anomaly detection (light, GPU-0 background).

Trains a small 3D DDPM on 32³ pure-parenchyma patches sampled from
nonPH cases (treated as the "normal" distribution). At inference time,
PH cases are scored by per-patch reconstruction NLL → anomaly heatmap.

Memory budget: 32³ × batch=8 × float16 ~ 2GB activations + 100M params
~12GB total → fits 1x RTX 3090 with AMP.

Run on remote GPU 0 (nohup). For the patch extractor, uses the legacy
nii-unified-282 mask path (lung-mask quality known better than the
Simple_AV_seg new100 — R16.A flagged plain-scan oversegmentation).

Outputs:
  outputs/r19/ddpm_train.log
  outputs/r19/ddpm_state_dict.pt
  outputs/r19/anomaly_scores_test.csv (after training, per-case mean NLL)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib

ROOT = Path.cwd()
NII_ROOT = Path("/home/imss/cw/GCN copdnoph copdph") / "nii-unified-282"
LABELS = ROOT / "data" / "labels_expanded_282.csv"
OUT = ROOT / "outputs" / "r19"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------- Patch extraction ----------------
class LungPatchDataset(Dataset):
    def __init__(self, case_dirs, n_patches_per_case=8, patch_size=32,
                  hu_clip=(-1024, 0), augment=True):
        self.case_dirs = case_dirs
        self.n_patches_per_case = n_patches_per_case
        self.patch_size = patch_size
        self.hu_clip = hu_clip
        self.augment = augment
        # Pre-extract all valid (case, lung-bbox) tuples
        self.entries = []
        for cd in case_dirs:
            ct_p = cd / "ct.nii.gz"; lung_p = cd / "lung.nii.gz"
            if not ct_p.exists() or not lung_p.exists(): continue
            self.entries.append((cd, ct_p, lung_p))
        print(f"[dataset] {len(self.entries)} cases")

    def __len__(self):
        return len(self.entries) * self.n_patches_per_case

    def __getitem__(self, idx):
        cd, ct_p, lung_p = self.entries[idx // self.n_patches_per_case]
        ct = nib.load(str(ct_p)).get_fdata().astype(np.float32)
        lung = nib.load(str(lung_p)).get_fdata() > 0.5
        if ct.shape != lung.shape:
            sl = tuple(slice(0, min(ct.shape[d], lung.shape[d])) for d in range(3))
            ct = ct[sl]; lung = lung[sl]
        # Sample a random voxel inside lung mask
        coords = np.argwhere(lung)
        if coords.size == 0:
            return torch.zeros(1, self.patch_size, self.patch_size, self.patch_size)
        rng = np.random.default_rng(idx)
        center = coords[rng.integers(0, len(coords))]
        s = self.patch_size // 2
        z = (max(0, center[0]-s), min(ct.shape[0], center[0]+s))
        y = (max(0, center[1]-s), min(ct.shape[1], center[1]+s))
        x = (max(0, center[2]-s), min(ct.shape[2], center[2]+s))
        patch = ct[z[0]:z[1], y[0]:y[1], x[0]:x[1]]
        # Pad to patch_size
        pad = [(0, self.patch_size - patch.shape[d]) for d in range(3)]
        patch = np.pad(patch, pad, mode="constant", constant_values=-1024)
        # Normalize HU to [-1, 1]
        patch = np.clip(patch, *self.hu_clip)
        patch = 2.0 * (patch - self.hu_clip[0]) / (self.hu_clip[1] - self.hu_clip[0]) - 1.0
        if self.augment and rng.random() < 0.5:
            patch = np.flip(patch, axis=rng.integers(0, 3))
        return torch.from_numpy(np.ascontiguousarray(patch[None])).float()


# ---------------- Tiny 3D U-Net for DDPM ----------------
class TimeEmbed(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim*2); self.fc2 = nn.Linear(dim*2, dim*2)
    def forward(self, t):
        half = self.fc1.in_features // 2
        freqs = torch.exp(-np.log(10000)*torch.arange(half, device=t.device).float()/half)
        emb = t[:, None].float() * freqs[None]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.fc2(F.silu(self.fc1(emb)))


class ResBlock3D(nn.Module):
    def __init__(self, ch, t_dim=256):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch); self.conv1 = nn.Conv3d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, ch); self.conv2 = nn.Conv3d(ch, ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, ch)
    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(t)[:, :, None, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class TinyUNet3D(nn.Module):
    def __init__(self, ch_base=32, t_dim=256):
        super().__init__()
        self.t_embed = TimeEmbed(t_dim // 2)
        self.in_conv = nn.Conv3d(1, ch_base, 3, padding=1)
        self.r1 = ResBlock3D(ch_base, t_dim)
        self.down1 = nn.Conv3d(ch_base, ch_base*2, 4, stride=2, padding=1)
        self.r2 = ResBlock3D(ch_base*2, t_dim)
        self.down2 = nn.Conv3d(ch_base*2, ch_base*4, 4, stride=2, padding=1)
        self.r3 = ResBlock3D(ch_base*4, t_dim)
        self.up1 = nn.ConvTranspose3d(ch_base*4, ch_base*2, 4, stride=2, padding=1)
        self.r4 = ResBlock3D(ch_base*2, t_dim)
        self.up2 = nn.ConvTranspose3d(ch_base*2, ch_base, 4, stride=2, padding=1)
        self.r5 = ResBlock3D(ch_base, t_dim)
        self.out_conv = nn.Conv3d(ch_base, 1, 3, padding=1)
    def forward(self, x, t):
        te = self.t_embed(t)
        h = self.in_conv(x); h = self.r1(h, te)
        h = self.down1(h); h = self.r2(h, te)
        h = self.down2(h); h = self.r3(h, te)
        h = self.up1(h); h = self.r4(h, te)
        h = self.up2(h); h = self.r5(h, te)
        return self.out_conv(h)


# ---------------- DDPM training loop ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--patch_size", type=int, default=32)
    p.add_argument("--n_patches_per_case", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_train_cases", type=int, default=80)
    args = p.parse_args()

    import pandas as pd
    labels_df = pd.read_csv(LABELS) if LABELS.exists() else None
    if labels_df is None:
        case_dirs = sorted(NII_ROOT.iterdir())[:args.max_train_cases]
    else:
        nonph_ids = labels_df.loc[labels_df["label"] == 0, "case_id"].tolist()
        case_dirs = [NII_ROOT / cid for cid in nonph_ids if (NII_ROOT / cid).is_dir()]
        case_dirs = case_dirs[:args.max_train_cases]
    print(f"[train] {len(case_dirs)} nonPH cases")
    if not case_dirs: raise SystemExit("no cases found")

    ds = LungPatchDataset(case_dirs, args.n_patches_per_case, args.patch_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
                     drop_last=True, pin_memory=True)

    device = torch.device(args.device)
    model = TinyUNet3D(ch_base=32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()

    T = 1000
    betas = torch.linspace(1e-4, 0.02, T, device=device)
    alphas = 1 - betas; alpha_bars = torch.cumprod(alphas, dim=0)

    for ep in range(args.epochs):
        t0 = time.time(); losses = []
        for x in dl:
            x = x.to(device, non_blocking=True)
            t = torch.randint(0, T, (x.size(0),), device=device).long()
            noise = torch.randn_like(x)
            ab = alpha_bars[t][:, None, None, None, None]
            x_noisy = ab.sqrt() * x + (1 - ab).sqrt() * noise
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                pred = model(x_noisy, t)
                loss = F.mse_loss(pred, noise)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            losses.append(loss.item())
        print(f"epoch {ep+1}/{args.epochs} loss={np.mean(losses):.4f} dt={time.time()-t0:.0f}s",
              flush=True)
        if (ep + 1) % 5 == 0:
            torch.save(model.state_dict(), OUT / "ddpm_state_dict.pt")

    torch.save(model.state_dict(), OUT / "ddpm_state_dict.pt")
    print(f"[done] saved → {OUT}/ddpm_state_dict.pt")


if __name__ == "__main__":
    main()
