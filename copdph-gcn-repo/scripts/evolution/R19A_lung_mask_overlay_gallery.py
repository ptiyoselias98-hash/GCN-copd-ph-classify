"""R19.A — Lung-mask overlay gallery (R18 reviewer must-fix).

Renders representative + worst-case lung mask overlays on axial CT slices
for blinded visual QC. Two galleries:

  1. Random 16 cases from each of (legacy 282 nonph_plain, new100 nii-new100,
     legacy 282 nonph_contrast, legacy 282 ph_contrast) — representative
     coverage across cohort
  2. Worst-case repaired-mask cases by lung_repair_drop_frac > 30% (most
     aggressive HU<-300 + top-2-CC trimming)

Per case: 1 PNG = axial mid-slice + lung-mask overlay (red transparent) +
artery overlay (yellow) + vein overlay (cyan), with title showing case_id,
group, lung volume, paren_mean_HU.

Output: outputs/r19/lung_overlay_gallery_{representative,worst}.png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
OUT = ROOT / "outputs" / "r19"
OUT.mkdir(parents=True, exist_ok=True)
NII_LEGACY = Path(r"E:\桌面文件\nii格式图\nii-unified-282")
NII_NEW100 = Path(r"E:\桌面文件\nii格式图\nii-new100")
LUNG_FEATS = ROOT / "outputs" / "lung_features_v2.csv"
NEW100_FEATS = ROOT / "outputs" / "r15" / "lung_features_new100.csv"
REPAIRED = ROOT / "outputs" / "r16" / "lung_features_new100_repaired.csv"


def render_overlay(case_dir, ax, title=""):
    ct_p = case_dir / "ct.nii.gz"
    lung_p = case_dir / "lung.nii.gz"
    art_p = case_dir / "artery.nii.gz"
    vein_p = case_dir / "vein.nii.gz"
    if not ct_p.exists() or not lung_p.exists():
        ax.text(0.5, 0.5, f"missing CT/lung\n{case_dir.name[:30]}", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        return
    ct = nib.load(str(ct_p)).get_fdata()
    lung = nib.load(str(lung_p)).get_fdata() > 0.5
    if ct.shape != lung.shape:
        sl = tuple(slice(0, min(ct.shape[d], lung.shape[d])) for d in range(3))
        ct = ct[sl]; lung = lung[sl]
    # Axial mid-slice based on lung mass center (avoid empty-slice problem)
    z_marg = lung.sum(axis=(0, 1))
    z_mid = int(np.argmax(z_marg)) if z_marg.max() > 0 else ct.shape[2] // 2
    sl_ct = ct[:, :, z_mid]
    sl_lung = lung[:, :, z_mid]
    # Display CT in lung window
    ax.imshow(np.rot90(sl_ct), cmap="gray", vmin=-1000, vmax=200, aspect="equal")
    # Lung mask red transparent
    rgba_lung = np.zeros((*sl_lung.shape, 4))
    rgba_lung[sl_lung, 0] = 1.0; rgba_lung[sl_lung, 3] = 0.25
    ax.imshow(np.rot90(rgba_lung), aspect="equal")
    # Artery yellow + vein cyan
    if art_p.exists():
        try:
            art = nib.load(str(art_p)).get_fdata() > 0.5
            if art.shape == ct.shape:
                sl_a = art[:, :, z_mid]
                rgba_a = np.zeros((*sl_a.shape, 4))
                rgba_a[sl_a, 0] = 1.0; rgba_a[sl_a, 1] = 1.0; rgba_a[sl_a, 3] = 0.7
                ax.imshow(np.rot90(rgba_a), aspect="equal")
        except Exception: pass
    if vein_p.exists():
        try:
            vein = nib.load(str(vein_p)).get_fdata() > 0.5
            if vein.shape == ct.shape:
                sl_v = vein[:, :, z_mid]
                rgba_v = np.zeros((*sl_v.shape, 4))
                rgba_v[sl_v, 1] = 1.0; rgba_v[sl_v, 2] = 1.0; rgba_v[sl_v, 3] = 0.7
                ax.imshow(np.rot90(rgba_v), aspect="equal")
        except Exception: pass
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def make_grid(cases, title, out_path, cols=4, rows=4):
    n = min(len(cases), cols * rows)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    for ax in axes.flat: ax.set_visible(False)
    for i, (case_dir, label) in enumerate(cases[:n]):
        ax = axes.flat[i]; ax.set_visible(True)
        render_overlay(case_dir, ax, title=label)
    fig.suptitle(title, fontsize=14, y=1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"saved {out_path}")


def main():
    cases_legacy = []
    if LUNG_FEATS.exists():
        leg = pd.read_csv(LUNG_FEATS)
        for _, row in leg.iterrows():
            cid = row["case_id"]; vol = row.get("lung_vol_mL", 0); hu = row.get("paren_mean_HU", 0)
            cd = NII_LEGACY / cid
            if cd.exists():
                grp = "PH" if cid.startswith("ph_") else ("nonPH-c" if "contrast" else "nonPH-p")
                cases_legacy.append((cd, f"{cid[:25]}...\n{grp} V={vol:.0f}mL HU={hu:.0f}"))

    # Representative gallery: take 16 from across legacy
    rng = np.random.default_rng(42)
    if len(cases_legacy) >= 16:
        idx = rng.choice(len(cases_legacy), 16, replace=False)
        rep_cases = [cases_legacy[i] for i in idx]
        make_grid(rep_cases, "R19.A — Representative legacy lung-mask overlays (random 16)",
                  OUT / "lung_overlay_gallery_representative.png")

    # Worst-case repaired: cases where lung repair dropped >30% of voxels
    if REPAIRED.exists():
        rep = pd.read_csv(REPAIRED)
        rep["drop_frac"] = rep.get("lung_repair_drop_frac", 0)
        worst = rep.sort_values("drop_frac", ascending=False).head(16)
        worst_cases = []
        for _, row in worst.iterrows():
            cid = row["case_id"]; cd = NII_NEW100 / cid
            drop = row["drop_frac"]; vol_r = row.get("lung_vol_mL_repaired", 0)
            vol_raw = row.get("lung_vol_mL_raw", 0)
            if cd.exists():
                worst_cases.append((cd, f"{cid[:25]}...\nrepair-drop={drop:.0%}\n{vol_raw:.0f}→{vol_r:.0f}mL"))
        if worst_cases:
            make_grid(worst_cases, "R19.A — Worst-case repaired masks (top 16 by HU<-300+CC drop fraction)",
                      OUT / "lung_overlay_gallery_worst_repaired.png")


if __name__ == "__main__":
    main()
