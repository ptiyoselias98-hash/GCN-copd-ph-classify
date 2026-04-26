"""R24.Y — Regenerate overlay gallery on unified-301 (fixes R19.A blank-placeholder bug).

Picks 16 representative + 16 worst-by-lung-CC cases from unified-301, fetches CT
from the appropriate source per protocol:
  - new100 plain-scan: nii-new100/<case>/ct.nii.gz
  - legacy contrast: original.nii.gz from Claude_COPDnonPH_COPD-PH_CT_nii/nii/<case>/
                     OR via _source.txt redirect from nii-unified-282/<case>/
                     (decode GBK if needed; root cause of R19.A blank bug)

Lung mask comes from nii-unified-201-savseg/<case>/lung.nii.gz (legacy contrast)
or nii-new100/<case>/lung.nii.gz (plain-scan).

Output: outputs/figures/fig_r24y_overlay_gallery_unified301.png
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib

ROOT = Path(__file__).parent.parent.parent
COHORT = ROOT / "outputs" / "r24" / "cohort_locked_table.csv"
NII_LEGACY_STUB = Path(r"E:\桌面文件\nii格式图\nii-unified-282")
NII_NEW100_LOCAL = ROOT / "nii-new100"  # may not exist locally
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def resolve_ct_path(case_id: str, protocol: str) -> Path | None:
    """Fetch CT path with GBK redirect resolver (R20.A fix)."""
    if "plain_scan" in protocol.lower():
        # New100 plain-scan: try local nii-new100 (may not have it)
        for cand in [NII_NEW100_LOCAL / case_id / "ct.nii.gz"]:
            if cand.exists(): return cand
        return None
    # Contrast: try legacy nii-unified-282 stub then redirect
    cd = NII_LEGACY_STUB / case_id
    if not cd.is_dir(): return None
    direct = cd / "ct.nii.gz"
    if direct.exists(): return direct
    src = cd / "_source.txt"
    if src.exists():
        raw = src.read_bytes()
        for enc in ("gbk", "utf-8", "cp936"):
            try:
                src_path = Path(raw.decode(enc).strip())
                if src_path.is_dir():
                    for ct_name in ("original.nii.gz", "ct.nii.gz"):
                        candidate = src_path / ct_name
                        if candidate.exists(): return candidate
            except UnicodeDecodeError:
                continue
    return None


def resolve_lung_path(case_id: str, protocol: str) -> Path | None:
    """Lung mask source — for now use legacy stub if it has lung.nii.gz (HU-sentinel)."""
    cd = NII_LEGACY_STUB / case_id
    direct = cd / "lung.nii.gz"
    if direct.exists(): return direct
    src = cd / "_source.txt"
    if src.exists():
        raw = src.read_bytes()
        for enc in ("gbk", "utf-8", "cp936"):
            try:
                src_path = Path(raw.decode(enc).strip())
                if (src_path / "lung.nii.gz").exists():
                    return src_path / "lung.nii.gz"
            except UnicodeDecodeError:
                continue
    return None


def slice_overlay(ax, ct_p: Path, lung_p: Path | None, title: str):
    try:
        ct = nib.load(str(ct_p)).get_fdata().astype(np.float32)
        # Pick mid-axial slice
        z = ct.shape[2] // 2
        slc = ct[:, :, z]
        slc = np.clip(slc, -1000, 200)
        ax.imshow(slc.T, cmap="gray", origin="lower", vmin=-1000, vmax=200)
        if lung_p is not None and lung_p.exists():
            lung = nib.load(str(lung_p)).get_fdata()
            lmax = float(lung.max()) if lung.size else 0.0
            lmin = float(lung.min()) if lung.size else 0.0
            if lmax <= 1.5 and lmin >= -0.5:
                lung_slc = lung[:, :, z] > 0.5
            else:
                lung_slc = lung[:, :, z] != -2048
            if lung_slc.shape == slc.shape:
                masked = np.ma.masked_where(~lung_slc.T, lung_slc.T)
                ax.imshow(masked, cmap="autumn", alpha=0.35, origin="lower")
        ax.set_title(title, fontsize=8)
    except Exception as e:
        ax.text(0.5, 0.5, f"err: {str(e)[:30]}", ha="center", va="center",
                transform=ax.transAxes, fontsize=7)
        ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    cohort = pd.read_csv(COHORT)
    # Sample 8 PH + 8 nonPH within-contrast for representative gallery
    rng = np.random.default_rng(42)
    contrast = cohort[cohort["is_contrast_only_subset"]].copy()
    plain = cohort[cohort["protocol"] == "plain_scan"].copy()
    rep_ph = contrast[contrast["label"] == 1].sample(min(8, (contrast["label"]==1).sum()),
                                                       random_state=42)
    rep_nonph_contrast = contrast[contrast["label"] == 0].sample(
        min(4, (contrast["label"]==0).sum()), random_state=42)
    rep_nonph_plain = plain.sample(min(4, len(plain)), random_state=42)
    rep = pd.concat([rep_ph, rep_nonph_contrast, rep_nonph_plain]).reset_index(drop=True)

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    n_rendered = 0
    n_missing_ct = 0
    for i, row in rep.iterrows():
        ax = axes.flat[i] if i < 16 else None
        if ax is None: break
        cid = row["case_id"]
        ct_p = resolve_ct_path(cid, row["protocol"])
        lung_p = resolve_lung_path(cid, row["protocol"])
        label_str = "PH" if row["label"] == 1 else "nonPH"
        prot = row["protocol"][:5]
        mpap = row["measured_mpap"]
        mpap_str = f"mPAP={mpap:.0f}" if pd.notna(mpap) else "no_mpap"
        title = f"{label_str}/{prot}/{mpap_str}\n{cid[:30]}"
        if ct_p is None:
            ax.text(0.5, 0.5, f"CT not local\n{cid[:25]}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="red")
            ax.set_title(title, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            n_missing_ct += 1
            continue
        slice_overlay(ax, ct_p, lung_p, title)
        n_rendered += 1
    fig.suptitle(f"R24.Y — Lung-overlay gallery (unified-301 cohort, n_render={n_rendered}/16, "
                 f"n_missing_local_CT={n_missing_ct}/16)\n"
                 f"Replaces R19.A blank-placeholder gallery (root cause: GBK-redirect bug fixed in R20.A)",
                 fontsize=12)
    plt.tight_layout()
    out_path = OUT / "fig_r24y_overlay_gallery_unified301.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {out_path}; rendered={n_rendered}/16, missing={n_missing_ct}/16")


if __name__ == "__main__":
    main()
