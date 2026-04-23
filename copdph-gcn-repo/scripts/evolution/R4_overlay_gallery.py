"""R4.3 — Overlay gallery for anatomical TEASAR QC.

Produces a PNG grid of 10 representative cases showing, per case:
  - axial mid-slice of lung + vessels overlay
  - axial mid-slice with skeleton overlay
  - coronal maximum-intensity projection of the vessel mask

Sampled: 5 PH + 5 nonPH, balanced 5 contrast + 5 plain-scan. Selected at
random with fixed seed from cases that passed v2 extraction.

Output:
  outputs/evolution/R4_overlay_gallery.png
  outputs/evolution/R4_overlay_gallery_cases.txt
"""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize as skeletonize_3d

ROOT = Path(__file__).parent.parent.parent
CASES_ROOT = Path("E:/桌面文件/nii格式图/nii-unified-282")
FALLBACK = Path("E:/桌面文件/nii格式图/nii")
V2 = ROOT / "outputs" / "lung_features_v2.csv"
PROTO = ROOT / "data" / "case_protocol.csv"
OUT_PNG = ROOT / "outputs" / "evolution" / "R4_overlay_gallery.png"
OUT_TXT = ROOT / "outputs" / "evolution" / "R4_overlay_gallery_cases.txt"

SEED = 20260423
BG = -2048


def resolve_case_dir(case_id: str) -> Path:
    p = CASES_ROOT / case_id
    if (p / "lung.nii.gz").exists():
        return p
    f = FALLBACK / case_id
    if (f / "lung.nii.gz").exists():
        return f
    s = p / "_source.txt"
    if s.exists():
        t = s.read_text(encoding="utf-8", errors="ignore").strip()
        return Path(t.replace("\\", "/"))
    return p


def detect_convention(arr: np.ndarray) -> str:
    slab = arr[::32, ::32, ::32]
    if (slab == BG).any():
        return "hu"
    if arr.max() <= 255 and arr.min() >= 0:
        return "binary"
    return "hu"


def load_mask(path: Path):
    if not path.exists():
        return None
    try:
        img = nib.load(str(path))
    except Exception:
        return None
    arr = np.asarray(img.get_fdata(dtype=np.float32))
    conv = detect_convention(arr)
    mask = (arr != BG) if conv == "hu" else (arr > 0)
    return mask


def select_cases() -> list[str]:
    v2 = pd.read_csv(V2)
    proto = pd.read_csv(PROTO)
    df = proto.merge(v2, on="case_id", how="inner")
    df = df[df["error"].isna() | (df["error"] == "")]
    # Only pick cases without placeholder vessels so the overlay is informative
    df = df[(df["artery_placeholder"] == 0) & (df["vein_placeholder"] == 0)]
    rng = random.Random(SEED)
    picks: list[str] = []
    for (label, protocol), want in [
        ((1, "contrast"), 3),
        ((0, "contrast"), 2),
        ((0, "plain_scan"), 3),
        ((1, "contrast"), 2),
    ]:
        pool = df[(df["label"] == label) & (df["protocol"] == protocol)]["case_id"].tolist()
        rng.shuffle(pool)
        for c in pool[: want]:
            if c not in picks:
                picks.append(c)
        if len(picks) >= 10:
            break
    return picks[:10]


def render(ax_row, case_id: str, case_dir: Path) -> None:
    lung = load_mask(case_dir / "lung.nii.gz")
    artery = load_mask(case_dir / "artery.nii.gz")
    vein = load_mask(case_dir / "vein.nii.gz")
    if lung is None or artery is None or vein is None:
        for ax in ax_row:
            ax.set_title(f"{case_id[:25]}\n(missing)")
            ax.axis("off")
        return
    if artery.shape != lung.shape or vein.shape != lung.shape:
        for ax in ax_row:
            ax.set_title(f"{case_id[:25]}\n(placeholder)")
            ax.axis("off")
        return

    vessels = (artery | vein)
    # mid-axial slice with most lung voxels
    per_z = lung.sum(axis=(0, 1))
    z_mid = int(np.argmax(per_z))

    ax = ax_row[0]
    ax.imshow(lung[:, :, z_mid].T, cmap="gray", origin="lower")
    overlay = np.zeros_like(lung[:, :, z_mid], dtype=np.uint8)
    overlay[artery[:, :, z_mid]] = 1
    overlay[vein[:, :, z_mid]] = 2
    ax.imshow(np.ma.masked_equal(overlay.T, 0), cmap="coolwarm", origin="lower", alpha=0.6)
    ax.set_title(f"{case_id[:22]}\naxial mid, vessels")
    ax.axis("off")

    # skeleton overlay
    skel_a = skeletonize_3d(artery).astype(bool)
    skel_v = skeletonize_3d(vein).astype(bool)
    skel = skel_a | skel_v
    ax = ax_row[1]
    ax.imshow(lung[:, :, z_mid].T, cmap="gray", origin="lower")
    overlay = np.zeros_like(lung[:, :, z_mid], dtype=np.uint8)
    overlay[skel_a[:, :, z_mid]] = 1
    overlay[skel_v[:, :, z_mid]] = 2
    ax.imshow(np.ma.masked_equal(overlay.T, 0), cmap="coolwarm", origin="lower", alpha=0.9)
    ax.set_title(f"skeleton — SL_A={int(skel_a.sum())}, SL_V={int(skel_v.sum())}")
    ax.axis("off")

    # coronal MIP of vessels
    mip = vessels.sum(axis=1).astype(float)
    ax = ax_row[2]
    ax.imshow(mip.T, cmap="hot", origin="lower")
    ax.set_title("coronal MIP (A∪V)")
    ax.axis("off")


def main() -> None:
    picks = select_cases()
    print(f"Selected cases: {picks}")
    OUT_TXT.write_text("\n".join(picks), encoding="utf-8")

    n = len(picks)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]
    for i, cid in enumerate(picks):
        cd = resolve_case_dir(cid)
        render(axes[i], cid, cd)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
