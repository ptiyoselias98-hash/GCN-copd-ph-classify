"""Build artery/vein flag lookup for each cached patient graph.

For every ``cache/<case_id>.pkl`` we read the matching
``<nii_root>/<case_id>/{artery,vein}.nii.gz`` masks, look each node's
voxel coordinate up against both masks, and assign:

    +1.0  node sits inside an artery voxel
    -1.0  node sits inside a vein   voxel
     0.0  neither / both

Output: dict ``{case_id: torch.FloatTensor(N)}`` saved via ``torch.save``.
Cases whose mask dir is missing (or has missing artery/vein files) are
simply skipped — downstream training treats them as the zero fallback.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import nibabel as nib
import numpy as np
import torch


def _mask_lookup(pos_int: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """pos_int: (N, 3) int64 voxel coords clipped to mask shape."""
    i = np.clip(pos_int[:, 0], 0, mask.shape[0] - 1)
    j = np.clip(pos_int[:, 1], 0, mask.shape[1] - 1)
    k = np.clip(pos_int[:, 2], 0, mask.shape[2] - 1)
    return mask[i, j, k]


def build(cache_dir: Path, nii_root: Path, out_path: Path) -> None:
    lookup: dict[str, torch.Tensor] = {}
    cached = sorted(cache_dir.glob("*.pkl"))
    n_ok = n_miss_dir = n_miss_file = n_size_fail = 0
    for pkl in cached:
        case = pkl.stem
        mdir = nii_root / case
        if not mdir.is_dir():
            n_miss_dir += 1
            continue
        ap = mdir / "artery.nii.gz"
        vp = mdir / "vein.nii.gz"
        if not (ap.exists() and vp.exists()):
            n_miss_file += 1
            continue
        try:
            with pkl.open("rb") as f:
                d = pickle.load(f)
            g = d["graph"]
            pos = g.pos.detach().cpu().numpy().astype(np.int64)
            art = nib.load(str(ap)).get_fdata() > 0
            vei = nib.load(str(vp)).get_fdata() > 0
            if art.shape != vei.shape:
                n_size_fail += 1
                continue
            in_a = _mask_lookup(pos, art)
            in_v = _mask_lookup(pos, vei)
            av = np.zeros(pos.shape[0], dtype=np.float32)
            av[in_a & ~in_v] = 1.0
            av[in_v & ~in_a] = -1.0
            lookup[case] = torch.from_numpy(av)
            n_ok += 1
            if n_ok % 10 == 0:
                print(f"  [{n_ok}/{len(cached)}] last={case} N={pos.shape[0]} "
                      f"art={int(in_a.sum())} vein={int(in_v.sum())} "
                      f"neither={int(((~in_a) & (~in_v)).sum())}",
                      flush=True)
        except Exception as e:
            print(f"  ERR {case}: {e}", flush=True)
            n_size_fail += 1

    print(f"OK={n_ok}  miss_dir={n_miss_dir}  miss_file={n_miss_file}  "
          f"size_fail={n_size_fail}", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(lookup, str(out_path))
    print(f"saved {out_path} ({out_path.stat().st_size} B)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--nii_root", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    build(Path(a.cache_dir), Path(a.nii_root), Path(a.out))


if __name__ == "__main__":
    main()
