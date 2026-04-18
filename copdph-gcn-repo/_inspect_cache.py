"""Remote-side inspector for one cache pkl + its artery/vein masks."""
import pickle, sys, torch
import nibabel as nib
import numpy as np
from pathlib import Path

CACHE = Path("/home/imss/cw/GCN copdnoph copdph/cache")
NII = Path("/home/imss/cw/COPDnonPH COPD-PH /data/nii")

pkl = next(CACHE.glob("ph_*.pkl"))
case = pkl.stem
print("case:", case)
with open(pkl, "rb") as f:
    d = pickle.load(f)
print("type:", type(d).__name__)
try:
    keys = list(d.keys())
except Exception:
    keys = [x for x in dir(d) if not x.startswith("_")][:40]
print("keys:", keys)
g = d["graph"]
print("graph type:", type(g).__name__)
try:
    print("graph keys:", list(g.keys()))
except Exception:
    print("graph attrs:", [x for x in dir(g) if not x.startswith("_")][:40])
# dump each tensor/array
for k in ["x","pos","edge_index","y","num_nodes","edge_attr"]:
    try: v = g[k]
    except Exception: v = getattr(g,k,None)
    if v is None: continue
    if torch.is_tensor(v):
        print("  g.", k, tuple(v.shape), str(v.dtype))
        if k == "pos":
            print("    pos min:", v.min(0).values.tolist(),
                  "max:", v.max(0).values.tolist(),
                  "mean:", v.float().mean(0).tolist())
        if k == "x":
            print("    x[0]:", v[0].tolist())
    else:
        print("  g.", k, "->", type(v).__name__, repr(v)[:120])
print("features keys:", list(d["features"].keys()) if isinstance(d["features"], dict) else type(d["features"]))
print("label:", d["label"])
for k in ["x", "pos", "edge_index", "y", "global_features", "case_id",
          "case", "affine", "origin", "spacing", "shape", "raw_shape",
          "voxel_shape", "node_voxel"]:
    v = getattr(d, k, None)
    if v is None:
        continue
    if torch.is_tensor(v):
        print(k, tuple(v.shape), str(v.dtype))
        if k == "pos":
            print("  pos min:", v.min(0).values.tolist(),
                  "max:", v.max(0).values.tolist())
        if k == "x":
            print("  x[0]:", v[0].tolist())
    else:
        print(k, "->", repr(v)[:200])

mask_dir = NII / case
print("mask dir exists:", mask_dir.exists())
if mask_dir.exists():
    for n in ["artery.nii.gz", "vein.nii.gz"]:
        p = mask_dir / n
        if p.exists():
            img = nib.load(str(p))
            arr = img.get_fdata()
            print(n, "shape:", img.shape, "diag:",
                  np.diag(img.affine)[:3].tolist(),
                  "nonzero:", int((arr > 0).sum()))
