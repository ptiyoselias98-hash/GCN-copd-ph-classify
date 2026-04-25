"""R19.C — patcher script that fixes _remote_build_v2_cache.py to handle
binary masks (Simple_AV_seg new100 output is 0/1 binary, but the legacy
v2 builder assumed HU-sentinel -2048 → ALL voxels treated as foreground
on binary masks → kimimaro hangs on 'whole-volume' skeletonization).

The patch detects mask type at load:
  - max(raw) ≤ 1.5 and min(raw) ≥ -0.5 → binary, use arr > 0
  - else → HU-sentinel -2048, use arr != -2048

Runs on remote: produces _R19C_build_v2_patched.py with the patched
mask-extraction block.
"""
import sys
sys.path.insert(0, '/home/imss/cw/GCN copdnoph copdph')

orig_path = '/home/imss/cw/GCN copdnoph copdph/_R19B_build_v2_new100.py'
with open(orig_path, 'r', encoding='utf-8') as f:
    src = f.read()

patched = src.replace(
    "raw = np.asarray(img.dataobj)\n    arr = (raw != -2048).astype(np.uint32)",
    """raw = np.asarray(img.dataobj)
    # R19.C patch: detect mask type — binary {0,1} vs HU-sentinel -2048
    raw_max = float(raw.max()) if raw.size else 0.0
    raw_min = float(raw.min()) if raw.size else 0.0
    if raw_max <= 1.5 and raw_min >= -0.5:
        # Binary mask (Simple_AV_seg output for new100)
        arr = (raw > 0).astype(np.uint32)
    else:
        # HU-sentinel mask (legacy nii-unified-282)
        arr = (raw != -2048).astype(np.uint32)"""
)

assert patched != src, "patch did not apply"

out_path = '/home/imss/cw/GCN copdnoph copdph/_R19C_build_v2_patched.py'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(patched)
print(f"saved patched script: {out_path}")
