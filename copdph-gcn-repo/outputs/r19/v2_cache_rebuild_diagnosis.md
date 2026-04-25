# R19.B/C v2 cache rebuild — diagnostic + fix (2026-04-25 22:58)

## Symptom (R19.B initial run, KILLED)

`_R19B_build_v2_new100.py` ran 12 parallel kimimaro workers on 100 new
plain-scan nonPH cases for **35+ minutes with 0 pkls written**. All 12
workers stayed pinned at 100% CPU + 4-6GB RSS each; killed via pkill.

## Root cause (R19.C diagnostic)

The legacy v2 builder (`_remote_build_v2_cache.py`) hard-codes:
```python
arr = (raw != -2048).astype(np.uint32)
```
assuming HU-sentinel mask convention (background = −2048, structure
carries raw HU values, used in legacy nii-unified-282 dataset).

**Simple_AV_seg `lung.pth`/`main_AV.pth` outputs are BINARY {0, 1}**, not
HU-sentinel. So `(raw != -2048)` returned `True` for **every voxel** (0
and 1 are both ≠ −2048) → the entire 512×512×500 volume was treated as
foreground → kimimaro tried to skeletonize the **whole 3D volume**
(O(n_voxels × surface_area)) → unbounded runtime.

## R19.C fix (committed at 5bc0fc9)

`scripts/evolution/R19C_build_v2_patcher.py` writes a patched builder
with mask-type auto-detection at load:
```python
raw = np.asarray(img.dataobj)
raw_max = float(raw.max()); raw_min = float(raw.min())
if raw_max <= 1.5 and raw_min >= -0.5:
    arr = (raw > 0).astype(np.uint32)        # binary (Simple_AV_seg)
else:
    arr = (raw != -2048).astype(np.uint32)   # HU-sentinel (legacy)
```

## Verification (R19.C 5-case test, PASSED)

```
[1/5 done] nonph_hujiaru_...                {'artery': 168, 'vein': 74, 'airway': 0}
[2/5 done] nonph_huangxiaokang_...          {'artery': 902, 'vein': 441, 'airway': 0}
[3/5 done] nonph_guruping_...               {'artery': 199, 'vein': 144, 'airway': 0}
[4/5 done] nonph_gaoyuguo_...               {'artery': 211, 'vein': 172, 'airway': 0}
[5/5 done] nonph_huangjianxin_...           {'artery': 152, 'vein': 200, 'airway': 0}
Summary: done=5 skip=0 miss=0 err=0 in 157.4s
```

**~30s per case** at 6 workers. 100-case full build ETA ~8-10 min.

Note: `airway: 0` for all cases because Simple_AV_seg lung.pth +
main_AV.pth don't produce airway segmentation; the new100 cohort has
artery + vein + lung only. R20 multi-branch will still work with airway
defaulting to empty Data on these cases.

## Full 100-case build LAUNCHED 23:00 (R19.C)

`nohup python _R19C_build_v2_patched.py --labels data/labels_new100.csv
--data_dirs nii-new100 --output_cache cache_tri_v2_new100 --workers 6`
Expected completion: 23:08-23:10. Then unblocks R18 must-fix #2
(embedding-level enlarged-stratum probe) and #5 (multi-seed CORAL on
enlarged stratum).

## Honest-debt status after this fire

1. R19 DDPM training: epoch 8/30 loss 0.041 (down from 0.81), ETA ~3-4h
   remaining after v2 builder I/O contention removed
2. Embedding-level enlarged probe: UNBLOCKED in ~10 min after v2 build
   completes
3. ✅ Lung overlay gallery: DONE (R19.A)
4. HiPaS re-segmentation 38: still pending
5. Multi-seed CORAL on enlarged: UNBLOCKED in ~10 min after v2 build
