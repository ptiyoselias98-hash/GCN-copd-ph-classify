# R19.B v2 cache rebuild on new100 — diagnostic note (2026-04-25 22:40)

## Symptom

`_R19B_build_v2_new100.py` ran 12 parallel kimimaro workers on the 100
new plain-scan nonPH cases (`nii-new100/`) for **35+ minutes with 0 pkls
written**. All 12 workers stayed pinned at 100% CPU + 4-6GB RSS each;
none completed even one case. Killed via `pkill -9 -f _R19B_build_v2`.

## Likely root cause

The lung masks in `nii-new100/<case_id>/lung.nii.gz` come from the
Simple_AV_seg `lung.pth` model and are **oversegmented on plain-scan CT**
(R16.A finding: median raw lung volume 10.8L vs adult plausible 1.5-8.5L,
79/100 cases >8.5L).

Kimimaro TEASAR skeleton extraction is **O(n_voxels × surface_area)** —
on a 10.8L mask, that's ~2× the work of a normal lung. Plus the
oversegmented voxels include extra-thoracic mediastinal/diaphragm regions
that kimimaro tries to skeletonize with high topological complexity.

Hypothesis: kimimaro hangs / runs unbounded on these dense oversegmented
volumes.

## Side effect

DDPM training on GPU 0 was also stuck at epoch 7 during the v2 builder
runtime — 12 workers × 5GB RSS = ~60GB RAM contention + heavy NIfTI I/O
contention with DDPM patch loader.

## Path forward (R19.C in next fire)

1. **Substitute repaired lung masks** before v2 rebuild.
   `R16.C` produced lung_features_new100_repaired.csv (HU<-300 + top-2-CC
   filter dropped median lung 10.8L → 7.7L). Need to write the repaired
   binary mask to `nii-new100/<cid>/lung_repaired.nii.gz` so the v2
   builder picks it up. OR: have v2 builder apply HU<-300+top-2-CC at
   load time (small patch to `_remote_build_v2_cache.py`).

2. **Run v2 builder with --limit 5** first to verify single-case time
   budget. If repaired-mask build of 1 case takes <5 min, full 100-case
   build at 12 workers should finish in ~50-100 min.

3. **Reduce concurrency to 6 workers** to leave RAM/I/O headroom for
   DDPM training on GPU 0.

## Decision

Kill + restart in next cron fire after R19.C patch ready. Not blocking
embedding-level enlarged-stratum probe permanently — just delays it
~1 hour.

## Honest-debt status (still 5 outstanding, no closures from R18)

1. R19 DDPM training: now resumed (GPU 0 unblocked) — epoch 7/30 with
   v2-builder gone, ETA recovers to ~3-5h
2. Embedding-level enlarged probe — still blocked on R19.C v2 rebuild
3. ✅ Lung overlay gallery — DONE (R19.A)
4. HiPaS re-segmentation 38 — pending
5. Multi-seed CORAL on enlarged — still blocked on R19.C v2 rebuild
