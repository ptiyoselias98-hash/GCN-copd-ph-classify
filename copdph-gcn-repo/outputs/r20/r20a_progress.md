# R20.A — Local CPU DDPM inference progress

**Status (this fire)**: running in background.

- Local torch CPU 2.11.0 installed
- TinyUNet3D loads checkpoint OK (`ddpm_state_dict.pt`, 12 MB)
- Local cohort `nii-unified-282/` confirmed (282 case dirs)
- Background command `bz0n5v6gq` running
  `python R20A_local_ddpm_inference.py --n_eval_patches 8`
- Log: `outputs/r20/r20a_inference.log`
- Python process at ~1.25 GB resident (model + first CT loaded)

## Lung mask read fix vs R19.G

R20.A uses `lung > -2000` (HU-sentinel: legacy masks have −2048 background, ≠0 foreground).
R19.G used `lung > 0.5` (binary), which would have failed silently on legacy
HU-sentinel masks → 0 foreground voxels → case skipped. R20.A path corrects this
for the 282 legacy cohort.

## Expected runtime

- 282 cases × 8 patches × ~30 ms forward pass each ≈ 70 s/case
- 282 × 70 s = ~5.5 h on a single CPU thread, but PyTorch CPU uses MKL multi-thread
- Realistic ETA: 60-180 min
- Periodic logging every 25 cases

## On completion (next fire)

1. Read `ddpm_anomaly_legacy.csv` + `ddpm_anomaly_legacy_eval.json`
2. Commit results
3. If AUC > 0.6 with PH > nonPH: closes R18 must-fix #1 (debt-item #2)
4. Move to debt-items #3-#5 (pipeline unification, embedding probe, multi-seed CORAL)
