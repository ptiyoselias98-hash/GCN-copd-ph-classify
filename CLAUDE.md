# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Graph convolution framework for analyzing pulmonary vascular remodeling between COPD (no PH) and COPD-PH, from CT + commercial segmentation masks.

Active code lives in `copdph-gcn-repo/`. `项目方案（下载未更改版）/` is the original unmodified skeleton — do not edit; copy into the active repo if reusing.

## Directory Layout

- `copdph-gcn-repo/` — active code (training, models, graph construction, remote helpers)
- `data/copd_ph_radiomics.csv` — commercial CT radiomics (original Chinese column names, 45 features used)
- `outputs/` — **canonical output root**; each experiment gets its own subfolder (e.g. `sprint2_v2/`, `cv_full/`, `features/`, `sprint1_hybrid/`, `sprint2_enhanced/`, `可视化解释/`). All new reports must go under `outputs/<name>/`.
- `copdph-gcn-repo/cache/<case_id>.pkl` — precomputed per-patient graphs + features (used by all training runs)

## Training Pipeline (Sprint 2)

Three modes × two feature sets = six configs per run; driven by `run_sprint2.py`:

- Modes: `radiomics_only`, `gcn_only`, `hybrid` (all defined in `hybrid_gcn.py::HybridGCN`).
- Feature sets:
  - `baseline` — 12D node features from `utils/graph_builder.py`.
  - `enhanced` — 13D node features (baseline + curvature) **plus** 12D graph-level `global_features` (commercial scalars) concatenated onto the pooled GCN embedding *before* fusion. This is the design after the April refactor; broadcasting globals to every node was abandoned because it diluted local morphology signals.
- Sprint 1 helpers (`attach_radiomics`, `load_labels`, `load_splits`, `train_one_fold`, `full_metrics`, `case_to_pinyin`) are imported from `run_hybrid.py`. Radiomics are loaded from the CSV and keyed by pinyin of the Chinese case ID.
- Graph-level global features must be propagated through `Data.clone()` manually — see `run_sprint2._wrap()`. PyG drops non-standard `Data` attributes on clone/collate; the model has a zero fallback but relying on it masks plumbing bugs.

Local smoke-run (not typical — training is done on remote):
```
cd copdph-gcn-repo
python run_sprint2.py --cache_dir ./cache --radiomics ./data/copd_ph_radiomics.csv \
  --labels <labels.csv> --splits <splits_dir> --output_dir ./outputs/<name> \
  --epochs 300 --batch_size 8 --lr 1e-3
```

## Evaluation — Always Report 6 Metrics

Every reported result must include **AUC, Accuracy, Precision, Sensitivity (recall of positive class), F1, Specificity**. Never AUC alone. `utils/training.py::full_metrics` and the aggregation loop in `run_sprint2.py` already compute all six; summary tables and radar charts must stay six-metric.

`make_report.py` (in `copdph-gcn-repo/`) reads a `sprint2_results.json` and emits an Excel + radar PNG. The newer in-folder variant used by `outputs/sprint2_v2/_make_report.py` writes xlsx + radar + AUC bar into the experiment's own subfolder — prefer this pattern (report co-located with its JSON) for new experiments.

## Remote Training Workflow

Training runs on GPU box at `imss@10.60.147.117:22` under `/home/imss/cw/GCN copdnoph copdph/` using conda env `pulmonary_bv5_py39`. Helpers in `copdph-gcn-repo/` (all paramiko-based):

- `_remote_sync.py` — SFTP push of the full file list
- `_remote_push_one.py`, `_remote_push_two.py` — push 1 or 2 files then relaunch
- `_remote_launch.py` — nohup launch with canonical label/split paths
- `_remote_status.py` — tail log, `pgrep` the training process, `nvidia-smi`
- `_remote_fetch.py` — SFTP results back to a local `outputs/<name>/` subfolder
- `_remote_verify.py` — remote compile + smoke test

Canonical remote paths (hard-coded — do not re-discover via `find`, directory names contain spaces and non-ASCII):
```
LABELS  = "/home/imss/cw/COPDnonPH COPD-PH /data/tables/labels.csv"
SPLITS  = "/home/imss/cw/COPDnonPH COPD-PH /data/splits/folds"
ENV     = "source /home/imss/miniconda3/etc/profile.d/conda.sh && conda activate pulmonary_bv5_py39"
```

Remote launch gotchas (learned the hard way):
- Wrap commands as `bash -lc "<ENV> && <launch>"`; use single quotes around paths inside the outer double-quoted string.
- `nohup python ... < /dev/null > log 2>&1 & disown; echo launched` — redirect stdin from `/dev/null` or the SSH `exec_command` call hangs.
- `pgrep -fa 'python.*run_sprint2\.py'` — scope to python, otherwise the bash wrapper also matches.

## Node Feature Conventions

- `BASELINE_IN_DIM = 12` (diameter, length, tortuosity, CT density, generation, 3D orientation, 3D centroid, Strahler, degree).
- `EXPECTED_OUT_DIM = 13` for enhanced — baseline + curvature.
- `GLOBAL_FEATURE_DIM = 12` — commercial scalars fed as graph-level priors (see `enhance_features.py`).
- `radiomics_dim = 45` — full commercial radiomics vector passed to the radiomics branch; there's intentional overlap with the 12 globals. Empirically (sprint2_v2), this overlap hurt enhanced hybrid vs baseline hybrid — flag this when proposing further global-feature additions.

## Sprint 3 — P0 Improvements (launched 2026-04-14)

Goal: lift Specificity (was 0.68–0.71 in sprint2_v2 baseline) by fixing three known issues. Driver is `run_sprint3.py`; orchestrator is `_remote_sprint3.py`.

Three remote arms run **serially** in one nohup session, each writes `outputs/sprint3_<arm>/sprint3_results.json`; launcher `touch`es `outputs/sprint3_done.flag` when all three finish.

| Arm | Loss | globals_keep | Purpose |
|---|---|---|---|
| `focal_local4` | focal (γ=2, CB-weighted α) | 4 local globals | Primary P0 bet |
| `focal_all` | focal | all 12 globals | Ablate the globals-pruning effect |
| `wce_local4` | weighted_ce (sprint2 default) | 4 local globals | Ablate the focal-loss effect |

`local4` = global indices `[5, 7, 10, 11]` = `bv5_ratio, total_bv5, total_branch_count, vessel_tortuosity` (see `enhance_features.py::globals_row`). Youden's J threshold is always calibrated per fold on val — primary reported metrics use Youden threshold; `*_argmax` keys in each fold dict keep the old 0.5-threshold numbers for comparison.

Remote commands:
```bash
python _remote_sprint3.py launch   # push + nohup launch (idempotent-ish)
python _remote_sprint3.py status   # tail 3 logs + check outputs/sprint3_done.flag
python _remote_sprint3.py fetch    # sftp all 3 JSONs + logs to outputs/sprint3_*/
```

Resume protocol if the conversation broke mid-run:
1. `python _remote_sprint3.py status` — if `sprint3_done.flag` exists, training finished; go to step 3.
2. If still running, wait and re-poll; the serial loop can't be interrupted without restart.
3. `python _remote_sprint3.py fetch` — download all arms.
4. Run a per-arm variant of `outputs/sprint2_v2/_make_report.py` (adapted for the new JSON: top-level `_config` key, fold dicts now include `threshold` and `*_argmax` keys — filter those out when aggregating). Emit xlsx + radar per arm and a combined bar comparing `AUC/Spec/F1` across the 3 arms × baseline-hybrid vs enhanced-hybrid.
5. Compare against sprint2_v2 hybrid numbers: baseline (AUC .898, Spec .710) / enhanced (AUC .852, Spec .870). Success criterion = AUC ≥ .90 AND Spec ≥ .80 in at least one arm's `enhanced/hybrid`.

## Sprint 5 — Final Improvements (completed 2026-04-16)

Three improvements stacked on Sprint 3 focal_local4 baseline:
1. **mPAP-stratified cross-validation** — ensures each fold has balanced mPAP distribution
2. **Node-drop augmentation** (p=0.1) — randomly drops graph nodes during training
3. **mPAP regression auxiliary head** (weight=0.1) — multi-task learning

**Best result:** enhanced/hybrid pooled AUC = 0.889 (+0.033 from Sprint 2 baseline 0.856), Specificity 0.620 -> 0.920.

Driver: `run_sprint5.py` on server. Results in `outputs/sprint5_v2/sprint5_results.json`.

Key fixes applied:
- `splits_mpap_stratified_v2.json` + `mpap_lookup_v2.json` — fixed ID format mismatch (Chinese name -> pinyin -> cache case_id) using pypinyin
- BatchNorm crash fix — `drop_last=True` in training DataLoader
- Scripts that generated these fixes: `C:\Users\cheng\Desktop\从claude下载的实验执行脚本\local_gen_data.py` and `patch_sprint5.py`

Reports generated:
- Radar charts + comparison tables in `outputs/sprint5_实验结果/`
- `gen_sprint5_report.py` and `gen_final_report.py` in `C:\Users\cheng\Desktop\从claude下载的实验执行脚本\`

## CT PA/Ao Diameter Measurement (in progress as of 2026-04-18)

Measuring PA trunk diameter and Aorta diameter directly from CT NII files on server.

**Problem:** Previous PA/Ao analysis used echo ultrasound text extraction (regex from Excel column), which gave AUC=0.510. This is NOT CT measurement — echo and CT are methodologically different.

**Current approach:** Two scripts running on server via nohup:
- `measure_pa_aorta_server.py` (v1) — area-equivalent diameter, overestimates (mean ~45mm)
- `measure_pa_aorta_v2.py` (v2, preferred) — minor axis from inertia tensor + circularity filtering, more accurate (nonPH PA ~23mm)

**Status:** Both scripts were running when SSH became unreachable (sshd overloaded). Check `/tmp/measure_pa_ao_v2.log` and `data/ct_pa_ao_measurements_v2.json` on resume.

**Next steps:**
1. Check server SSH, download v2 JSON
2. Run `viz_ct_pa_ao.py` locally for 6-panel publication figure
3. Also run `accurate_pa_ao_viz.py` for echo-derived comparison

Scripts in `C:\Users\cheng\Desktop\从claude下载的实验执行脚本\`:
- `measure_pa_aorta_v2.py` — server-side CT measurement
- `viz_ct_pa_ao.py` — local visualization from CT JSON
- `accurate_pa_ao_viz.py` — echo-derived PA/Ao analysis

## Server Connection Details

- **IP:** 10.60.147.117, port 22, user `imss`, password `imsslab`
- **SSH key auth DOES NOT WORK** — use paramiko password auth only
- **Python env:** `/home/imss/.virtualenvs/causcal_time_series/bin/python` (Python 3.7, nibabel 4.0.2, scipy, numpy, torch)
- **CAUTION:** Do not run more than 1 SSH monitor loop concurrently — overloads sshd MaxStartups
- For long scripts: `nohup python -u script.py > /tmp/log 2>&1 &`
- JSON serialization: always use `cls=NumpyEncoder` for numpy float32

## Conventions

- Windows bash: forward-slash paths in scripts; paramiko/SFTP is the only remote bridge (no ssh/scp CLI).
- When a script prints to stdout and paths/results are Chinese, wrap stdout with `io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")` — several helpers already do this.
- Result JSONs use the shape `{feat_set: {mode: {folds: [...], mean: {...}, std: {...}, pooled_AUC: float}}}`. Keep this contract so `make_report.py` and downstream notebooks continue to work.
