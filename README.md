# GCN Pulmonary Vascular Remodeling for COPD vs COPD-PH

This repository packages a pulmonary vascular graph convolution workflow for distinguishing COPD without pulmonary hypertension from COPD-PH using chest CT, commercial vessel segmentation outputs, and fused radiomics priors. The active implementation lives in `copdph-gcn-repo/`, with experiment reports collected under `outputs/`; the original untouched project skeleton is preserved alongside the active code as historical reference.

## Directory Layout

- `copdph-gcn-repo/`: active Python code for graph construction, training, reporting, attribution, and remote orchestration.
- `outputs/`: canonical experiment and visualization root, including Sprint 2 to Sprint 5 reports, radar charts, comparison figures, and attribution summaries.
- `data/`: local data drop used during development. Patient-level radiomics tables are excluded from this public import for privacy.
- `项目方案（下载未更改版）/`: original unmodified skeleton kept for reference only.
- `CLAUDE.md`: project operating notes, sprint history, remote workflow details, and implementation constraints.
- `AUTO_PIPELINE_README.md`, `analysis_and_plan.md`, `run_all.bat`: local experiment automation and planning notes.

## Training Pipeline

The main experimentation path is the Sprint 2 matrix driven by `copdph-gcn-repo/run_sprint2.py`: three model modes (`radiomics_only`, `gcn_only`, `hybrid`) crossed with two feature sets (`baseline`, `enhanced`) for six configurations per run. Baseline uses 12-dimensional node features from `utils/graph_builder.py`; enhanced adds curvature and injects 12 graph-level commercial vascular scalars before fusion rather than broadcasting them to every node. The reporting contract is fixed across all experiments: every result table and plot must include AUC, Accuracy, Precision, Sensitivity, F1, and Specificity, not AUC alone. Canonical Sprint 2 report artifacts in this workspace include `outputs/sprint2_v2/sprint2_results.json`, `outputs/sprint2_v2/sprint2_metrics.xlsx`, and `outputs/sprint2_v2/sprint2_radar.png`.

## Sprint History

- `2026-04-14`, Sprint 3 P0: `run_sprint3.py` launched three arms (`focal_local4`, `focal_all`, `wce_local4`) to improve specificity through Youden-threshold calibration, focal loss, and pruning graph-level globals down to `bv5_ratio`, `total_bv5`, `total_branch_count`, and `vessel_tortuosity`.
- `2026-04-16`, Sprint 5 final improvements: `run_sprint5.py` stacked mPAP-stratified cross-validation, node-drop augmentation (`p=0.1`), and an mPAP auxiliary regression head on the Sprint 3 baseline. Supporting outputs are summarized in `outputs/sprint5_实验结果/README.txt`.
- `2026-04-18`, CT PA/Ao measurement update: the workflow moved from echo-text extraction to direct CT-derived PA trunk and aortic measurements using `measure_pa_aorta_v2.py`, with local follow-up scripts `viz_ct_pa_ao.py` and `accurate_pa_ao_viz.py` referenced from `CLAUDE.md`.
- Additional intermediate reports remain in `outputs/sprint3_*`, `outputs/sprint4a_gated/`, `outputs/sprint4b_av/`, `outputs/attribution/`, and `outputs/improvements/`.

## Key Results

- Sprint 5 best model: enhanced/hybrid pooled AUC `0.889`, fold-mean AUC `0.924 +/- 0.047`, Sensitivity `0.870`, and Specificity `0.920` (`outputs/sprint5_实验结果/README.txt`).
- Relative to the Sprint 2 baseline comparator documented in the same report, pooled AUC improved by `+0.033` and Specificity improved by `+0.300`.
- Sprint 3 showed the core specificity lift: the `focal_local4` enhanced/hybrid arm reached pooled-style headline performance around AUC `0.912`, F1 `0.920`, and Specificity `0.950`, confirming the value of threshold calibration plus local-global pruning.
- Echo-based attribution analysis in `outputs/attribution/ablation_results.json` showed PA diameter plus PA/Ao alone was weak (`AUC 0.510`), whereas small-vessel features reached `AUC 0.815` and the full CT feature stack reached `AUC 0.913`.
- The CT-derived PA/Ao update from `copdph-gcn-repo/data/ct_pa_ao_measurements_v2.json` showed PH PA diameter `34.4 +/- 6.3 mm` versus non-PH `31.4 +/- 6.4 mm`; a two-sided Mann-Whitney comparison on the local measurement set gave `p = 0.024`, consistent with the project note that this effect is significant at roughly `p = 0.026`.

## Remote Training Workflow

Remote GPU training is handled by the paramiko-based helpers in `copdph-gcn-repo/`, especially `_remote_sync.py`, `_remote_launch.py`, `_remote_status.py`, `_remote_fetch.py`, `_remote_verify.py`, and the sprint-specific launchers such as `_remote_sprint3.py` and `_remote_sprint5.py`. These scripts target the canonical server paths documented in `CLAUDE.md`, and that file remains the source of truth for environment details and credentials; no secrets are duplicated in this repository README.

## Data Availability

This GitHub target is publicly reachable, so direct patient-identifying data has been excluded from the committed tree. That includes the patient spreadsheet, radiomics tables keyed by patient identifiers, mPAP lookup/split files derived from the clinical workbook, CT PA/Ao per-case measurement JSON, and any generated feature tables containing `patient_id` or `case_id`. Aggregate reports, figures, and code are retained, but the underlying clinical data cannot be shared for privacy reasons.

## License

License TBD.
