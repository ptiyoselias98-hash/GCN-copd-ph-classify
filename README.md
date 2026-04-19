# GCN Pulmonary Vascular Remodeling — COPD vs COPD-PH

Graph-convolutional classification of pulmonary hypertension (PH) from chest-CT
pulmonary vessel trees, fused with commercial radiomics features. This
repository tracks the full experimental progression from Sprint 2 baseline
through Sprint 5 final improvements plus a follow-up CT-based PA/Ao
measurement study.

Active code lives in `copdph-gcn-repo/`. Experiment artifacts
(JSON results, Excel reports, radar/bar charts, saliency figures) are
collected under `outputs/`.

## Task

Binary classification — **COPD-PH vs COPD-nonPH** — from:
- **Graph modality**: per-patient pulmonary vessel tree. Nodes = vessel
  segments, edges = bifurcation connectivity. 12D baseline / 13D enhanced node
  features (geometry, CT density, Strahler order, orientation, curvature).
- **Global priors**: up to 12 commercial vessel scalars (e.g. `bv5_ratio`,
  `total_branch_count`, `vessel_tortuosity`) concatenated onto the pooled GCN
  embedding before fusion.
- **Radiomics modality**: 45-dim commercial CT radiomics vector per patient.

Three training modes, two feature sets → six configurations per run:

| mode | inputs |
|---|---|
| `radiomics_only` | 45D radiomics → MLP head |
| `gcn_only` | vessel graph → GCN + global pool |
| `hybrid` | GCN embedding ⊕ radiomics → MLP head |

All results report **six metrics** (AUC, Accuracy, Precision, Sensitivity, F1,
Specificity) on a 5-fold CV split over 96–113 matched patients.

## Sprint Progression

| Sprint | Driver | Key change | Best AUC | Best Spec |
|---|---|---|---|---|
| 2 baseline | `run_sprint2.py` | 3-mode × baseline/enhanced sweep | 0.926 | 0.820 |
| 2 v2 | `run_sprint2.py` | globals concat on pooled embedding (not broadcast) | 0.898 | 0.870 |
| 3 P0 | `run_sprint3.py` | focal loss + Youden threshold + globals pruning | **0.912** | **0.950** |
| 4a / 4b | `run_sprint3.py` + gated / A-V flag | fusion gate + artery/vein node prior | on par | 0.95 |
| 5 final | `run_sprint5.py` | mPAP-stratified CV + node-drop aug + mPAP aux head | 0.924 ± 0.047 | 0.920 |

Sprint-progression summary (pooled AUC across sprints):

![Sprint progression radar](outputs/sprint5_实验结果/sprint_progression_radar.png)

---

## Sprint 2 — baseline matrix

6-metric radar, split by feature set and overlaid:

![Sprint 2 radar — split](outputs/sprint2_radar.png)

![Sprint 2 radar — all 6 configs overlaid](outputs/sprint2_radar_combined.png)

v2 (globals concat on pooled embedding):

![Sprint 2 v2 radar](outputs/sprint2_v2/sprint2_radar.png)
![Sprint 2 v2 AUC bar](outputs/sprint2_v2/sprint2_auc_bar.png)

ROC comparison (full CV):

![ROC comparison](outputs/cv_full/plots/roc_comparison.png)
![CV radar](outputs/cv_full/plots/radar_chart.png)

---

## Sprint 3 — P0 improvements (focal + Youden + local4 globals)

Sprint 2 v2 had Specificity stuck at 0.71 on baseline/hybrid. Three P0 changes
landed in `run_sprint3.py`:

1. **Youden's J threshold calibration** per fold (primary metrics now use the
   val-optimal threshold; old 0.5-argmax metrics kept in `*_argmax`).
2. **Focal loss** (γ=2, class-balanced α via Cui 2019) replacing weighted CE.
3. **Globals pruning** (`--globals_keep local4`): keep only
   `bv5_ratio, total_bv5, total_branch_count, vessel_tortuosity` — the 4
   locality-meaningful commercial scalars; drop the 8 whole-lung scalars that
   overlap with the 45D radiomics.

Three arms, each a full 5-fold × 3-mode × 2-feature-set sweep:

| arm | loss | `globals_keep` | purpose |
|---|---|---|---|
| `focal_local4` | focal | 4 local globals | primary P0 combination |
| `focal_all` | focal | all 12 | ablate globals pruning |
| `wce_local4` | weighted_ce | 4 local globals | ablate focal loss |

### Headline (5-fold CV, enhanced / hybrid)

| run | AUC | ACC | Prec | Sens | F1 | **Spec** |
|---|---|---|---|---|---|---|
| sprint2_v2 baseline/hybrid | 0.898 | 0.804 | 0.905 | 0.836 | 0.857 | 0.710 |
| sprint2_v2 enhanced/hybrid | 0.852 | 0.797 | 0.961 | 0.769 | 0.841 | 0.870 |
| **sp3 focal_local4** enh/hyb | **0.912** | 0.889 | 0.983 | 0.870 | 0.920 | **0.950** |
| sp3 focal_all enh/hyb | 0.890 | 0.901 | 0.987 | 0.883 | 0.924 | 0.950 |
| sp3 wce_local4 enh/hyb | 0.895 | **0.919** | 0.974 | **0.920** | **0.944** | 0.910 |

**Specificity lifted from 0.71 → 0.95 (+24 points)**, AUC also improved.

### Cross-arm comparison

![sprint3 arms radar — enhanced](outputs/sprint3_arms_radar.png)
![sprint3 vs sprint2 bar](outputs/sprint3_combined_bar.png)

### Per-arm radars

![focal_local4 radar](outputs/sprint3_focal_local4/sprint3_radar.png)
![focal_all radar](outputs/sprint3_focal_all/sprint3_radar.png)
![wce_local4 radar](outputs/sprint3_wce_local4/sprint3_radar.png)

### Ablation read

- **Youden calibration** is the dominant factor — all three arms share it and
  all three beat sprint2_v2 handily.
- **`local4 > all`** (focal AUC .912 vs .890) confirms the 8 whole-lung
  scalars were redundant with the 45-dim radiomics vector.
- **focal vs wce** (both `local4`): focal wins AUC (.912 vs .895); wce wins
  F1 (.944 vs .920). Both strongly beat uncalibrated sprint2.

---

## Sprint 4 — gated fusion + A/V node flag

Two P1 upgrades layered on `focal_local4`:

| Arm | Change | Layer |
|---|---|---|
| **4a — gated fusion** | replace `concat(graph_emb, rad_emb)` with `gate * graph + (1-gate) * proj(rad)` | fusion |
| **4b — A/V node flag** | append 3-valued artery/vein/neither flag per node (from commercial masks) | node features |

![sprint4 arms radar](outputs/sprint4_arms_radar.png)
![sprint4 combined bar](outputs/sprint4_combined_bar.png)

Per-arm:

![4a gated radar](outputs/sprint4a_gated/sprint4_radar.png)
![4b A/V radar](outputs/sprint4b_av/sprint4_radar.png)

---

## Sprint 5 — final improvements

Three stacked improvements on top of `focal_local4`:

1. **mPAP-stratified CV** (`splits_mpap_stratified_v2.json`) — each fold has
   balanced mPAP distribution. Fixes the fold 4/5 imbalance issue.
2. **Node-drop augmentation** (p = 0.1) — randomly drops graph nodes during
   training to regularize against any single-branch reliance.
3. **mPAP regression auxiliary head** (weight 0.1) — multi-task objective
   forcing the GCN embedding to also be informative of continuous mPAP.

### Headline

- **enhanced/hybrid pooled AUC = 0.889** (+0.033 vs Sprint 2 baseline 0.856)
- **fold-mean AUC = 0.924 ± 0.047**
- **Sensitivity 0.870, Specificity 0.920** (+0.300 in Spec vs Sprint 2 baseline)

### Figures

Combined radar (all 6 configs):

![sprint5 radar combined](outputs/sprint5_实验结果/sprint5_radar_combined.png)

Sprint 5 vs Sprint 2 comparison:

![sprint5 vs sprint2 radar](outputs/sprint5_实验结果/sprint5_vs_sprint2_radar.png)

Pooled AUC bar:

![pooled AUC comparison](outputs/sprint5_实验结果/pooled_auc_comparison.png)

Feature-set ablation:

![Sprint 5 feature ablation](outputs/sprint5_实验结果/feature_ablation.png)

---

## CT-based PA / Aorta measurement (2026-04-18)

Replacing the earlier echo-text extraction (AUC 0.510 — poor) with direct
CT measurement via `measure_pa_aorta_v2.py`:

- **Method**: minor-axis length from inertia-tensor eigendecomposition on
  commercial `artery.nii.gz`; circularity & solidity filtering to pick the
  best PA trunk slice.
- **Sample**: 189 / 201 cases successfully measured (12 failed, mostly
  degenerate masks).

### Results (v2, minor-axis)

| group | n | PA diameter (mm) | PA/Ao ratio |
|---|---|---|---|
| COPD (no PH) | 27 | 31.4 ± 6.4 | 0.943 ± 0.268 |
| COPD-PH | 162 | 34.4 ± 6.3 | 1.130 ± 0.451 |
| t-test | | **p = 0.026** | **p = 0.039** |

6-panel publication figure (boxplots, scatter, histogram, summary table):

![CT PA/Ao analysis](outputs/sprint5_实验结果/ct_pa_ao_analysis.png)

---

## Feature attribution & ablation

Same 5-fold RF ablation, run twice with different PA/Ao sources to isolate the
measurement-method effect. The CT v2 version is the current one; the echo
version is kept as a legacy comparator.

### CT v2 PA/Ao (current — direct 3D measurement from `artery.nii.gz`)

Source: `outputs/attribution/ablation_results_ct.json`. Generator:
`copdph-gcn-repo/attribution_analysis_ct.py`. Cohort: 100 patients (74 PH /
26 nPH), 93 matched to the CT v2 measurement set.

| Config | AUC |
|---|---|
| **PA diam + PA/Ao only (CT, 3D)** | **0.542** |
| Small vessel features (13D) | 0.815 |
| Global vascular (26D) | 0.837 |
| Parenchyma + airway (15D) | 0.732 |
| Lobe-level vascular (60D) | 0.909 |
| All CT features (101D) | 0.913 |
| All CT + PA/Ao (CT, 104D) | 0.906 |
| All CT − small vessels (88D) | 0.904 |

![Attribution feature ablation (CT v2)](outputs/attribution/feature_ablation_ct.png)

CT v2 PA/Ao alone (AUC 0.542) is only marginally stronger than the echo
version (0.510); small-vessel topology features still beat it by +0.27 AUC.
The "measurement modality" is not where the signal lives — the full small-
vessel topology is.

### Echo-derived PA/Ao (legacy — 2D, parsed from 超声报告)

Source: `outputs/attribution/ablation_results_echo.json`. This predates the
CT v2 measurement work and is kept only as a historical comparator. The PA
diameter here is extracted by regex from the ultrasound report column — a
2D thumbnail view, not a direct CT measurement.

| Config | AUC |
|---|---|
| PA diam + PA/Ao only (echo, 3D) | 0.510 |
| Small vessel features (13D) | 0.815 |
| All CT features (101D) | 0.913 |

![Attribution feature ablation (echo, legacy)](outputs/attribution/feature_ablation_echo.png)

Per-feature violin plots (PH vs non-PH, top 9 commercial scalars):

![Feature comparison grid](outputs/features/plots/feature_comparison_grid.png)

Individual violins:

| | |
|---|---|
| ![BV5](outputs/features/plots/violin_vascular__bv5.png) | ![BV10](outputs/features/plots/violin_vascular__bv10.png) |
| ![Mean diameter](outputs/features/plots/violin_vascular__mean_diameter.png) | ![Num branches](outputs/features/plots/violin_vascular__num_total_branches.png) |
| ![Pruning index](outputs/features/plots/violin_vascular__pruning_index.png) | ![Artery/vein ratio](outputs/features/plots/violin_vascular__artery_vein_ratio.png) |
| ![Vessel/lung ratio](outputs/features/plots/violin_vascular__vessel_to_lung_ratio.png) | ![Wall thickness](outputs/features/plots/violin_airway__wall_thickness_ratio.png) |
| ![Wall area %](outputs/features/plots/violin_airway__wall_area_pct.png) | |

Improvement study summary:

![Improvement results](outputs/improvements/improvement_results.png)

---

## Interpretability

Group statistics — PH vs non-PH over 4 commercial vessel features
(boxplots + Mann-Whitney p):

![Group stats](outputs/viz_group_stats.png)

3D pulmonary vessel graphs (one PH + one non-PH, nodes coloured by
branching degree):

![Vessel tree samples](outputs/viz_vessel_tree_samples.png)

Per-node saliency `|∂p(PH)/∂x_node|` for the enhanced-hybrid GCN, fold 1:

![Saliency trees](outputs/viz_saliency_trees.png)

All 5 folds (each fold retrained, saliency on first PH + nonPH of val set):

| Fold | val AUC |
|---|---|
| 1 | 0.96 |
| 2 | 1.00 |
| 3 | 1.00 |
| 4 | 0.78 |
| 5 | 0.78 |

![Saliency — fold 1](outputs/viz_saliency_fold1.png)
![Saliency — fold 2](outputs/viz_saliency_fold2.png)
![Saliency — fold 3](outputs/viz_saliency_fold3.png)
![Saliency — fold 4](outputs/viz_saliency_fold4.png)
![Saliency — fold 5](outputs/viz_saliency_fold5.png)

Combined 5×2 grid:

![Saliency — all folds](outputs/viz_saliency_all_folds.png)

### Why do folds 4 & 5 underperform? — mPAP audit

For each fold's val set we join `case_id` back to the patient spreadsheet,
recover mPAP, and check whether folds 4/5 are dominated by borderline
(mPAP ≈ 20 mmHg) cases:

![Per-fold mPAP audit](outputs/fold_mpap_audit.png)

| Fold | val AUC | n_val | n_PH | n_nonPH | nonPH max mPAP | PH min mPAP | gap | borderline ∈[18,22] |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.96 | 21 | 15 | 6 | 20 | 22 | +2 | 5 |
| 2 | 1.00 | 17 | 11 | 6 | 18 | 26 | +8 | 1 |
| 3 | 1.00 | 21 | 17 | 4 | 15 | 23 | +8 | 0 |
| 4 | 0.78 | 24 | 19 | 5 | 20 | 26 | +6 | 2 |
| 5 | 0.78 | 22 | 17 | 5 | 19 | 24 | +5 | 3 |

Conclusion: labels are clean, borderline density does not explain the gap.
Folds 4/5 have only 5 nonPH cases each (~20% negatives) — a single
misclassified negative pushes specificity down by 0.20. Sprint 5's
mPAP-stratified CV directly addresses this.

---

## Directory Layout

```
.
├── CLAUDE.md                       # project operating notes (source of truth)
├── README.md                       # this file
├── copdph-gcn-repo/                # active Python code
│   ├── hybrid_gcn.py               # HybridGCN(radiomics_only | gcn_only | hybrid)
│   ├── gcn_models.py               # GCN backbones
│   ├── graph_builder.py            # vessel tree → PyG graph
│   ├── enhance_features.py         # 12D → 13D + 12 graph-level globals
│   ├── run_sprint2.py              # Sprint 2: baseline vs enhanced × 3 modes
│   ├── run_sprint3.py              # Sprint 3 P0 (focal + Youden + local4)
│   ├── run_sprint5.py              # Sprint 5 final (stratified CV + node-drop + mPAP aux)
│   ├── measure_pa_aorta_v2.py      # CT PA/Ao direct measurement (v2, minor-axis)
│   ├── make_report.py              # radar chart + xlsx summary
│   ├── visualize.py                # group stats + 3D trees + saliency
│   ├── utils/                      # shared pipeline utilities
│   └── _remote_*.py                # paramiko helpers for remote GPU training
├── outputs/
│   ├── sprint2_v2/, sprint3_*/, sprint4a_gated/, sprint4b_av/
│   ├── sprint5_实验结果/           # final sprint charts + CT PA/Ao figure
│   ├── cv_full/plots/              # ROC + radar for full-CV sweeps
│   ├── features/plots/             # violin plots + comparison grid
│   ├── attribution/                # ablation over feature subsets
│   ├── improvements/               # improvement study bar charts
│   ├── viz_saliency_*.png          # per-fold saliency figures
│   ├── viz_group_stats.png         # PH vs non-PH distribution plots
│   └── fold_mpap_audit.png         # mPAP audit figure
├── data/                           # small CSVs only — patient tables excluded
├── 项目方案（下载未更改版）/        # original unmodified skeleton (reference)
└── copd-ph患者113例0331.xlsx        # (gitignored) clinical spreadsheet
```

## Training Pipeline

1. **Preprocess** CT + commercial `artery.nii.gz` masks → 3D vessel skeleton →
   PyG graph (`skeleton.py`, `graph_builder.py`, `quantification.py`) → cache
   as `copdph-gcn-repo/cache/<case_id>.pkl`.
2. **Enhance** (optional) — augment 12D node features with curvature; attach
   12 graph-level commercial scalars (`enhance_features.py`).
3. **Train** 5-fold CV across 3 modes × baseline/enhanced
   (`run_sprint2.py` / `run_sprint3.py` / `run_sprint5.py`).
4. **Visualize** (`visualize.py`) — group stats, 3D trees, per-node saliency.
5. **Report** (`make_report.py` or in-folder `_make_report.py`) — radar +
   Excel summary.

Example (remote, Sprint 3 recommended config):

```bash
conda activate pulmonary_bv5_py39
CUDA_VISIBLE_DEVICES=0 python run_sprint3.py \
    --cache_dir ./cache --radiomics ./data/copd_ph_radiomics.csv \
    --labels /path/to/labels.csv --splits /path/to/splits/folds \
    --output_dir outputs/sprint3_focal_local4 \
    --epochs 300 --batch_size 8 --lr 1e-3 \
    --loss focal --globals_keep local4
```

## Remote Training Workflow

Training runs on a GPU box reached via paramiko (password auth). Helpers in
`copdph-gcn-repo/`:

- `_remote_sync.py` / `_remote_push_one.py` / `_remote_push_two.py` — SFTP push
- `_remote_launch.py` — nohup launch with canonical label / split paths
- `_remote_status.py` — tail log + `pgrep` + `nvidia-smi`
- `_remote_fetch.py` — SFTP results back to local `outputs/<name>/`
- `_remote_sprint3.py` / `_remote_sprint4.py` / `_remote_sprint5.py` —
  per-sprint orchestrators (serial multi-arm nohup + done flag)

Canonical remote paths and environment details are documented in `CLAUDE.md`.

## Dependencies

Python 3.9, PyTorch 2.x, `torch_geometric`, pandas, numpy, matplotlib,
scikit-learn, scipy, openpyxl, paramiko (remote helpers), nibabel (CT PA/Ao
measurement). See `pulmonary_bv5_py39` conda env on the training server for
exact versions; CT measurement uses the server's
`/home/imss/.virtualenvs/causcal_time_series` (Python 3.7, nibabel 4.0.2).

## Data & Privacy

Patient-identifying data is **excluded** from this public repository:
- `copd-ph患者113例0331.xlsx` — clinical spreadsheet with Chinese names
- `data/labels.csv`, `data/copd_ph_radiomics.csv` — keyed by patient ID
- `copdph-gcn-repo/data/ct_pa_ao_measurements_v2.json`,
  `mpap_lookup.json`, `splits_mpap_stratified.json` — derived from the
  clinical spreadsheet
- `outputs/features/features.csv` — per-case feature table

Aggregate reports, figures, and code are retained. Generator scripts
(e.g. `gen_mpap_lookup.py`) are kept as source even when their outputs are
excluded.

## License

License TBD.

## Follow-up experiments (gold subset + clustering)

These follow-up analyses stress-tested three questions on the 106-case gold subset against the broader 197-label setting: how much the original 197-vs-106 mismatch was contributing label leakage, whether the dropped `mPAP` auxiliary triple could recover the Sprint 5 operating point when reintroduced with per-fold Youden calibration, and whether vessel-topology fingerprints show unsupervised phenotypes that align with PH labels. The packaged results below archive the full 4-way comparison, 3-mode ablation, and clustering sanity check without rerunning training.

| run                                 | folds | AUC               | Sens              | Spec              | Acc               | F1                | Prec              |
|---|---|---|---|---|---|---|---|
| Sprint 5 Final (GitHub, no Youden)  |   5   | 0.924             | 0.870             | 0.920             | n/r               | n/r               | n/r               |
| `medium`        (0.5 threshold)     |   5   | 0.919 ± 0.082     | 0.431 ± 0.170     | 0.920 ± 0.160     | 0.557 ± 0.104     | 0.571 ± 0.157     | 0.969 ± 0.062     |
| **`medium_youden`** (per-fold J)    |   5   | 0.899 ± 0.100     | **0.798 ± 0.195** | **0.960 ± 0.080** | 0.838 ± 0.143     | 0.866 ± 0.135     | **0.988 ± 0.025** |
| **`medium_youden_rep`** (3×5-fold)  |  15   | 0.883 ± 0.094     | **0.835 ± 0.161** | **0.900 ± 0.141** | 0.851 ± 0.115     | 0.885 ± 0.098     | **0.966 ± 0.048** |

| mode                 | AUC                | Sens               | Spec               | Acc                | F1                 | Prec               |
|---|---|---|---|---|---|---|
| `mode_gcn`           | 0.872 ± 0.101      | 0.865 ± 0.127      | 0.853 ± 0.170      | 0.862 ± 0.095      | 0.898 ± 0.079      | 0.948 ± 0.055      |
| `mode_hybrid`        | 0.886 ± 0.090      | 0.838 ± 0.128      | **0.936 ± 0.117**  | 0.865 ± 0.088      | 0.897 ± 0.075      | **0.979 ± 0.038**  |
| **`mode_radiomics`** | 0.885 ± 0.088      | **0.895 ± 0.101**  | 0.862 ± 0.146      | **0.887 ± 0.075**  | **0.919 ± 0.059**  | 0.954 ± 0.047      |

![Topology fingerprint UMAP (gmm_k2, best label-aligned clustering ARI +0.091)](followup_experiments/outputs/cluster_topology/umap_topology.png)

![Vascular-full UMAP (kmeans_k2, sil 0.72 is outlier-vs-bulk artefact)](followup_experiments/outputs/cluster_topology/umap_vascular_full.png)

Key findings:
- Label leakage from the 197-vs-106 mismatch is modest rather than dominant: the follow-up delta is about `ΔAUC ≈ 0.05`, not a collapse.
- Reintroducing the `mPAP` auxiliary triple with Youden calibration restores specificity to `0.96` on `medium_youden`.
- Per-fold Youden on the repeated `3×5` CV run recovers sensitivity to `0.83` while keeping specificity high enough to stay competitive with Sprint 5.
- `RadiomicsMLP` is the surprise leader on sensitivity and F1 in the 3-mode ablation, despite using no graph topology.
- Vessel topology shows reproducible unsupervised phenotypes, but they are largely orthogonal to PH labels: best label-aligned clustering is only `ARI +0.091`.

Full details: [followup_experiments/outputs/comparison_full_report.md](followup_experiments/outputs/comparison_full_report.md)
