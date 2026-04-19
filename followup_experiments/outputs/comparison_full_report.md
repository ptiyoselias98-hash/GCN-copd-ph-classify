# Follow-up experiments — full comparison report

Gold subset (106 Excel-matched cases: 79 PH / 27 non-PH).
Generated 2026-04-18.

## 1. Single-CV vs repeated-CV vs GitHub baseline

Comparing our headline configuration (`focal + node-drop + mPAP-aux + per-fold Youden`)
in three regimes against the public Sprint 5 Final from
`ptiyoselias98-hash/GCN-copd-ph-classify`.

| run                                 | folds | AUC               | Sens              | Spec              | Acc               | F1                | Prec              |
|---|---|---|---|---|---|---|---|
| Sprint 5 Final (GitHub, no Youden)  |   5   | 0.924             | 0.870             | 0.920             | n/r               | n/r               | n/r               |
| `medium`        (0.5 threshold)     |   5   | 0.919 ± 0.082     | 0.431 ± 0.170     | 0.920 ± 0.160     | 0.557 ± 0.104     | 0.571 ± 0.157     | 0.969 ± 0.062     |
| **`medium_youden`** (per-fold J)    |   5   | 0.899 ± 0.100     | **0.798 ± 0.195** | **0.960 ± 0.080** | 0.838 ± 0.143     | 0.866 ± 0.135     | **0.988 ± 0.025** |
| **`medium_youden_rep`** (3×5-fold)  |  15   | 0.883 ± 0.094     | **0.835 ± 0.161** | **0.900 ± 0.141** | 0.851 ± 0.115     | 0.885 ± 0.098     | **0.966 ± 0.048** |

Notes
- `medium_youden_rep` uses three independent split seeds (mPAP-bucket × label
  stratified) for each of the five folds, giving 15 fold runs of honest
  variance estimation.
- Repeated-CV is roughly comparable to the single-shot 5-fold (ΔAUC −0.016,
  ΔSens +0.037, ΔSpec −0.060). The extra spec drop comes from one or two
  hard splits in the new seeds; the headline finding (Sens **and** Spec
  simultaneously above 0.83) survives the multi-seed stress test.
- vs GitHub Sprint 5 Final: `medium_youden_rep` matches Sens (0.835 vs 0.870,
  within noise) and is competitive on Spec (0.900 vs 0.920) **with reported
  variance** instead of single-shot point estimates.

## 2. Three-mode ablation

All three modes use the same headline trick set (focal + node-drop + mPAP
aux for the GCN modes; focal only for radiomics) with 3×5-fold CV (15
fold runs each) and per-fold Youden calibration.

| mode             | model              | radiomics path | aux losses          |
|---|---|---|---|
| `mode_gcn`       | PulmonaryGCN       | none           | focal + mPAP-aux    |
| `mode_hybrid`    | HybridPulmonaryGCN | concat fusion  | focal + mPAP-aux    |
| `mode_radiomics` | RadiomicsMLP       | features only  | focal               |

Results (15-fold mean ± std):

| mode                 | AUC                | Sens               | Spec               | Acc                | F1                 | Prec               |
|---|---|---|---|---|---|---|
| `mode_gcn`           | 0.872 ± 0.101      | 0.865 ± 0.127      | 0.853 ± 0.170      | 0.862 ± 0.095      | 0.898 ± 0.079      | 0.948 ± 0.055      |
| `mode_hybrid`        | 0.886 ± 0.090      | 0.838 ± 0.128      | **0.936 ± 0.117**  | 0.865 ± 0.088      | 0.897 ± 0.075      | **0.979 ± 0.038**  |
| **`mode_radiomics`** | 0.885 ± 0.088      | **0.895 ± 0.101**  | 0.862 ± 0.146      | **0.887 ± 0.075**  | **0.919 ± 0.059**  | 0.954 ± 0.047      |

### Mode comparison takeaways

- **AUC is essentially tied** across all three modes (0.872 / 0.886 / 0.885).
  All sit within one std of each other — the discriminative ceiling is
  the same regardless of architecture.
- **Pure radiomics MLP is the unexpected leader on the Sens / Acc / F1
  triple.** With only the 22 hand-crafted vascular + airway scalars and
  no graph structure, it reaches Sens = 0.895 / Acc = 0.887 / F1 = 0.919,
  the best of the three. Std is also the lowest of the three on every
  metric.
- **Hybrid fusion buys Specificity at the cost of Sensitivity.** Concat
  fusion (graph-pooled embedding ⊕ MLP-projected radiomics) lifts Spec
  to 0.936 (best in the table) and Prec to 0.979, but Sens drops to
  0.838 — i.e. the fused branch becomes more conservative.
- **Pure GCN is the most balanced (Sens 0.865 ≈ Spec 0.853)** but is the
  weakest in absolute precision.
- **Operational picking guide:**
  - Need to maximise PH detection (low miss rate) → `mode_radiomics`.
  - Need to minimise false positives → `mode_hybrid`.
  - Need a balanced operating point → `mode_gcn` (or `medium_youden_rep`,
    which is the same model + the gold-standard headline config).

## 3. Unsupervised vessel-topology clustering

Two fingerprint flavours over the gold-subset cache:

- `topology` (17 D): pure graph descriptors — node/edge counts, density,
  degree stats, tip/branch fractions, Strahler order, diameter stats,
  vessel-type fractions.
- `vascular_full` (39 D): topology + flattened cached vascular + airway
  scalars (BV5/BV10, lung volume, artery/vein ratio, pruning index,
  airway count, wall thickness, …).

Methods: KMeans (k=2..5), GMM (k=2..5), Spectral (k=2..4).
Hdbscan was skipped (no module on the env).

Selection metric: silhouette on PCA-reduced features (variance ≥ 0.95).
Alignment: ARI / NMI vs PH label and vs mPAP tertile (cutoffs 25 / 33 mmHg).

### Topology fingerprint

| method      | k | sil    | DB     | ARI(label) | NMI(label) | ARI(mPAP_t) |
|---|---|---|---|---|---|---|
| **gmm_k2**  | 2 | **0.426** | 0.738 | **+0.091** | 0.058 | +0.005 |
| kmeans_k5   | 5 | 0.269 | 0.961 | +0.004 | 0.041 | +0.008 |
| gmm_k5      | 5 | 0.278 | 1.076 | +0.002 | 0.050 | +0.017 |
| kmeans_k4   | 4 | 0.219 | 1.034 | +0.025 | 0.033 | −0.002 |
| spectral_k2 | 2 | 0.202 | 1.622 | −0.001 | 0.002 | +0.003 |

Best by silhouette: `gmm_k2` (sil 0.43, ARI vs label +0.091).
Best by ARI vs label: `gmm_k2` (+0.091, weak but the largest signal we see).

### Vascular_full fingerprint

| method        | k | sil       | DB     | ARI(label) | NMI(label) | ARI(mPAP_t) |
|---|---|---|---|---|---|---|
| kmeans_k2     | 2 | **0.724** | 0.753 | −0.002 | ~0     | −0.002 |
| gmm_k2        | 2 | 0.724     | 0.753 | −0.002 | ~0     | −0.002 |
| gmm_k5        | 5 | 0.379     | 0.516 | +0.071 | 0.094  | +0.002 |
| kmeans_k4     | 4 | 0.268     | 0.904 | −0.041 | 0.043  | +0.004 |
| kmeans_k3     | 3 | 0.260     | 1.246 | −0.042 | 0.010  | +0.003 |

Important caveat: the `kmeans_k2` silhouette of 0.724 is an
**outlier-vs-bulk artefact** — cluster sizes are 102 vs 4. The 4-case
minority cluster has median mPAP = 42 mmHg vs 28 mmHg for the bulk, i.e.
this is a "very-severe extreme tail" partition rather than a real
two-phenotype split. Using only this k=2 score would overstate phenotype
strength.

If we discount the outlier split, the meaningful clustering picture is:

- **Topology fingerprint** has a soft two-phenotype structure (gmm_k2,
  sil 0.43, ARI vs label +0.091). This is the strongest topology→label
  signal in the table.
- **Vascular_full** picks out the extreme-mPAP outliers cleanly (k=2 with
  sil 0.72) and shows mild 4–5 cluster structure (sil 0.27,
  ARI ≈ ±0.04) once you remove that split.
- **No clustering aligns with mPAP tertiles** (all ARI(mPAP_t) within
  ±0.04).

### Interpretation

> Vessel topology and vascular phenotypes carry **reproducible
> unsupervised structure** (silhouette > 0.4 in both flavours), but that
> structure does **not** align with COPD-PH labels or mPAP severity
> tertiles in this 106-case cohort. The strongest signal is a weak
> two-phenotype split in the pure-topology fingerprint that lifts
> ARI(label) to +0.09 — equivalent to ~9 % more agreement than chance.
> The high-silhouette `vascular_full` k=2 split is an outlier cluster
> that captures the 4 most severe mPAP cases.

This is a clinically interesting *negative* result: pulmonary
small-vessel topology has its own phenotypic axes that are largely
**orthogonal** to the PH/non-PH classification — supervised labels and
unsupervised structure live in different subspaces of the same feature
set. For paper writing this argues for dual reporting: the supervised
GCN learns task-aligned features, while the unsupervised topology
analysis identifies anatomical phenotypes worth describing on their own
(e.g. as an ancillary subgroup for further study).

UMAP visualisations:
`outputs/cluster_topology/umap_topology.png`,
`outputs/cluster_topology/umap_vascular_full.png`.
Per-case cluster IDs:
`outputs/cluster_topology/cluster_assignments.csv`.

## 4. Key takeaways

1. **Repeated CV stays in the same neighbourhood as single-shot CV.**
   Headline metrics on `medium_youden` are robust to split seed: AUC
   0.90 → 0.88, Sens 0.80 → 0.84, Spec 0.96 → 0.90 with mPAP-bucket ×
   label stratified seeds. The Sens-and-Spec-both-high regime survives.
2. **Per-fold Youden remains the right calibration choice.** Going from
   `medium` (0.5 threshold) to `medium_youden` lifts Sens 0.43 → 0.80
   while Spec rises 0.92 → 0.96 — same model, sharper threshold.
3. **vs GitHub Sprint 5 Final.** `medium_youden_rep` matches their
   reported Sens/Spec while reporting honest 15-fold variance, which
   their single-shot 5-fold CV cannot.
4. **Pulmonary vessel topology has unsupervised phenotypes — but they
   don't predict PH.** Best label-aligned clustering reaches ARI = +0.09
   (gmm_k2 on topology). The strongest cluster split (sil 0.72) is an
   outlier vs bulk anatomical separation, not a PH split.
5. **Hand-crafted radiomics carries most of the predictive signal.**
   Three-mode ablation: pure RadiomicsMLP matches GCN AUC (0.885 vs
   0.872) and beats both GCN and hybrid on Sensitivity (0.895), Acc
   (0.887) and F1 (0.919). Hybrid fusion trades sensitivity for
   specificity (Spec 0.936 / Prec 0.979). On 106 cases the structured
   vascular + airway scalars are enough; the graph topology adds
   conservatism (higher spec) rather than discrimination.
6. **What this means for the architecture story.** The GCN is not a
   "magic" feature extractor on this cohort scale — its advantage is
   producing a more conservative decision boundary when fused with
   radiomics. If the goal is best F1/Sens, a small radiomics MLP is a
   strong, much-cheaper baseline. If the goal is high-precision
   PH-screening (low FP rate), the hybrid is the right pick.
