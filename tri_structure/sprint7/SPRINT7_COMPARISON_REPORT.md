# Sprint 7 Comparison Report — Task 6

Generated 2026-04-20. Plan: `4月20日改进的部分/SPRINT7_IMPROVEMENT_PLAN (1).md`.
Source data: `outputs/sprint7_phase2/cv_results.json`, `outputs/sprint7_sweep_edrop/*/cv_results.json`,
`outputs/sprint7_phase2/sprint7_phase2.log`, `outputs/sprint7_phase2/cluster_analysis.json`.
Figures: [`outputs/figures/`](outputs/figures/).

## 1. Scope

Sprint 7 changed two axes relative to tri-structure v2 (2026-04-19):

1. **Cache**: heuristic unified partition → real per-structure cache with three fixes
   (largest-component filter, airway `<3`-node fallback, Strahler default = 1).
2. **Regularisation**: added three knobs — `edge_drop_p`, `label_smoothing`, cosine
   `warmup_epochs` — with new CLI flags wired through `src/tri_structure_pipeline.py`.

Phase 1 training config (pool=mean, no attention pooling, no per-case signature, no regularisation)
is reused as the reference. v2 numbers come from [`tri_structure/RESULTS.md`](../RESULTS.md).

## 2. Edge-dropout sweep (Task 3)

5-fold CV, 1 repeat, 200 epochs, `label_smoothing=0.0`, `warmup_epochs=0`.

| p | AUC (mean ± std) |
|---|---|
| 0.00 | 0.673 ± 0.110 |
| 0.05 | 0.678 ± 0.166 |
| **0.10** | **0.721 ± 0.160** |
| 0.15 | 0.694 ± 0.106 |

The curve is essentially flat within one standard deviation (the spread is ~0.16 vs a
point-to-point range of ~0.05). p = 0.10 was carried into Task 5 as the best point, but the
choice is not statistically robust.

![edge-dropout sweep](outputs/figures/sweep_edrop_auc.png)

## 3. Phase 2 full training (Task 5) — 6-metric comparison

Sprint 7 config: `edge_drop_p=0.10`, `label_smoothing=0.1`, `warmup_epochs=20`, `mpap_aux`,
`use_signature`, `pool=mean`, 3 × 5-fold CV (15 folds total, n = 106).

| Metric | Phase 1 (mean pool) | v2 (attn + signatures) | **Sprint 7 (tri-cache + reg.)** | Plan target |
|---|---|---|---|---|
| AUC | 0.880 ± 0.093 | 0.734 ± 0.142 | **0.729 ± 0.125** | ≥ Phase 1 |
| Accuracy | 0.877 ± 0.089 | 0.739 ± 0.115 | 0.743 ± 0.129 | ≥ Phase 1 |
| Sensitivity | 0.869 ± 0.113 | 0.708 ± 0.144 | 0.719 ± 0.210 | — |
| Specificity | 0.900 ± 0.119 | 0.824 ± 0.166 | 0.818 ± 0.200 | — |
| F1 | 0.910 ± 0.070 | 0.794 ± 0.101 | 0.789 ± 0.129 | — |
| Precision | 0.964 ± 0.041 | 0.926 ± 0.064 | 0.933 ± 0.067 | — |

![6-metric grouped bar](outputs/figures/phase2_vs_phase1_auc.png)

Sprint 7 is statistically indistinguishable from v2 on every metric (overlapping std) and
~15 AUC points below Phase 1. Regularisation did not recover the performance gap opened by
moving off the heuristic partition.

## 4. Per-structure attention vs mPAP

Pearson r of per-structure attention weight and mPAP (absolute value used for the plan's
threshold; signs below are raw from the log).

| Structure | Phase 1 \|r\| | Sprint 7 r | Sprint 7 \|r\| | Plan threshold |
|---|---|---|---|---|
| artery | 0.486 | −0.073 | 0.073 | ≥ 0.45 |
| vein | — (not reported) | +0.175 | 0.175 | — |
| airway | 0.468 | −0.042 | 0.042 | ≥ 0.50 (investigate if < 0.40) |

The Phase 1 signal collapses. Airway's correlation drops from 0.468 to 0.042 — an order of
magnitude below the plan's own 0.40 investigation threshold.

![attention × mPAP](outputs/figures/attention_mpap_sprint7.png)

Per-class mean attention (normalised; pipeline log):

| Class | artery | vein | airway |
|---|---|---|---|
| COPD (non-PH) | 0.572 | 0.188 | 0.240 |
| COPD-PH | 0.540 | 0.219 | 0.241 |

The model spreads attention nearly identically across both classes — there is no learned
structural preference that distinguishes PH from non-PH, which is consistent with the near-zero
attention-mPAP correlations.

## 5. Clustering on fused embeddings

From `cluster_analysis.json` / pipeline log (z_fused and per-case 24-D signature):

| View | Algorithm | k | ARI (label) | NMI | Silhouette | Sizes |
|---|---|---|---|---|---|---|
| embedding | kmeans | 2 | +0.0044 | 0.0004 | 0.1782 | [22, 84] |
| embedding | kmeans | 3 | −0.0225 | 0.0045 | 0.1944 | [67, 17, 22] |
| embedding | gmm | 2 | +0.0003 | 0.0006 | 0.1370 | [69, 37] |
| signature | kmeans | 2 | −0.0021 | 0.0000 | 0.5687 | [4, 102] |
| signature | gmm | 3 | +0.0047 | 0.0005 | 0.3409 | [84, 18, 4] |
| hybrid | kmeans | 2 | +0.0044 | 0.0004 | 0.1318 | [22, 84] |

Every ARI is within noise of zero (|ARI| ≤ 0.036); the high-silhouette signature clusters are
driven by a 4-case outlier group, not biology. Plan target (ARI > 0.15 on z_fused) is missed
by ~2 orders of magnitude.

## 6. Plan success criteria — scorecard

| # | Criterion | Target | Sprint 7 | Met? |
|---|---|---|---|---|
| 1 | AUC vs Phase 1 | ≥ 0.880 | 0.729 | no |
| 2 | \|r(artery, mPAP)\| | ≥ 0.45 | 0.073 | no |
| 3 | \|r(airway, mPAP)\| | ≥ 0.50 | 0.042 | no |
| 4 | ARI on z_fused | > 0.15 | 0.004 | no |
| 5 | Silhouette on z_fused | ≥ 0.20 | 0.18 | marginal |
| 6 | Airway quality gate | \|r(airway, mPAP)\| ≥ 0.40 else investigate | 0.042 | **investigation triggered** |

Five of six failed; criterion 6 is the plan's own escape hatch and is now active.

## 7. Interpretation

**The regression is in the cache, not the regulariser.** The Task 3 sweep varied `edge_drop_p`
on the same cache and all four points sit within ±0.025 AUC of each other — regularisation
has no leverage. The large gap to Phase 1 is locked in by whatever the per-structure cache
does differently from the heuristic unified partition.

**Candidate explanations, in order of plausibility:**

1. **Airway segmentation is too noisy.** QA reported 18.87% of airway graphs fall back to the
   `<3`-node path; the remaining 81.13% were kept by the largest-component filter which silently
   drops disconnected airway branches. Phase 1's unified partition did not make this cut, so
   Phase 1 actually saw *more* airway topology per case, even if mis-labelled.
2. **Largest-component filter on artery/vein discards real disease signal.** PH cases are
   expected to have pruned distal vasculature — the largest-component step will preferentially
   remove exactly the fragmented distal branches that differentiate PH from non-PH. This is
   a specification bug in the cache, not a data issue.
3. **Strahler default = 1 flattens the generation feature** for any node that could not resolve
   an order. In the unified partition this happened implicitly via connectivity; here it happens
   statically on ~hundreds of airway nodes per case.

(1) and (2) are testable with ablations on the existing cache; (3) requires a cache rebuild.

## 8. Recommended next steps

The plan's guidance is to halt and investigate before any further training on this cache.
Concrete experiments in priority order:

1. **Per-structure node retention audit.** For each case, compare `n_nodes` in
   `cache_tri/{case}.pkl` vs the old unified cache. Histogram the ratio separately by
   artery / vein / airway. If any structure retains <70% of nodes on average, the
   largest-component filter is the culprit (explanation 2).
2. **Phase 1 config on the new cache.** Re-run the Phase 1 reference (mean pool, no
   attention pooling, no signature, no regularisation) on `cache_tri/` alone. If it still
   regresses, the cache is the problem; if it recovers, attention pooling + signatures
   interact badly with the new cache.
3. **Disable the largest-component filter for artery/vein** and keep it only for airway. One
   retrain, 5 × 3 CV, same other settings as Task 5. Expected to recover 5–10 AUC points if
   explanation (2) holds.
4. **Airway segmentation spot-check.** Render the 20 cases with the largest drop in airway
   node-count vs the old cache against their CT volumes; visual sanity-check whether distal
   airway branches are being segmented at all.

Only after (1)–(4) resolve the airway quality gate should the Sprint 7 regularisation
(edge dropout, label smoothing, warmup) be re-evaluated — at present, all three are masked
by the cache regression.

## 9. Bottom line

Phase 1 (heuristic unified partition, mean pool, no regularisation) remains the production
reference configuration. Sprint 7's two changes (real per-structure cache, new regularisers)
went in together; ablation (Task 3 sweep) isolates the regression to the cache. Do not
adopt Sprint 7 until the airway quality gate clears.
