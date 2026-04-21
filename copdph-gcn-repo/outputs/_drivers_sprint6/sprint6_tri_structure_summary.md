# Sprint 6 — tri-structure GCN, 6-job parallel run + LR-sweep follow-up

_run date: 2026-04-21 ;  cohort sources: gold n=106, expanded n=269 (after cache conversion)_

## Headline

**Best model so far**: `p_zeta_sig` — tri-structure GCN on **n=269** with mean pool, mPAP aux, `use_signature=True`.

| metric | value |
|---|---|
| AUC | **0.9232 ± 0.034** |
| Accuracy | 0.898 ± 0.030 |
| Sensitivity | 0.933 ± 0.067 |
| Specificity | 0.844 ± 0.088 |
| Precision | 0.908 ± 0.047 |
| F1 | 0.918 ± 0.027 |

vs. yesterday's best (`arm_a_ensemble`, AUC 0.944): the tri_structure approach on **n=269** lands ~2 pts below the radiomics-ensemble baseline, but with substantially lower variance and a single end-to-end pipeline (no manual feature stack). The gap is ~1 σ — not yet a clear winner, but a viable end-to-end architecture.

## All 6 jobs (5-fold × 3-rep CV, mpap_aux always on)

| job | n | pool | sig | AUC | Acc | Sens | Spec | F1 |
|---|---|---|---|---|---|---|---|---|
| **p_zeta_sig**     | **269** | mean | ✓ | **0.923 ± 0.034** | 0.898 | 0.933 | 0.844 | 0.918 |
| p_zeta_attn        | 269 | attn | – | 0.917 ± 0.026 | 0.890 | 0.939 | 0.813 | 0.911 |
| p_zeta_tri_282     | 269 | mean | – | 0.917 ± 0.027 | 0.885 | 0.941 | 0.797 | 0.908 |
| p_eta_pool_attn    | 106 | attn | – | 0.697 ± 0.095 | 0.698 | 0.665 | 0.811 | 0.749 |
| p_eta_pool_add     | 106 | add  | – | 0.682 ± 0.110 | 0.755 | 0.781 | 0.676 | 0.818 |
| p_eta_sig          | 106 | mean | ✓ | 0.672 ± 0.102 | 0.701 | 0.662 | 0.811 | 0.750 |

### What this tells us

1. **Cohort size dominates everything else.** Going from n=106 → n=269 lifts AUC from ~0.68 to ~0.92 with the *same* architecture. The expanded cohort isn't just "more data" — it crosses a stability threshold where the tri-structure attention model can fit reliably.
2. **Pool mode is a wash on n=269.** mean (0.917) ≈ attn (0.917). attn pool was *catastrophic* on n=106 (0.697) — likely because attention layers need more samples per parameter.
3. **Signature view adds a real, small bump on n=269.** +0.006 AUC over mean baseline (0.923 vs 0.917). Worth keeping, but not a regime change.
4. **`add` pool on n=106 is the worst pool variant.** F1 stayed high (0.818) because of high sensitivity, but AUC collapsed (0.682) — sign of poor calibration.

### Sensitivity vs. specificity profile (n=269 group)

All three n=269 variants prefer high sensitivity (~0.93-0.94) over specificity (~0.80-0.84). Useful for **screening** (don't miss PH). For confirmatory diagnosis you'd want higher specificity — `p_zeta_sig` is best on that axis (0.844 vs 0.797).

## LR-sweep follow-up (4/4 done)

Auto-launched at 12:45:47 by the watcher script the moment all 6 done.flag files appeared. Initial 4-way parallel scheme caused 2× slowdown via GPU contention; pivoted at 13:00 to **kill 2 lr2x jobs and re-run sequentially solo on each GPU** (kept lrhalf jobs that had more progress). Sequential watcher fired correctly on each lrhalf flag.

Why LR-sweep and not `--batch_size 32`: the pipeline trains **per-sample** (no DataLoader), so VRAM is architecturally capped at ~3GB regardless of any batch_size value. A real batched implementation needs a model rewrite (PyG `Batch.from_data_list` + forward changes) — not safe under the "quality first" instruction. LR sweep around the 1e-3 default is the equivalent honest probe.

### Results

| job | n | lr | AUC | std | F1 | Acc | Sens | Spec |
|---|---|---|---|---|---|---|---|---|
| **p_theta_269_lr2x**   | 269 | 2e-3 | **0.928** | 0.027 | 0.907 | 0.886 | 0.927 | 0.822 |
| p_theta_269_lrhalf | 269 | 5e-4 | 0.908 | 0.031 | 0.912 | 0.890 | 0.945 | 0.803 |
| p_theta_106_lr2x   | 106 | 2e-3 | 0.699 | 0.124 | 0.749 | 0.701 | 0.655 | 0.840 |
| p_theta_106_lrhalf | 106 | 5e-4 | 0.632 | 0.079 | 0.771 | 0.704 | 0.706 | 0.707 |

### LR sensitivity at n=269 (mean pool + mpap_aux, clean comparison)

| lr | AUC | std | rank |
|---|---|---|---|
| 5e-4 (half) | 0.908 | 0.031 | 3 |
| 1e-3 (default) | 0.917 | 0.027 | 2 |
| **2e-3 (double)** | **0.928** | **0.027** | **1** |

**Higher LR wins** — lr=2e-3 lifts AUC by ~0.011 over baseline with same variance. This makes lr=2e-3 the new tri_structure best for n=269 (still below arm_a_ensemble 0.944, but the gap shrinks to ~1.6 pts).

### LR sensitivity at n=106 (mean pool + mpap_aux)

| lr | AUC | std |
|---|---|---|
| 5e-4 (half) | 0.632 | 0.079 |
| 2e-3 (double) | 0.699 | 0.124 |

On the small cohort, lr=2e-3 beats lr=5e-4 by ~7 pts but with substantially higher variance. Both are far below n=269 numbers — confirms **n=106 cohort is the binding constraint, not LR or pool variant**.

## Final updated headline (replaces top-of-doc estimate)

| rank | model | n | AUC | F1 | Sens | Spec |
|---|---|---|---|---|---|---|
| 1 | arm_a_ensemble (yesterday baseline) | 113 | 0.944 | – | – | – |
| 2 | **p_theta_269_lr2x** (best tri_structure) | 269 | **0.928 ± 0.027** | 0.907 | 0.927 | 0.822 |
| 3 | p_zeta_sig (signature view, default lr) | 269 | 0.923 ± 0.034 | 0.918 | 0.933 | 0.844 |
| 4 | p_zeta_tri_282 / p_zeta_attn (default lr) | 269 | 0.917 | 0.908 / 0.911 | 0.939 / 0.94 | 0.81 / 0.81 |

**Promoted config for next sprint**: tri_structure_pipeline on n=269 with `--lr 2e-3 --mpap_aux --pool_mode mean` → `p_theta_269_lr2x`. Optionally add `--use_signature` for the +0.005 AUC if compute is free.

## Verdict — what to keep, what to retire

**Keep / promote to canonical:**
- `tri_structure_pipeline.py` on n=269 (cache_tri_converted) with `--mpap_aux --use_signature` → `p_zeta_sig` config.
- Compare end-to-end against `arm_a_ensemble` on a held-out test set (next sprint).

**Retire from rotation (n=106 cohort):**
- `pool_mode=attn` and `pool_mode=add` on n=106 — clearly under-fit.
- `use_signature` on n=106 — same regime, no benefit.

**Open questions (LR sweep will answer):**
- Does lr=2e-3 close the 0.02 gap to arm_a_ensemble?
- Or does lr=5e-4 buy more stability without losing performance?
