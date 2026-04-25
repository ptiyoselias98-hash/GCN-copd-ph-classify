# ARIS Round 11 Context — bug-fixes from R10

History: R1=2, R2=3, R3=4, R4=5, R5=6, R6=5, R7=5, R8=6, R9=6, R10=6.2.
Target ≥ 9.5/10.

## Round 10 reviewer-flagged bugs (now fixed)

1. **λ double-scaling** — old GRL applied λ in `GradReverse.backward`
   AND multiplied loss by λ → encoder gradients scaled ~λ². Fixed in
   `run_sprint6_v2_grl_fix.py`: GRL coef hard-set to 1.0; only `λ * adv_loss`
   on the loss.
2. **Objective mismatch** — old adversary trained on full cohort where
   PH ≈ contrast, so GRL incentivised erasing disease signal rather than
   protocol leakage. Fixed: per-batch mask `y == 0` to train adversary
   on **nonPH-only** samples, matching the within-nonPH evaluation target.
3. **No adversary diagnostics** — fixed: per-epoch batch AUC of the
   adversary head logged to run.log.

## R11 deliverables

- 4 λ values × 3 seeds = 12 runs, GPU 0 + GPU 1 in parallel.
- λ ∈ {0.0, 1.0, 5.0, 10.0}; seeds ∈ {42, 1042, 2042}.
- Both LR + small MLP probes for protocol decoding (worst-case leakage check
  per Round 10 reviewer).
- Cohort reconciliation table (`outputs/r11/cohort_reconciliation.md`)
  documenting full 282 vs in-cache 243 with missingness-by-protocol.
- Multi-seed mean ± SD for protocol AUC + disease AUC.

## Open question for Round 11 reviewer

Does the corrected GRL (single-scaled λ, nonPH-only adversary, 3-seed,
LR+MLP probe) achieve a meaningfully different protocol-AUC trajectory
than R10? If yes — what's the new minimum-to-9.5? If still stuck, what's
the path beyond GRL (different objective, more data, alternative arch)?
