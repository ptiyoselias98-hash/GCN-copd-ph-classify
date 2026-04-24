# ARIS Autonomous Loop Plan (Rounds 5–40)

**Target** (updated 2026-04-24): score ≥ **9.5/10**. If not reached by
Round **40**, stop and report. Earlier target of 8/10 was raised by user.

Each cron fire does ONE round. The Claude instance MUST:

1. Read `review-stage/REVIEW_STATE.json`. Determine `round_to_execute = max_round_in_history + 1`.
2. Pull latest `git pull --ff-only origin main`.
3. Execute the experiments queued for that round (see per-round plan below).
4. Commit + push deliverables.
5. Invoke codex via `mcp__codex__codex` with `model=gpt-5.2`, `approval-policy=never`, `sandbox=read-only`, `cwd=copdph-gcn-repo`, `config={"model_reasoning_effort":"high"}`. Context: REVIEWER_MEMORY.md + context_round{N}.md + artifact list.
6. Parse codex verdict → update REVIEW_STATE.json + AUTO_REVIEW.md + REVIEWER_MEMORY.md.
7. If `score >= 8` → delete the cron job (`CronDelete`), commit a "target reached" flag file, exit.
8. If `round >= 20` and still < 8 → write a final `review-stage/FINAL_REPORT.md`, delete cron, exit.
9. Otherwise → exit (cron fires again later).

## No-confirm requirements

- NEVER use HEREDOC inside `ssh "..."` (triggers safety prompt). Instead: write script locally with `Write`, `scp` to `/tmp/`, then `ssh bash /tmp/script.sh`.
- NEVER use newlines inside single-quoted SSH arguments.
- For git commits: always use `-m "..."` with single-quoted message (no HEREDOC).
- For remote job launches: `ssh imss@10.60.147.117 "nohup bash /tmp/X.sh > /tmp/X.log 2>&1 &"` is one line — no prompt.
- Codex MCP has explicit `approval-policy=never` set.

## Per-round plan

### Round 5 — DeLong + GCN-level protocol test

**Prereqs**: GPU 1 arm_b + arm_c contrast-only prob dumps completed (started 2026-04-23 ~18:30).

Experiments:
- `scripts/evolution/R5_delong.py` — fetch remote probs, compute case-level DeLong paired arm_c − arm_b contrast-only (PRIMARY ENDPOINT). Report p + 95% CI on Δ-AUC.
- `scripts/evolution/R5_gcn_feature_protocol.py` — on remote, compute per-case graph-level aggregates from `cache_v2_tri_flat/*.pkl` (node count, mean degree, mean_diameter, mean_tortuosity, max_strahler, edge_len_p90, pooled for artery+vein+airway). Train protocol classifier WITHIN nonPH only.

Commit → push → codex Round 5.

### Round 6 — TEASAR parameter sensitivity

- Launch on GPU 1 remote: rebuild `cache_v2_tri_flat_scale{0.8,1.0,1.5}_const{2,5,10}` on a pre-registered 20-case random subset (seed 20260423).
- Compute per-case skeleton-to-mask coverage, num_components, total SL_mm per combo. Produce a stability table (±σ across 9 hyperparam settings).
- Report: median AUC change under hyperparam variation. Reviewer expects <0.02.

Commit → push → codex Round 6.

### Round 7 — Domain-adversarial arm

- Patch `gcn_models.py` (remote) to add a gradient-reversal head predicting `is_contrast`. Tune λ in {0.1, 0.5, 1.0}.
- Train arm_b_adv on GPU 1 with --adv_lambda=0.5. 5-fold × 3 repeats × 120 epochs on the 189 contrast-only cohort.
- Measure both disease AUC and within-nonPH protocol AUC on the penultimate embeddings.

Commit → push → codex Round 7.

### Round 8 — Propensity-matched primary endpoint

- Match 26 contrast nonPH to 26 contrast PH on age, sex, smoking pack-years (from `copd-ph患者113例0331.xlsx`).
- Retrain arm_b on the 52 matched cases. 5-fold × 3 repeats.
- DeLong on matched-vs-unmatched Δ-AUC. Propensity-score overlap diagnostics.

Commit → push → codex Round 8.

### Round 9 — Locked reproducibility + kimimaro pin

- `ssh imss@... "source /home/imss/miniconda3/etc/profile.d/conda.sh && conda activate pulmonary_bv5_py39 && pip show kimimaro | grep Version && conda env export --no-builds > environment.lock.yml"`
- `scp` environment.lock.yml back to repo.
- Patch `_remote_build_v2_cache.py` to record `git_sha` + `kimimaro.__version__` in each pkl; rebuild a single test case to verify.
- Update `scripts/cache_provenance.py` output.

Commit → push → codex Round 9.

### Round 10–12 — Whatever the last codex review asks

Read the Round-9 `must_fix_before_next_round` list. Prioritize by severity.
Implement 1–2 items per round. Commit + codex after each.

### Round 13–20 — Diminishing returns

If Round 12 still < 8, likely the remaining blockers are structural (need more data,
external validation cohort). Record honest limitations in `REPORT_v2.md §Conclusions`
and stop at Round 20 with a full FINAL_REPORT.md.

## State file schema

`review-stage/REVIEW_STATE.json`:
```
{
  "round": <last completed>,
  "max_rounds": 20,
  "positive_threshold_score": 8,
  "target_reached": <bool>,
  "history": [{round, score, verdict, delta, must_fix, ...}, ...],
  "running_gpu_jobs": [...],
  "next_round_plan": "<free text>"
}
```

## Failure modes

- **GPU 1 job crashed**: inspect `/tmp/gpu1_*.log` via ssh; fix and relaunch. If fix not obvious, SKIP the experiment for that round and note it in REPORT_v2 as a pending item. Never block on a single experiment.
- **Codex MCP at capacity**: retry once with `gpt-5.2` + no reasoning_effort override; if still capacity, try `gpt-5.2-codex`; if still failing, write a "CODEX_UNAVAILABLE_<timestamp>.md" note and skip to the next cron fire.
- **Git push conflict**: `git pull --rebase origin main` and retry. If still conflict, abort and write a `MANUAL_MERGE_NEEDED.md` marker.
- **SSH unreachable**: skip remote work this round, do local-only tasks, and note in REPORT_v2.

## Stop conditions (updated 2026-04-24)

- `score >= 9.5`: delete cron, commit `review-stage/TARGET_REACHED.md` with round + score.
- `round >= 40` and `score < 9.5`: delete cron, commit `review-stage/FINAL_REPORT.md` with honest limitations.
- `score == 10`: same as ≥9.5 stop.

## Pre-approved operations (user confirmed 2026-04-24)

See `AUTONOMOUS_CRON_PROMPT.md` for the definitive list. Summary:
- SSH/scp + launch training on GPU 0 AND GPU 1 anytime
- git commit/push (never force, never amend)
- codex MCP (gpt-5.2 high, approval-policy=never)
- CronDelete on stop condition
- Multi-hour remote jobs (kimimaro rebuilds, TEASAR sweeps, adversarial arms)

## Blocker policy

Auto-try all plausible alternatives silently. Do NOT write
NEEDS_USER_INPUT markers. If a round can't deliver one fix, commit an
honest-negative sub-section in REPORT_v2 and let the next cron pick up.
