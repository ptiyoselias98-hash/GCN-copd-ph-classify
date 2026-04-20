# Sprint 7 Results — Tri-structure GCN with per-structure cache

Run date: 2026-04-20
Plan: `4月20日改进的部分/SPRINT7_IMPROVEMENT_PLAN (1).md`
Remote workspace: `/home/imss/cw/GCN copdnoph copdph/sprint7/`
Local automation: `sprint7_automation/` (see `automation/` here)

## Summary

Sprint 7 rebuilt a per-structure (artery / vein / airway) skeleton cache with three fixes
(largest-component filter, airway <3-node fallback, Strahler default 1) and added three
regularisation knobs (edge dropout, label smoothing, cosine warmup). Phase 2 training on the
new cache **regressed** relative to the Phase 1 benchmark that used the heuristic unified
partition.

## Task-by-task outcome

| Task | What | Result |
| --- | --- | --- |
| 1 | Rebuild per-structure cache | 106/106 pkls, ok |
| 2 | QA gate (cross-cache reproducibility) | rc=0, pairwise r = 0.995, airway real-rate 81.13% |
| 3 | Edge-drop sweep p ∈ {0.0, 0.05, 0.10, 0.15} | best p = 0.10, AUC 0.721 ± 0.160 |
| 5 | Phase 2 full training (p=0.10, label_smoothing=0.1, warmup=20) | AUC 0.729 ± 0.125 |
| 6 | Comparison report | deferred — waiting on root-cause investigation |

## Phase 2 vs Phase 1

| Metric | Phase 1 (unified) | Phase 2 (tri) | Plan target |
| --- | --- | --- | --- |
| AUC | 0.880 ± 0.093 | 0.729 ± 0.125 | ≥ Phase 1 |
| \|r(artery, mPAP)\| | — | 0.073 | ≥ 0.45 |
| \|r(airway, mPAP)\| | — | 0.042 | ≥ 0.50 |
| ARI(label) on z_fused | — | ≤ 0.004 | > 0.15 |

All six plan success criteria failed. The plan itself flagged an investigation threshold:
> If \|r(airway, mPAP)\| drops below 0.40, investigate airway segmentation quality.

That condition is triggered (0.042 ≪ 0.40).

## Files

- `src/` — pipeline sources as trained (includes the edge_dropout / warmup_cosine patches and the new `--cache_format`, `--edge_drop_p`, `--label_smoothing`, `--warmup_epochs` CLI flags)
- `automation/` — `auto_run_sprint7.py` orchestrator + the four helper shell scripts actually run on the server
- `outputs/sprint7_qa/` — QA report, diameter histogram, node-count histogram
- `outputs/sprint7_sweep_edrop/` — one cv_results.json per p-value
- `outputs/sprint7_phase2/` — cv_results.json, shared_embeddings.npz, cluster_analysis.json

## Known quirks fixed during the run

- 2026-04-19: `run()` wrapped remote commands in `bash -lc "..."`; the login shell expanded `$!` and `$(...)` before the inner shell. Fixed by `exec_command(cmd)` directly.
- 2026-04-20: `_run_qa_cache_tri.sh` word-split the space-containing cache path because `${OLD_ARG}` was unquoted. Fixed with a bash array and `"${OLD_ARG[@]}"`.
- SSH: concurrent paramiko loops (background watcher + foreground status polls) trigger sshd banner-read failures that clear after ~5 min. Run one loop at a time; use ScheduleWakeup pacing instead.
