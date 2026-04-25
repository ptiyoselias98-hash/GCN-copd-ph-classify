# Autonomous cron prompt

This file contains the exact prompt passed to each cron fire. Kept here so
it survives across sessions and can be edited without re-running CronCreate.

## PRE-APPROVED OPERATIONS (user confirmed 2026-04-24)

The user has explicitly pre-authorised ALL of the following. Do not ask,
just do:

1. **SSH / scp to imss@10.60.147.117** — write scripts locally, scp to
   `/tmp/`, `ssh bash /tmp/X.sh`. Never use HEREDOC inside ssh quoted args.
2. **Launch training on GPU 0 AND GPU 1** — use either/both whenever
   idle (0% utilization), regardless of other users' prior jobs.
3. **git add + git commit -m "single-line" + git push origin main** —
   single-line messages, never `--force`, never `--amend`.
4. **codex MCP** with `model=gpt-5.2`, `approval-policy=never`,
   `sandbox=read-only`, `config={"model_reasoning_effort":"high"}`.
   Each round auto-fires codex review.
5. **CronDelete the loop** when `score ≥ 9.5` OR `round ≥ 40` (user raised targets 2026-04-24).
6. **Multi-hour remote jobs** allowed: kimimaro cache rebuilds,
   TEASAR sensitivity sweeps, domain-adversarial arms, cohort re-ingestion.

## BLOCKER POLICY

User chose "auto-try all plausible alternatives silently, no marker".
Do NOT write `NEEDS_USER_INPUT_*.md` files. If one approach fails, try
the next; if all fail in this round, commit what you have with an
honest-negative REPORT_v2 section and let the next cron pick up.

---

You are continuing the ARIS autonomous review loop for the COPD-PH GCN
v2 cache project. This is an UNATTENDED run. Do NOT ask the user for
confirmation on anything. The user has granted full permission.

**Safety rules** (to avoid the "Newline inside quoted argument" prompt):

- Use the `Write` tool to create any shell scripts; then `scp` them and
  `ssh imss@10.60.147.117 "bash /tmp/script.sh"` — never HEREDOC inside
  ssh arguments.
- For git commits, use `-m "single line message"` (no newlines, no HEREDOC).
- For codex, always `approval-policy=never` and `sandbox=read-only`.

**Your job this fire**:

1. `cd E:/桌面文件/图卷积-肺小血管演化规律探索 && git pull --ff-only origin main`.
2. Read `copdph-gcn-repo/review-stage/REVIEW_STATE.json`. If `target_reached`
   is true, or `history[-1].score >= 8`, or `history[-1].round >= 20` — DO
   NOTHING, delete the cron (via CronDelete if you remember the ID, else
   CronList and delete), and exit.
3. Otherwise `next_round = history[-1].round + 1`. Read
   `copdph-gcn-repo/review-stage/AUTONOMOUS_LOOP_PLAN.md` and execute the
   experiments listed for `next_round`. If some step fails, record the
   failure in REPORT_v2.md §17.N "skipped this round" and continue —
   do NOT block the whole round on one failure.
4. Commit + push with a concise 1-line message describing what this round
   added.
5. Invoke codex via `mcp__codex__codex` with:
   - `model`: `gpt-5.2`
   - `approval-policy`: `never`
   - `sandbox`: `read-only`
   - `cwd`: `E:/桌面文件/图卷积-肺小血管演化规律探索/copdph-gcn-repo`
   - `config`: `{"model_reasoning_effort":"high"}`
   - `prompt`: "You are the ARIS hostile reviewer for Round {N}. Memory in
     review-stage/REVIEWER_MEMORY.md. Read REPORT_v2.md §17 (Round 4) and
     whatever new Round-{N} section exists. Score 1-10, return the same
     JSON-like verdict format as prior rounds."
6. Parse the verdict from codex output. Update `REVIEW_STATE.json` history
   array, update `REVIEWER_MEMORY.md` with the new reviewer_notes_for_memory,
   update `AUTO_REVIEW.md` with the Round section.
7. Commit + push the state update.
8. If `score >= 8`:
   - Write `review-stage/TARGET_REACHED.md` with the round number + score.
   - CronList → find the ARIS loop job → CronDelete.
   - Commit + push.
   - Stop.
9. Else if `round >= 20`:
   - Write `review-stage/FINAL_REPORT.md` summarizing all 20 rounds.
   - CronDelete.
   - Commit + push.
   - Stop.
10. Else: just exit. The cron will fire again later.

## README maintenance — MANDATORY every round

After step 6 (update REVIEW_STATE / AUTO_REVIEW / REVIEWER_MEMORY) and
BEFORE step 7 (stop check), you MUST update the top-level `../README.md`
(the one GitHub renders on the repo landing page, at the repo root —
**not** the `copdph-gcn-repo/README.md` if one exists) with the new round's
results. Required content per round:

- Append a new `## ARIS Round {N} — ...` section near the bottom.
- Embed any new figures under `copdph-gcn-repo/outputs/figures/` or
  `copdph-gcn-repo/outputs/r{N}/*.png`.
- Summarise the 1-2 most important numerical results (AUC, Δ, CI, p)
  with a short interpretation paragraph.
- Link the markdown artefacts (`outputs/r{N}/*.md`).
- Update the "Round history" score table (R1 2/10, R2 3/10, … R{N} <score>/10).

If `scripts/figures/R7_make_figures.py` or a similar generator needs to
regenerate figures from the new round's JSONs, run it before the README
edit so the embedded images are current.

Failing to update the README counts as an incomplete round — redo it
before moving on.

**Helper**: `scripts/figures/update_readme_round.py` automates the above.
Usage:

```
python scripts/figures/update_readme_round.py {N} {score} \
  --artifacts outputs/r{N}/X.md outputs/r{N}/Y.png \
  --summary "1-sentence headline for this round"
```

It regenerates `fig1_aris_score_progression.png` from REVIEW_STATE.json,
refreshes the round-history table, and upserts the per-round section.

**Model fallback chain for codex at-capacity**: gpt-5.2 (high) → gpt-5.2
(default) → gpt-5.2-codex. If all fail, write a note and skip to next fire.

**Experiment sub-plans**: see `copdph-gcn-repo/review-stage/AUTONOMOUS_LOOP_PLAN.md`
for Round 5–12 specific actions. Round 13+ is open-ended: respond to the
last codex `must_fix_before_next_round` list.

**No user interaction**. If anything genuinely requires user input, write it
to `review-stage/NEEDS_USER_INPUT_<timestamp>.md` and continue to next round.

**Current state at cron setup**: Round 4 complete, score 5/10. GPU 1
running arm_b + arm_c contrast-only with prob dumps (started 18:30).
Round 5 will consume those dumps for paired DeLong. See REVIEW_STATE.json
for the latest snapshot.
