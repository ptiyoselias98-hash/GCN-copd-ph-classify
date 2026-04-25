# Auto Review Loop — COPD-PH GCN v2 Cache Project

**Started**: 2026-04-22 23:50
**Reviewer**: codex-mcp gpt-5 high-reasoning (hard difficulty + reviewer-memory)
**Max rounds**: 4
**Stop condition**: score >= 6/10 AND verdict in {"ready", "almost"}

## Project under review

- v2 graph cache builder (kimimaro TEASAR) — `_remote_build_v2_cache.py`
- 282-case cohort with 2 sub-cohorts (197 contrast-enhanced, 85 plain-scan CT)
- Sprint 6 results on flat A+V cache → arm_a pooled AUC ~0.95 (vs v1 0.78)
- Outstanding work captured in `REPORT_v2.md`

## Round history

---

### Round 1 — 2026-04-23 10:33

**Score**: 2/10 | **Verdict**: `reject` (hard-mode)

**Top critical issues**
- **W1** (critical): Cohort/protocol confounding — all 170 PH = contrast-enhanced; many nonPH = plain-scan. AUC ~0.95 may be acquisition-protocol classification, not disease signal.
- **W2** (critical): No independent validation. Internal 5-fold CV only; no patient-level grouping check; no external/temporal test; possible multi-scan-per-patient leakage across folds.
- **W3** (critical): Graph construction not anatomically validated (no overlay audits, no TEASAR sensitivity).
- **W4** (major): Feature engineering defects (voxel-space coords, ct_density=0, Strahler-approx no cycle handling).
- **W5** (major): Ad-hoc QC rules with label-correlated exclusions (27 placeholder nonPH excluded changes balance).
- **W6** (major): Multiplicity uncontrolled; no CIs/DeLong/bootstrap on the v1→v2 AUC jump.
- **W7** (critical): Airway claims unsupportable (41/47 valid, 2 invalid, 4 errored; no airway QC; arm_b_v2 unstable).
- **W8** (major): Reproducibility package incomplete (no env lockfile, no kimimaro pin, no immutable manifest).

**Strengths**
- Mask-extraction bug correctly identified + sentinel-based fix directionally sound.
- Report transparently exposes QC fields, cache formats, limitations.
- arm_a vessel-only v2 fold AUC ~0.96 internally strong (but not yet credible as disease perf).

**Must fix before Round 2**: W1, W2, W3, W5, W6, W7, W8

**Reviewer memory (persistent)**:
> Central threat is acquisition/source confounding: all PH=contrast-enhanced, many nonPH=plain-scan → AUC ~0.95 may be protocol leakage. Airway claims are no-go. Round 2 must show protocol-balanced + patient-level + locked validation + QC + reproducibility manifests.

**Applied immediately (doc-level)**:
- W7: scope airway from main claim to caveat (REPORT §6 already does this; reinforced in §12 Limitations).
- W8: add kimimaro/env pin details to §8 Reproducibility.
- W4: lift the deferred-fix list from §6 into an explicit §12 Limitations section with reviewer-visible wording.

**Queued for user attention (requires new experiments / decisions)**:
- W1 protocol-confound controls: run arm_a restricted to PH-vs-nonPH cases with same acquisition protocol (needs protocol metadata).
- W2 patient-level split verification: audit case_id → patient_id mapping; check no patient leaks across folds.
- W3 TEASAR parameter sensitivity: 2–3 alternative settings; overlay QC for ~20 random cases.
- W5 exclusion sensitivity: re-run arm_a keeping all 27 placeholder nonPH with imputed graphs.
- W6 paired DeLong / bootstrap CIs on v1 vs v2 results.

**Next round**: wait for arm_b_triflat_v2 + arm_c_quad_v2 training (running now, ETA 2-3 h); then build round 2 context with fixes + new arm results, rerun codex with reviewer memory.

---

### Round 3 — 2026-04-23 16:40

**Score**: 4/10 | **Verdict**: `reject` (gpt-5.2 high-reasoning)

**Delta since Round 2**:
- `data/case_protocol.csv` + explicit cross-tab documents label↔protocol entanglement
- R3 cache-adjacent feature protocol decodability (proxy for GCN inputs)
- E3 HiPaS-aligned T2 confirmed (COPD→↓vein, ρ=−0.75/−0.51 across protocols); T1 direction opposite to HiPaS (PH has higher artery volume → central PA dilation signature, not distal pruning; needs skeleton length)
- R3 W8 reproducibility package (REPRODUCE.md, environment.yml, requirements-local.txt)

**Critical methodological insight from reviewer** (for Round 4):
> Protocol AUC computed across full cohort lets the model shortcut via `label → protocol` (since all 170 PH are contrast). The honest test must run **within label=0 only** (27 contrast nonPH vs 85 plain-scan nonPH). Our Round 3 protocol AUCs are therefore not isolating protocol leakage — they conflate it with label signal.

**Must fix before Round 4**: W1, W2, W3, W4, W6, W8

**Minimum to reach 8/10** (reviewer-specified):
1. Protocol-matched primary analysis (matching/weighting OR adequate n)
2. Protocol decodability on exact GCN inputs WITHIN nonPH only + CIs
3. Per-case val-prob dumps + paired DeLong on headline deltas
4. Anatomical overlay gallery + TEASAR parameter sensitivity
5. Exclusion sensitivity with placeholders included via degraded handling
6. Locked reproducibility with embedded git SHA + kimimaro pin in cache metadata

---

### Round 2 — 2026-04-23 15:45

**Score**: 3/10 | **Verdict**: `reject` (hard-mode gpt-5.2, high-reasoning)

**Delta since Round 1**:
- W1 quantified: §13 contrast-only ablation (AUC drops 0.05–0.08); §14.3 scalar-lung protocol AUC = 1.000
- W2 resolved: 282 cases = 282 unique patients (one scan each); 0 leakage by construction
- W6 partial: fold-level paired bootstrap + Wilcoxon + paired-t (case-level DeLong still missing)
- W7 scoped: airway claims moved to §11.5 / appendix

**Round 2 per-W disposition**:

| W | Severity | Addressed | Note |
|---|---|---|---|
| W1 | critical | partial | Protocol-matched eval underpowered (26 nonPH negatives); label/protocol near-perfectly entangled |
| W2 | minor | **yes** | Audit confirms patient-disjoint; external validity still absent |
| W3 | critical | **no** | TEASAR overlays + parameter sweep deferred |
| W4 | major | **no** | mm-coords / ct_density / Strahler cycles still not fixed in v2 cache |
| W5 | major | **no** | Exclusion-sensitivity not run |
| W6 | major | partial | Fold-level only; case-level DeLong missing |
| W7 | minor | partial | Scoped to appendix; airway QC still pending |
| W8 | major | **no** | No env lockfile, no kimimaro pin, no one-command rebuild |

**Must fix before Round 3**: W1, W3, W6, W5, W8, W4

**Round 3 reviewer priorities** (persisted to REVIEWER_MEMORY.md):
1. Protocol decodability from **exact GCN/cache features** (not separate lung scalars)
2. Anatomical validation of graph construction (TEASAR overlays, skeleton-to-mask coverage)
3. Case-level paired AUC inference (DeLong) requiring per-case val-prob dump
4. Reproducibility artifacts (env lockfile, kimimaro pin, one-command rebuild)

**Strengths acknowledged this round**:
- Inflated v1→v2 +0.17 AUC narrative explicitly retracted (§13.5)
- Protocol AUC = 1.000 evidence is clear and falsifiable (§14.3)
- Patient-disjoint splits backed by audit (zero leakage)

---

### Post-Round-1 fixes applied (2026-04-23 ~11:35)

**arm_b_triflat_v2 + arm_c_quad_v2 complete** (full 243-case cohort):

| Arm | AUC | pooled AUC |
|---|---|---|
| arm_b | 0.920 ± 0.030 | 0.900 |
| arm_c | 0.959 ± 0.033 | 0.947 |

**W1 protocol-confound ablation complete** (contrast-only 189-case subset):

| Arm | AUC | pooled AUC | Δ vs full |
|---|---|---|---|
| arm_b contrast-only | 0.871 ± 0.092 | 0.821 | −0.049 |
| arm_c contrast-only | 0.877 ± 0.085 | 0.862 | −0.082 |

**Honest interpretation** (now documented in `REPORT_v2.md §13`):
- Both arms drop 0.05–0.08 AUC under protocol balancing → headline §10.1 numbers were inflated by protocol cues.
- arm_c lung-feature advantage mostly vanishes (+0.04 → +0.006 AUC) on the balanced cohort — lung HU/LAA distributions are the most protocol-confounded features.
- Residual ~0.87 AUC is meaningful disease signal above chance but the honest upper bound under current methodology.
- Small-N high-variance: 26 nonPH contrast cases, 3–7 per fold. Round 2 must report DeLong / bootstrap CIs on Δ.

**Round 2 context will include**: §13 (W1 ablation), REVIEWER_MEMORY.md, and the retracted lung-feature gain claim.

---

## Round 12 (2026-04-25 11:30) — score 7.0 / verdict revise

Reviewer: codex-mcp gpt-5.5 high-reasoning. Up from R11=5.0 (+2.0).

**R12 deliverables**:
- `R12_missingness_probe.py` → within-nonPH LR(is_contrast ~ is_in_v2_cache) AUC 0.664 [0.599, 0.724], 31/32 missing nonPH are plain-scan
- `R12_fetch_advauc_logs.py` → per-(λ,seed) run.log + `outputs/r11/adv_auc_per_epoch.json` (12/12 configs, λ=0 baseline excluded by design)
- `R12_aggregate_seed_CIs.py` → pooled-prob + hierarchical bootstrap CIs; λ=10 [0.719, 0.935]
- REPORT_v2.md §22 (R11 summary) + §23 (R12 honest-impossibility framing — to be narrowed in R13)

**Reviewer regressions**:
- "Impossibility" framing too broad: applies only to corrected-GRL + nonPH-only adversary + legacy 243 + n=80. Non-GRL variants (CORAL/MMD/HSIC/IRM/propensity) untried.
- 345-cohort PH=160 vs legacy PH=170 not reconciled by case_id diff
- Held-out adversary validation curves missing (batch-mean is noisy)

**Must fix before R13**: cohort reconciliation, ≥1 non-GRL deconfounder, narrow §23.4 framing, report disease AUC alongside protocol-AUC under any control.

**Path to ≥9.5**: ingest 158 plain-scan nonPH → unified 345 cohort with QC, freeze single manifest, achieve protocol AUC ≤0.60 with upper-CI ≤0.65 while preserving disease AUC within −0.03, demonstrate robustness across multiple deconfounder families.
