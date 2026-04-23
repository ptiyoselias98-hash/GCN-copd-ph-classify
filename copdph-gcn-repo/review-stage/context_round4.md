# ARIS Round 4 Context — COPD-PH GCN v2 Cache

**Round history**: R1=2/10, R2=3/10, R3=4/10 (all reject). Target ≥ 8/10.

**Round 3 reviewer memory** (central new insight):

> Protocol AUC across the full 282 cohort conflates label↔protocol coupling
> (all 170 PH cases are contrast). The honest test must run **within
> label=0 only** (27 contrast nonPH vs 85 plain-scan nonPH). Our Round 3
> protocol AUCs were not isolating protocol leakage — they were
> conflating it with label signal.

**Round 3 minimum-to-reach-8 list** (reviewer-specified):

1. Protocol-matched primary analysis (matching/weighting OR adequate contrast nonPH)
2. Protocol decodability on exact GCN inputs WITHIN nonPH only + CIs
3. Per-case out-of-fold prob dumps + paired DeLong on headline deltas
4. Anatomical overlay gallery + TEASAR parameter sensitivity sweep
5. Exclusion sensitivity with placeholders retained via degraded handling
6. Locked reproducibility with embedded git SHA + kimimaro pin in cache metadata

## What Round 4 delivers

### R4.1 — within-nonPH protocol decoder (CORE W1 CORRECTION)

`scripts/evolution/R4_within_nonph_protocol.py`. 5-fold stratified CV with
`class_weight=balanced` LR, bootstrap CI (2000) on mean AUC.

| Feature set | n | LR AUC (95% CI) | GB AUC (95% CI) | R3 full-cohort LR | Δ |
|---|---|---|---|---|---|
| v1_whole_lung_HU | 110 | **0.765 [0.697, 0.833]** | 0.757 [0.666, 0.837] | 1.000 | **−0.24** |
| v2_parenchyma_only | 93 | 0.794 [0.705, 0.886] | 0.731 [0.652, 0.825] | 0.857 | −0.06 |
| v2_paren_LAA_only | 93 | 0.715 [0.646, 0.789] | 0.673 [0.566, 0.762] | 0.591 | +0.12 |
| v2_spatial_paren | 93 | 0.669 [0.543, 0.795] | 0.652 [0.548, 0.748] | 0.732 | −0.06 |
| **v2_per_structure_volumes** | 110 | **0.529 [0.429, 0.631]** | 0.702 [0.615, 0.771] | 0.524 | ≈0 |
| v2_vessel_ratios | 85 | 0.674 [0.542, 0.805] | 0.632 [0.441, 0.800] | 0.885 | −0.21 |
| v2_combined_no_HU | 73 | 0.731 [0.653, 0.810] | 0.664 [0.590, 0.737] | 0.860 | −0.13 |

**Core findings**:
- v1's "perfect protocol decoder" (AUC 1.000) was ~75% **label-leakage**,
  not protocol-leakage. Real protocol decodability on v1 within-nonPH is 0.77.
- **v2 per_structure_volumes** (artery/vein/airway/vessel_airway_over_lung):
  LR protocol AUC **0.529**, 95% CI **[0.429, 0.631]** — CI straddles 0.5
  random chance. First feature set that clears the honest W1 linear bar.
  GB finds a 0.70 non-linear decoder, so a domain-adversarial step is
  still warranted for a neural model.

### R4.3 — overlay gallery (anatomical QC)

`outputs/evolution/R4_overlay_gallery.png`. 10 cases (5 PH + 5 nonPH,
balanced across protocols) × 3 panels each (axial vessels + skeleton +
coronal MIP) using `skimage.morphology.skeletonize_3d`. First-pass
anatomical inspection enabling a reviewer to visually confirm topology.

### R4.4 — exclusion sensitivity

`scripts/evolution/R4_exclusion_sensitivity.py`. A vs B cohort (placeholders
excluded vs retained with degraded features).

| Cohort | n | disease LR full | disease LR contrast |
|---|---|---|---|
| A excluded paren_only | 231 | 0.862 | 0.858 |
| B included paren_only | 252 | 0.870 | 0.860 |
| A excluded paren+spatial | 231 | 0.871 | 0.851 |
| B included paren+spatial | 252 | 0.879 | 0.855 |

**Max |Δ disease contrast LR| = 0.004** ≪ bootstrap CI half-width 0.05.
Disease claim robust to exclusion rule.

### R4.5 — reproducibility hardening

- `requirements-local.lock.txt` (pip freeze on local analysis Python).
- `scripts/cache_provenance.py` — extracts `builder_version`, `git_sha`,
  `kimimaro_version` from cache pkls (placeholders for future remote rebuild).
- `REPRODUCE.md` updated; `environment.yml` declares kimimaro 4.0.4 pending
  remote `pip show` verification.

### R4.2 — skeleton-length HiPaS test (PENDING / IN FLIGHT)

`scripts/evolution/R4_skeleton_length.py` computing `skimage.skeletonize_3d`
skeleton length (mm per L lung) for artery/vein/airway on all 282 cases.
Script auto-re-runs HiPaS T1 (PAH→↓artery) and T2 (COPD→↓vein) with the
proper metric on completion. Results in
`outputs/evolution/R4_skeleton_directions.md` (may not be in the repo yet
if the job is still running — check mtime).

## Still NOT in Round 4 (honest)

- **Case-level DeLong** — needs server rerun with per-case val-prob dump;
  local fold-level Wilcoxon from R2 §14.2 is still the best we have.
- **TEASAR parameter sensitivity sweep** — not run; overlay gallery is
  a point-in-time check, not a sweep.
- **Domain-adversarial GCN** — new arm not trained; R4.1 shows the linear
  floor, but an actual arm_b/c retraining with gradient reversal is still
  server work.
- **Locked kimimaro version string** — placeholder still 4.0.4 pending SSH.
- **Blinded airway QC** — unchanged.

## The reviewer question for Round 4

1. Does R4.1's within-nonPH linear decoder correction (v2 per_structure_volumes
   AUC 0.529 CI straddles chance) sufficiently weaken the W1 concern for
   a top-venue submission, or does GB's non-linear 0.70 keep it blocked?
2. Is the exclusion-sensitivity (|Δ| = 0.004) adequate for W5?
3. Does the overlay gallery PNG address W3 at a Nature-Medicine level, or
   is a blinded radiologist review non-negotiable?
4. Is the reproducibility pack (lockfile + provenance scripts + REPRODUCE.md)
   sufficient for W8 given the remote-kimimaro-pin still pending?
5. How close are we to 8/10 and what is the *specific* remaining gap?

Please score 1-10 and return the same JSON-like verdict format as prior rounds.
