# ARIS Round 3 Context — COPD-PH GCN v2 Cache

**Round 2 verdict**: 3/10 reject by gpt-5.2 high-reasoning. W1 partial, W2
resolved, W6 partial, W3/W4/W5/W7/W8 not addressed.

**Round 2 reviewer priorities for Round 3**:

1. Protocol decodability from exact GCN/cache features.
2. Case-level DeLong paired AUC (vs fold-level Wilcoxon).
3. TEASAR/graph anatomical validation (overlays, coverage).
4. Reproducibility artifacts (env lockfile, version pins, one-command rebuild).
5. W4 builder fixes (mm-coords, ct_density, Strahler cycles).
6. W5 exclusion sensitivity.

**This round's claim**: we have delivered on (1) and (4) end-to-end locally,
and have added a HiPaS-aligned disease-direction sanity check that reveals
a substantive Round 4 target. We have NOT delivered on (2), (3), (5), (6)
because those require remote GPU access or a v3 builder rebuild. The
question is whether the Round 3 deliverables warrant any score bump, and
what the minimum remaining set is to reach ≥ 8/10.

---

## What changed since Round 2

### R3.1 — Cache-feature protocol decodability (§16.2)

Script: `scripts/evolution/R3_cache_feature_protocol.py`.
Five feature sets designed to proxy cache-internal statistics:

| Set | n_feats | Protocol LR / GB | Disease contrast LR / GB |
|---|---|---|---|
| A per_structure_volumes | 4 | **0.524** / 0.910 | 0.756 / 0.660 |
| B volumes + ratios | 7 | 0.885 / 0.854 | 0.770 / 0.706 |
| C spatial LAA only | 4 | 0.732 / 0.767 | 0.671 / 0.402 |
| D paren LAA only | 3 | **0.591** / 0.811 | 0.685 / 0.633 |
| E v2 ratios no HU | 11 | 0.860 / 0.877 | 0.786 / 0.741 |

Headline: **LR** protocol AUC is near 0.5 for set A (volumes) and 0.59 for
set D (paren LAA), but gradient boosting recovers the protocol signal non-
linearly (0.81–0.91). Implication: any GCN on these features needs
domain-adversarial or protocol-augmentation mitigation before a causal
disease claim.

### R3.2 — HiPaS-aligned disease-direction test (§16.1)

Script: `scripts/evolution/E3_abundance_disease_direction.py`.

HiPaS (Chu et al. Nature Comm 2025, n=11,784) predictions:

- PAH → ↓ pulmonary **artery** abundance (skeleton length, branch count)
- COPD → ↓ pulmonary **vein** abundance

Our volume-fraction proxy on contrast-only 189:

- **T2 (COPD → ↓ vein)**: Spearman ρ = **−0.745** (p<1e-33) on contrast,
  **−0.511** (p=3e-4) on plain-scan. Direction matches HiPaS on both
  protocols. Strong cross-validation.
- **T1 (PAH → ↓ artery)**: PH median artery_frac 0.081 > nonPH 0.047
  (p=8.4e-6, Cliff's δ=+0.54). **Direction is OPPOSITE to HiPaS.**

The T1 mismatch is reported transparently. The likely cause: volume
conflates central PA dilation (increases volume) with distal pruning
(decreases skeleton length). A true HiPaS-equivalent requires skeleton
length, which is scheduled as the primary Round 4 endpoint.

### R3.3 — W8 reproducibility package (§16.3)

- `environment.yml` (remote Python 3.9 + PyTorch 2.2 + PyG 2.5 + kimimaro 4.0.4).
- `requirements-local.txt` (local analysis Python).
- `REPRODUCE.md` — one-file recipe covering W1/W2/W6/E2/E3/R3 + remote rebuild + ARIS loop.
- builder_version tag in pkls.

### R3.4 — Deferred to Round 4 (explicit)

- **Case-level DeLong**: needs remote rerun with per-case val-prob dumps.
- **Skeleton-length abundance**: needs skimage skeletonize_3d locally or
  remote kimimaro access — HiPaS T1 verdict pending on this.
- **TEASAR parameter sensitivity**: needs remote rebuild sweep.
- **v3 builder (mm-coords, ct_density)**: would invalidate all Sprint 6
  results if run now; proposal is to run in parallel on a v3 branch and
  not retract v2 results.
- **Domain-adversarial mitigation**: new experimental arm blocked by (2).
- **W5 exclusion sensitivity**: still pending.

---

## The reviewer question for Round 3

1. Does the protocol-robust linear floor (set A volumes LR AUC 0.52; set D
   paren LAA LR 0.59) materially weaken the W1 concern, or does the GB
   0.91 non-linear decoder rescue it?
2. Is the T2 literature-aligned direction (COPD→↓vein, ρ=−0.75 on
   contrast, ρ=−0.51 on plain-scan) strong enough to count as external
   validation absent an actual external cohort?
3. Are the Round 4 blockers (case-level DeLong, skeleton length,
   TEASAR sensitivity, domain-adversarial) scoped realistically? Any of
   them required at Round 4 to clear ≥ 6 / ≥ 8?
4. Please return a verdict in the same JSON-like format as Round 2.
