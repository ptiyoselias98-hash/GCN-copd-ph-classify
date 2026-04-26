# R23 Final Claims Table — manuscript scope decision

_Closes R22 codex must-fix #3 (final claims table to prevent scope creep during manuscript drafting)._

Each row is a candidate paper claim. `Status` is one of:
- **CITE**: keep, formally cite in manuscript
- **DEMOTE**: keep but as supplementary/sanity-check; not a headline
- **RETIRE**: removed from manuscript scope (R22 repositioning decision)

| # | Claim | Cohort | Pipeline | Evidence | Limitations | Status |
|---|---|---|---|---|---|---|
| 1 | Within-contrast PH<nonPH artery edge-length p25 | n=190 (163 PH + 27 nonPH) | both HiPaS-style legacy AND Simple_AV_seg unified | d = -0.298 (unified, p=0.013); d ≈ -1.25 (legacy R17.A) | n_nonPH=27 | **CITE** |
| 2 | Within-contrast PH<nonPH artery edge-length p50 | n=190 | both | d = -0.473, MWU p = 0.001 (Holm-sig in unified pipeline 4-feature panel) | n_nonPH=27 | **CITE** |
| 3 | Within-contrast PH<nonPH artery tortuosity p10 | n=190 | both | d = -0.370, MWU p = 0.032 (unified) | n_nonPH=27 | **CITE** |
| 4 | Within-contrast PH<nonPH vein edge-length p25 | n=190 | both | d = -0.712, MWU p = 0.12 (unified; n_nonPH limits power); d ≈ -1.19 legacy | n_nonPH=27; legacy magnitude pipeline-specific | **CITE** with caveat |
| 5 | TDA: PH<nonPH `vein_persH1_total` | n=190 | legacy HiPaS-style (R17.5) | d = -1.214, Holm p = 2.98e-6 in 18-panel; bootstrap-1000 sign-stab 100% | LOO d range [-1.335, -1.127]; not yet replicated in unified pipeline | **CITE** with replication note |
| 6 | Within-contrast 3-modality endotype panel | n=197 | legacy HiPaS-style (R18.F) | 26 Holm-sig features across vessel + parenchyma + TDA | one pipeline only; unified replication pending | **CITE** with caveat |
| 7 | parenchyma_std_HU PH endotype | n=197 | legacy R16.B + R18.E | d = +1.10 (raw + year-residualized) | one pipeline only | **CITE** |
| 8 | Cross-sectional 5-stage mPAP severity ordering | n=261 (legacy 282 minus excluded) | legacy HiPaS-style (R18.B) | Spearman ρ = -0.767 for artery_len_p25 | LEGACY-PIPELINE MAGNITUDE — pipeline-specific | **DEMOTE** (cite only as legacy-pipeline result, not headline) |
| 9 | Cross-sectional 5-stage mPAP ordering (unified) | n=102 mPAP-resolved | Simple_AV_seg unified (R20.H) | ρ = -0.211 for artery_len_p25 | direction preserved, magnitude reduced | **DEMOTE** to caveat for #8 |
| 10 | Within-contrast disease classifier AUC | n=190 | unified Simple_AV_seg (R21.D / R22.A) | 5-seed × 5-fold mean = 0.710; per-fold range [0.438, 0.879] | n_nonPH=27 small; per-fold variance real, not artifact; **bootstrap-500 CI [0.729, 0.948] is over-optimistic due to resample duplication of 27 nonPH cases — DO NOT cite as the AUC's true 95% CI**; report only mean + fold range + n_nonPH caveat | **DEMOTE** to sanity-check |
| 11 | Full-cohort enlarged 290-case PH-vs-nonPH AUC | n=290 | unified | 0.886 ± 0.006 | mostly protocol decoding (within-nonPH protocol-AUC = 0.912) | **RETIRE** |
| 12 | Cross-pipeline single ρ magnitude estimate | unified-301 | both | legacy ρ=-0.767 vs unified ρ=-0.211 | pipeline-specific | **RETIRE** as single-magnitude claim; cite per pipeline |
| 13 | DDPM label-free PH detector | n=247 (163 PH + 84 nonPH) | unified | AUC=0.129 inverted (0.871 oriented) but in WRONG direction; protocol-shift artifact | trained on plain-scan, tested on CTPA — out-of-distribution | **RETIRE** as detector; **CITE** as honest-negative case study |
| 14 | GCN-embedding-level enlarged-cohort deconfounding | full | GCN training | feature-level CORAL fails (R21.D / R22.A) | requires per-structure cache adapter + training-time GRL/MMD; out of scope for this paper | **RETIRE** |
| 15 | R17 numerical-artifact features | 282 | legacy | 8 features with n_terminals=0 / lap_eig0~1e-18 / near-zero SD | excluded from biological interpretation | **CITE** as audit; not as biological evidence |

## Summary

- **Cited as headline biology**: claims 1–7 (within-contrast endotype panel, multi-modality)
- **Demoted to supplementary**: claims 8 + 9 (legacy ρ + unified ρ as paired pipeline-specific estimates), 10 (within-contrast classifier as sanity check)
- **Retired**: 11, 12, 14 (out-of-scope), 13 retired-as-detector but kept as honest-negative case study

## Key sentences for the manuscript abstract / discussion (R22+ scope)

> "We identify a reproducible cross-sectional vascular endotype within
> contrast-enhanced CT (n=190): COPD-PH cases show artery and vein
> edge-length percentile downshift, reduced low-percentile tortuosity, and
> reduced vein H1-loop topology persistence — direction preserved across
> two independent segmentation pipelines (HiPaS-style legacy and unified
> Simple_AV_seg). The magnitude of severity-correlation is pipeline-specific
> and is reported per pipeline. Cross-protocol classifier claims and
> label-free anomaly-based PH detection are not supported and were
> retired during peer-review-equivalent rounds."

## What still needs final manuscript work (post-R23)

- README.md and FINAL_FINDINGS.md headline language must match this table (R22 added headline correction; R23 closes consistency check)
- AUTO_REVIEW.md should reference this table when arbitrating future claims
