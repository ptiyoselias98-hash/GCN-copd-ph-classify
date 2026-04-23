# COPD → COPD-PH Evolution Clustering — Experimental Design

**Scientific question (user-facing)**: How do pulmonary vascular imaging phenotypes
evolve during COPD → COPD-PH progression, and what auxiliary role do lung-parenchyma
and airway phenotypes play in this evolution?

**Design-driving constraint (ARIS Round 2 reviewer memory)**: the 282-case cohort
has near-perfect label/protocol entanglement (170 PH contrast / 85 nonPH plain-scan /
27 nonPH contrast / 0 PH plain-scan). Any evolution analysis must survive
protocol balancing or else it is describing contrast-vs-plain-scan, not disease.

---

## E1 — Vessel phenotype cluster within contrast-only subset (PRIMARY)

**Sample**: 189 contrast-enhanced cases (163 PH + 26 nonPH) from
`data/splits_contrast_only/`.

**Features** (from `cache_v2_tri_flat/<case>.pkl` graph-level summaries):

- artery: `n_nodes`, `n_edges`, `mean_diameter_mm`, `std_diameter_mm`,
  `mean_tortuosity`, `mean_length_mm`, `max_strahler`, `edge_len_p90_mm`,
  `branching_ratio = n_edges / n_nodes`.
- vein: same 9 features.
- 18 features per case.

**Pipeline**:
1. Z-score within contrast-only cohort.
2. UMAP (n_components=2, n_neighbors=15, min_dist=0.1) for visualization.
3. Cluster selection: GMM and KMeans at k ∈ {2, 3, 4, 5}; pick k by BIC
   (GMM) and silhouette (KMeans); report both.
4. Per-cluster:
   - Size, PH proportion, 95% CI on proportion (Wilson interval).
   - Top-3 features by |z-score| of cluster centroid.
   - Pairwise separability vs other clusters (Mann-Whitney U per feature,
     Holm-corrected).
5. **Identify the "PH-enriched cluster"** and report the vessel
   signature (which topology metrics are shifted).

**Hypothesis to test**: PH cluster has smaller mean_diameter + higher
tortuosity + fewer distal branches (Strahler + edge_len_p90). If
confirmed, this is the *vessel remodelling signature*.

**Falsification test**: if PH cases distribute roughly uniformly across
clusters (max PH proportion < 2× overall PH rate 163/189 ≈ 86%), then
there is no vessel-topology signature within the contrast subset and
any vessel claim is moot.

**Output**: `outputs/evolution/E1_vessel_cluster_contrast.json`,
`outputs/evolution/E1_vessel_umap.png`.

---

## E2 — Parenchyma phenotype cluster across full 282 (REGISTRATION w/ protocol)

**Sample**: 282 cases (using protocol-robust v2 features).

**Features** (from `outputs/lung_features_v2.csv` — parenchyma-only):

- `paren_LAA_950_frac`, `paren_LAA_910_frac`, `paren_LAA_856_frac`
- `paren_mean_HU`, `paren_std_HU`
- `apical_LAA_950_frac`, `middle_LAA_950_frac`, `basal_LAA_950_frac`
- `apical_basal_LAA950_gradient`

9 features per case.

**Pipeline**:
1. Z-score globally.
2. UMAP + GMM as above.
3. **Dual reporting**:
   - Pooled: PH proportion per cluster on the 282-cohort.
   - Stratified: PH proportion per cluster within contrast-only (189) and
     within plain-scan (85, all nonPH by construction → only informative
     for the parenchyma severity distribution).
4. **Protocol check**: for each cluster, report protocol mix. A
   protocol-robust feature set should produce clusters with protocol
   proportions ≈ the overall 197/85 ratio.

**Hypothesis**: parenchyma emphysema severity is *modifier*, not *predictor*,
of PH — i.e. within each severity bucket, vessel topology still separates
PH from nonPH. Report as effect-modification via a 2×2 table (severe vs
mild emphysema × PH vs nonPH) restricted to the contrast cohort.

**Falsification**: if clusters align perfectly with protocol
(>95% single-protocol dominance), the parenchyma signal is not
protocol-robust even after vessel/airway subtraction, and the w2 lung
feature story is dead.

**Output**: `outputs/evolution/E2_paren_cluster.json`,
`outputs/evolution/E2_paren_umap.png`.

---

## E3 — Airway phenotype cluster (APPENDIX)

Blocked until airway QC in W7 is complete. Script written but not run in
main results.

---

## E4 — Joint evolution trajectory (SECONDARY)

**Sample**: 189 contrast-only cases.

**Pipeline**:
1. Compute centroid of PH cluster in E1's feature space (Mahalanobis).
2. For each contrast nonPH case, compute distance to PH centroid.
3. Rank nonPH by proximity → "high-risk COPD" list (top-5 / top-10).
4. For each high-risk case, report its parenchyma cluster assignment
   from E2 to probe: "do vessel-trajectory-advanced nonPH patients show
   parenchyma signatures closer to emphysema PH?"

**Output**: `outputs/evolution/E4_high_risk_nonph.md` — a table of
nonPH cases ranked by vessel proximity to PH centroid, with their
parenchyma cluster, mean_diameter, tortuosity, apical LAA gradient.

**Clinical translation hypothesis**: high-risk COPD-noPH cases identified
by vessel topology + parenchyma severity deserve follow-up imaging or
right-heart catheterization — an actionable imaging biomarker.

---

## E5 — Protocol-matched evolution sanity check (CONFIRMATORY)

**Sample**: 26 contrast nonPH + 26 matched contrast PH (propensity-score
matched on age, sex, smoking pack-years where xlsx has them).

**Pipeline**:
1. Propensity matching using scikit-learn `MatchIt`-equivalent.
2. Repeat E1 on this balanced 52-case subset.
3. If PH-enriched cluster still emerges with a coherent vessel signature,
   the disease effect is real.
4. If not, the contrast-only n=189 result was still partly a mere-imbalance
   artefact.

**Why this matters for Round 3**: it turns "weak confound control" into
a design-matched sub-cohort with n=52 — still small, but much less
biased. Reviewer needs to see this before a disease claim is accepted.

**Output**: `outputs/evolution/E5_matched_cluster.json`,
`outputs/evolution/E5_matched_vs_unmatched_deltas.md`.

---

## Execution order

1. **NOW (local)**: E2 parenchyma cluster — depends only on
   `lung_features_v2.csv` currently extracting.
2. **Next (server)**: E1 vessel cluster — requires reading 189 pkl files
   from `cache_v2_tri_flat/` on the remote box; can be run via a small
   remote script.
3. **Then (local)**: E4 trajectory from E1 output.
4. **Then (local)**: E5 matched sub-cohort (requires xlsx for covariates).
5. **Deferred**: E3 after airway QC lands.

All scripts will live in `scripts/evolution/` with each script producing
a single markdown + JSON artefact for the paper's results section.
