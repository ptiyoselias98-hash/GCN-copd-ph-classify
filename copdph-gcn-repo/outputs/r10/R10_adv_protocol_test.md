# R10 — Adversarial-debiased embedding protocol test

Compares protocol decodability WITHIN nonPH on (a) baseline arm_a embeddings
(from R9 full-cohort training) vs (b) adversarially-debiased arm_a
embeddings (this round, GRL λ=0.5 on is_contrast).

Case-level OOF predictions + bootstrap CI over cases (addressing Round-9
reviewer's stats-hygiene flag on fold-level bootstrap).

| Setting | n nonPH | Protocol AUC (case-boot CI) | Disease AUC (contrast) (case-boot CI) |
|---|---|---|---|
| **baseline** | 80 | 0.816 [0.720, 0.903] | 0.752 [0.625, 0.864] |
| **adv_lambda_0.5** | 80 | 0.858 [0.775, 0.931] | 0.677 [0.553, 0.800] |

**Δ (adv − baseline)**: protocol AUC = **+0.043**, disease AUC = **-0.075**

**Reviewer target**: within-nonPH protocol AUC ≤ 0.60 (❌ NOT MET)

## Reading

- Adversarial debiasing at λ=0.5 moved protocol AUC by +0.043
  but did not yet reach the 0.60 target. Next: sweep λ ∈ {1.0, 2.0, 5.0} and
  potentially warm-up schedule; re-check disease AUC preservation.