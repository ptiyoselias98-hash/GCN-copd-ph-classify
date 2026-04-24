# R10 — GRL λ sweep results

Protocol AUC WITHIN nonPH + Disease AUC WITHIN contrast (case-level OOF + bootstrap CI)

| Setting | n nonPH | Protocol AUC (CI) | Disease AUC (CI) |
|---|---|---|---|
| baseline (λ=0) | 80 | 0.816 [0.720, 0.903] | 0.752 [0.625, 0.864] |
| λ=0.5 | 80 | 0.858 [0.775, 0.931] | 0.677 [0.553, 0.800] |
| λ=1.0 | 80 | 0.816 [0.715, 0.908] | 0.684 [0.546, 0.815] |
| λ=2.0 | 80 | 0.823 [0.732, 0.908] | 0.636 [0.499, 0.771] |
| λ=5.0 | 80 | 0.816 [0.729, 0.900] | 0.666 [0.531, 0.798] |

**Best operating point** (lowest protocol AUC with disease ≥ 0.75 if possible): `baseline (λ=0)` — protocol 0.816, disease 0.752
- Target was protocol ≤ 0.60. **❌ NOT MET** — try stronger λ, warm-up schedule, or different architecture.