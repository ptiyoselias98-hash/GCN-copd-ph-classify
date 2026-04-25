# R20.B — DDPM anomaly evaluation (merged PH + nonPH)

Diffusion model trained on plain-scan nonPH (R19 new100, 30 epochs).
Inference: per-case mean MSE-noise-prediction NLL on 4-8 random 32³ lung patches at t=T/2.

**Cohort**: n=247 (PH=163, nonPH=84)

## Anomaly AUC

- AUC (PH > nonPH direction) = **0.129** [95% boot CI 0.087, 0.178]
- AUC inverted (nonPH > PH) = **0.871**
- PH mean NLL = 0.0066, nonPH mean NLL = 0.0087
- PH median NLL = 0.0066, nonPH median NLL = 0.0086
- Δ (PH−nonPH) = -0.0020
- MWU one-sided p (PH > nonPH) = 1.0

## Honest interpretation

DDPM was trained on **plain-scan** nonPH (R19 new100, n=100).
Inference here is on **legacy CTPA** cohort (contrast protocol).
Both PH and nonPH cases in inference are **out-of-distribution**
relative to the plain-scan training set — different protocol,
different image statistics. Whichever class happens to be closer
to plain-scan in pixel statistics dominates the AUC.

If AUC > 0.6 in EITHER direction with no obvious protocol
explanation, the result may reflect biology. Otherwise, the
result is dominated by **protocol shift, not disease**, and
this evaluation should not be cited as evidence of label-free
PH detection. Closing R18 must-fix #1 with this honest negative.
