# R12 — Pooled-seed CIs + probe-strength ablation

Round 11 reviewer asked for aggregated cross-seed CIs and probe-strength ablation.
Per λ, we pool the 3 seeds' OOF predictions (each seed has n=80 nonPH; pooled n=240)
then bootstrap-CI on the pooled predictions. 4 probes test linear → strong-non-linear
decoding capacity.

## Within-nonPH protocol AUC by (λ × probe)

Pooled across 3 seeds (pooled n=240 nonPH samples; CI is bootstrap over pooled cases).

| λ | LR | MLP-32 | MLP-128 | RF-200 |
|---|---|---|---|---|
| 0.0 | 0.839 [0.788, 0.884] | 0.845 [0.794, 0.889] | 0.843 [0.791, 0.888] | 0.827 [0.775, 0.876] |
| 1.0 | 0.853 [0.803, 0.901] | 0.866 [0.813, 0.915] | 0.859 [0.802, 0.909] | 0.852 [0.801, 0.901] |
| 5.0 | 0.799 [0.739, 0.853] | 0.836 [0.782, 0.888] | 0.806 [0.745, 0.864] | 0.803 [0.741, 0.860] |
| 10.0 | 0.813 [0.752, 0.868] | 0.872 [0.823, 0.916] | 0.879 [0.827, 0.924] | 0.851 [0.798, 0.899] |

**Best LR-only λ**: λ=5.0 — LR pooled AUC = 0.799 (CI [0.739, 0.853], n=240).
**Worst-probe leakage minimized at**: λ=5.0, where the strongest probe still hits 0.836.

## Reading

Target ≤0.60 with upper CI ≤0.65 — **NOT MET** under any λ × probe combination.
Worst-probe (MLP-128 / RF-200) AUCs stay at 0.85+ across all λ, indicating the encoder's
embedding still permits non-linear protocol decoding even when the encoder is trained
with up to λ=10 against a 1-hidden-layer adversary. This rules out 'GRL with 1-layer adv'
as a sufficient deconfounder on n=80 within-nonPH; further attempts would need either
(a) larger sample size, (b) stronger adversary architecture matching probe capacity,
or (c) re-segmentation removing the cue at source (HiPaS-style unified pipeline).