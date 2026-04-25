# R12.2 — Missingness-only protocol probe

Tests whether the cache-missingness pattern (39 dropped cases from
`cache_v2_tri_flat`) leaks protocol information by itself. If yes,
any future degraded-graph or principled-missingness approach inherits
the leak.

**Cohort**: total 282, in cache 243, missing 39.
Within-nonPH: n=112.

## Within-nonPH LR(protocol ~ is_in_v2_cache)

- AUC = 0.664, bootstrap mean 0.664, 95% CI [0.599, 0.724]

## Within-nonPH cross-tab (rows = is_in_v2_cache, cols = is_contrast)

```
is_contrast      0   1
is_in_v2_cache        
0               31   1
1               54  26
```

## Full-cohort cross-tab

```
is_contrast      0    1
is_in_v2_cache         
0               31    8
1               54  189
```

**Verdict**: MISSINGNESS LEAKS PROTOCOL — principled-missingness rescue is unsafe
