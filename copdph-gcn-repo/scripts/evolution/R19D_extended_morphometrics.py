"""R19.D — Extend R17 per-structure morphometrics to full 360 cohort.

Reads BOTH cache_tri_v2/ (legacy 282) AND cache_tri_v2_new100/ (new 100,
R19.C build), extracts per-structure topology fingerprints, writes the
extended CSV.

Then runs the R18.B mPAP 5-stage Spearman + Jonckheere on the enlarged
n≈350 cohort to refine evolution trends.

Run on remote 24-core CPU pool.
"""
from __future__ import annotations
import csv, json, pickle
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

# Two cache directories: legacy 282 + new 100
CACHE_LEGACY = Path.cwd() / "tri_structure" / "cache_tri_v2"
CACHE_NEW100 = Path.cwd() / "cache_tri_v2_new100"
OUT = Path.cwd() / "outputs" / "r19"
OUT.mkdir(parents=True, exist_ok=True)


def degree(ei, n):
    if ei.size == 0: return np.zeros(n, int)
    deg = np.zeros(n, int)
    np.add.at(deg, ei[0], 1); np.add.at(deg, ei[1], 1)
    return deg


def safe_skew_kurt(x):
    if x.size < 4 or x.std(ddof=0) == 0: return 0.0, 0.0
    z = (x - x.mean()) / x.std(ddof=0)
    return float((z**3).mean()), float((z**4).mean() - 3.0)


def graph_morph(g, name):
    rec = {f"{name}_n_nodes": 0, f"{name}_n_edges": 0}
    if g is None: return rec
    x = np.asarray(getattr(g, "x", None) if hasattr(g, "x") else g.get("x"))
    ei = np.asarray(g.edge_index if hasattr(g, "edge_index") else g.get("edge_index"))
    ea = np.asarray(g.edge_attr) if hasattr(g, "edge_attr") and g.edge_attr is not None else None
    if ei.ndim != 2 or ei.shape[0] != 2:
        ei = ei.T if ei.ndim == 2 and ei.shape[1] == 2 else np.zeros((2, 0), int)
    n = int(x.shape[0]) if (x is not None and x.ndim == 2) else 0
    rec[f"{name}_n_nodes"] = n
    rec[f"{name}_n_edges"] = int(ei.shape[1])
    if n == 0: return rec
    deg = degree(ei, n)
    rec[f"{name}_n_branches"] = int((deg >= 3).sum())
    rec[f"{name}_n_terminals"] = int((deg == 1).sum())
    rec[f"{name}_branch_per_node"] = float((deg >= 3).mean())
    rec[f"{name}_term_per_node"] = float((deg == 1).mean())
    rec[f"{name}_mean_degree"] = float(deg.mean())
    rec[f"{name}_max_degree"] = int(deg.max())
    if n > 1:
        rec[f"{name}_tortuosity_proxy"] = float(ei.shape[1] / max(n - 1, 1))
    if ea is not None and ea.ndim == 2 and ea.shape[0] > 0:
        unique_ea = ea[::2] if ea.shape[0] >= 2 else ea
        for col_idx, col_name in [(0, "diam"), (1, "len"), (2, "tort")]:
            if col_idx < unique_ea.shape[1]:
                vals = unique_ea[:, col_idx].astype("float32")
                vals = vals[np.isfinite(vals) & (vals > 0)]
                if vals.size >= 3:
                    for q, qname in [(10, "p10"), (25, "p25"), (50, "p50"),
                                       (75, "p75"), (90, "p90")]:
                        rec[f"{name}_{col_name}_{qname}"] = float(np.percentile(vals, q))
                    rec[f"{name}_{col_name}_mean"] = float(vals.mean())
                    rec[f"{name}_{col_name}_sd"] = float(vals.std(ddof=0))
                    sk, kt = safe_skew_kurt(vals)
                    rec[f"{name}_{col_name}_skew"] = sk
                    rec[f"{name}_{col_name}_kurt"] = kt
        if unique_ea.shape[1] >= 2:
            rec[f"{name}_total_len_mm"] = float(unique_ea[:, 1].sum())
            r = (unique_ea[:, 0] / 2.0).clip(0, 50)
            rec[f"{name}_total_vol_proxy_mm3"] = float((np.pi * r * r * unique_ea[:, 1]).sum())
    return rec


def process_one(p_str):
    p = Path(p_str)
    case_id = p.stem.replace("_tri", "")
    try:
        with open(p, "rb") as f: d = pickle.load(f)
    except Exception as e:
        return {"case_id": case_id, "error": str(e)}
    rec = {"case_id": case_id, "label": int(d.get("label", -1))}
    rec["source_cache"] = "legacy" if "cache_tri_v2_new100" not in p_str else "new100"
    for s in ("artery", "vein", "airway"):
        rec.update(graph_morph(d.get(s), s))
    return rec


def main():
    pkls = []
    for cache, source in [(CACHE_LEGACY, "legacy"), (CACHE_NEW100, "new100")]:
        if cache.exists():
            ps = sorted(cache.glob("*_tri.pkl"))
            print(f"  {source}: {len(ps)} pkls in {cache}")
            pkls.extend(ps)
    print(f"[start] {len(pkls)} total pkls")
    if not pkls:
        raise SystemExit("no pkls")
    rows = []
    with ProcessPoolExecutor(max_workers=24) as ex:
        futures = {ex.submit(process_one, str(p)): p for p in pkls}
        for i, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if i % 50 == 0: print(f"  ...{i}/{len(pkls)}")
    keys = sorted({k for r in rows for k in r.keys()})
    out_csv = OUT / "per_structure_morphometrics_extended.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "label", "source_cache"] +
                            [k for k in keys if k not in ("case_id", "label", "source_cache")])
        w.writeheader()
        w.writerows(rows)
    print(f"[done] {len(rows)} rows × {len(keys)-3} feature cols → {out_csv}")
    legacy = sum(1 for r in rows if r.get("source_cache") == "legacy")
    new100 = sum(1 for r in rows if r.get("source_cache") == "new100")
    print(f"  breakdown: legacy={legacy} new100={new100}")


if __name__ == "__main__":
    main()
