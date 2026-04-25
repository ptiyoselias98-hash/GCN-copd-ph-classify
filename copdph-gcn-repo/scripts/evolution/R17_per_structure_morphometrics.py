"""R17 — Per-structure topological fingerprints (artery / vein / airway).

Reframed per user 2026-04-25: GCN mean-pool compresses topology; need
distribution-shape signatures, not just scalars. Per-structure (NOT merged)
morphometrics + topology distributions:

  - Basic graph-level: n_nodes, n_edges, n_branches (deg≥3), n_terminals (deg=1),
    branch_per_node, term_per_node, mean/max degree
  - Path geometry: longest_path_hops, eccentricity, n_components
  - Edge-attr distributions: edge_len/diameter/tortuosity p10/p25/p50/p75/p90
    + skewness + kurtosis (shape, not just mean)
  - Strahler-like depth from leaves: distribution percentiles
  - Spectral signature: graph-Laplacian top-5 eigenvalues (algebraic connectivity)
  - Total physical: total_skel_length_mm, total_volume_proxy

Run on remote 24-core CPU pool over 282 legacy cache_tri_v2 pkls.
After v2 cache rebuild on new100 (R17 sub-task), extend to 360 cohort.

Output: outputs/r17/per_structure_morphometrics.csv
"""
from __future__ import annotations
import csv, json, pickle
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

CACHE = Path.cwd() / "tri_structure" / "cache_tri_v2"
OUT = Path.cwd() / "outputs" / "r17"
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


def laplacian_eigs(ei, n, k=5):
    """Top-k smallest eigenvalues of unnormalized graph Laplacian (spectral fingerprint)."""
    if n < 3 or ei.size == 0: return [0.0]*k
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        A = csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n))
        A = A + A.T
        A.data = np.minimum(A.data, 1)
        deg_diag = np.array(A.sum(axis=1)).flatten()
        L = csr_matrix(np.diag(deg_diag)) - A
        kk = min(k, n - 2)
        if kk <= 0: return [0.0]*k
        vals = eigsh(L, k=kk, which="SM", return_eigenvectors=False)
        vals = sorted([float(v) for v in vals])
        while len(vals) < k: vals.append(0.0)
        return vals
    except Exception:
        return [0.0]*k


def graph_morph(g, name):
    rec = {f"{name}_n_nodes": 0, f"{name}_n_edges": 0}
    if g is None: return rec
    x = np.asarray(getattr(g, "x", None) if hasattr(g, "x") else g.get("x"))
    ei = np.asarray(g.edge_index if hasattr(g, "edge_index") else g.get("edge_index"))
    ea = np.asarray(g.edge_attr) if hasattr(g, "edge_attr") and g.edge_attr is not None else None
    if ea is None and "edge_attr" in (g if isinstance(g, dict) else {}):
        ea = np.asarray(g["edge_attr"])
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

    # Edge-attr distributions (col 0=diameter, col 1=length, col 2=tortuosity per builder schema)
    if ea is not None and ea.ndim == 2 and ea.shape[0] > 0:
        # Edges are doubled (a,b) + (b,a) — take every other to avoid double count
        unique_ea = ea[::2] if ea.shape[0] >= 2 else ea
        for col_idx, col_name in [(0, "diam"), (1, "len"), (2, "tort")]:
            if col_idx < unique_ea.shape[1]:
                vals = unique_ea[:, col_idx].astype("float32")
                vals = vals[np.isfinite(vals) & (vals > 0)]
                if vals.size >= 3:
                    for q, qname in [(10, "p10"), (25, "p25"), (50, "p50"), (75, "p75"), (90, "p90")]:
                        rec[f"{name}_{col_name}_{qname}"] = float(np.percentile(vals, q))
                    rec[f"{name}_{col_name}_mean"] = float(vals.mean())
                    rec[f"{name}_{col_name}_sd"] = float(vals.std(ddof=0))
                    sk, kt = safe_skew_kurt(vals)
                    rec[f"{name}_{col_name}_skew"] = sk
                    rec[f"{name}_{col_name}_kurt"] = kt
        # Total physical length (sum of unique edge lengths)
        if unique_ea.shape[1] >= 2:
            rec[f"{name}_total_len_mm"] = float(unique_ea[:, 1].sum())
        # Total volume proxy = sum(pi * r^2 * len) per edge using diameter col 0
        if unique_ea.shape[1] >= 2:
            r = (unique_ea[:, 0] / 2.0).clip(0, 50)
            rec[f"{name}_total_vol_proxy_mm3"] = float((np.pi * r * r * unique_ea[:, 1]).sum())

    # BFS-based depth distribution (Strahler-like proxy via leaf-distance)
    if ei.size > 0:
        adj = [[] for _ in range(n)]
        for s, t in zip(ei[0], ei[1]):
            adj[int(s)].append(int(t)); adj[int(t)].append(int(s))
        leaves = np.where(deg == 1)[0]
        if leaves.size > 0:
            # For each leaf, BFS distance to all other leaves (eccentricity proxy)
            # Use a few sampled leaves to avoid O(n^2)
            sample = leaves[:min(10, len(leaves))]
            depths = []
            for start in sample:
                d1 = [-1]*n; d1[int(start)] = 0; q = deque([int(start)])
                while q:
                    u = q.popleft()
                    for v in adj[u]:
                        if d1[v] < 0: d1[v] = d1[u] + 1; q.append(v)
                d_arr = np.array([x for x in d1 if x >= 0])
                depths.extend(d_arr.tolist())
            depths = np.array(depths, "float32")
            if depths.size >= 3:
                for q, qname in [(50, "p50"), (75, "p75"), (90, "p90")]:
                    rec[f"{name}_depth_{qname}"] = float(np.percentile(depths, q))
                rec[f"{name}_depth_max"] = int(depths.max())
                rec[f"{name}_depth_mean"] = float(depths.mean())

        # Connected components count
        seen = np.zeros(n, bool); cc = 0
        for s in range(n):
            if not seen[s]:
                cc += 1; seen[s] = True; q = deque([s])
                while q:
                    u = q.popleft()
                    for v in adj[u]:
                        if not seen[v]: seen[v] = True; q.append(v)
        rec[f"{name}_n_components"] = cc

    # Spectral signature (top-5 smallest Laplacian eigenvalues)
    eigs = laplacian_eigs(ei, n, k=5)
    for i, ev in enumerate(eigs):
        rec[f"{name}_lap_eig{i}"] = float(ev)

    return rec


def process_one(p_str):
    p = Path(p_str)
    case_id = p.stem.replace("_tri", "")
    try:
        with open(p, "rb") as f:
            d = pickle.load(f)
    except Exception as e:
        return {"case_id": case_id, "error": str(e)}
    rec = {"case_id": case_id, "label": int(d.get("label", -1))}
    for s in ("artery", "vein", "airway"):
        rec.update(graph_morph(d.get(s), s))
    return rec


def main():
    if not CACHE.exists():
        print(f"[abort] {CACHE}"); return
    pkls = sorted(CACHE.glob("*_tri.pkl"))
    print(f"[start] {len(pkls)} pkls; using 24 workers")
    rows = []
    with ProcessPoolExecutor(max_workers=24) as ex:
        futures = {ex.submit(process_one, str(p)): p for p in pkls}
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            rows.append(rec)
            if i % 25 == 0: print(f"  ...{i}/{len(pkls)}")
    keys = sorted({k for r in rows for k in r.keys()})
    out_csv = OUT / "per_structure_morphometrics.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "label"] +
                            [k for k in keys if k not in ("case_id", "label")])
        w.writeheader()
        w.writerows(rows)
    n_cols = len(keys) - 2
    print(f"[done] {len(rows)} rows × {n_cols} feature cols → {out_csv}")
    sample_cols = [k for k in keys if k.startswith("artery_")][:8]
    print(f"  sample artery cols: {sample_cols}")


if __name__ == "__main__":
    main()
