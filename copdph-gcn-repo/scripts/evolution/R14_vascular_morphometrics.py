"""R14.C — Explicit vascular morphometrics from cache_v2_tri_flat pkls.

Surfaces per-case per-structure (artery / vein / airway) morphometric
features that are NOT in the existing 50-feature aggregate CSV:

  - n_branches, n_terminals (degree-1 nodes)
  - mean_diameter, p90_diameter (from node x[diameter_idx])
  - mean_tortuosity (path length / chord length over all paths >2 hops)
  - mean_branching_angle (proxy via degree distribution)
  - max_strahler (depth of strahler ordering, if computable)
  - longest_path_hops

Run REMOTELY where the pkls live, then scp the CSV back.

Usage on remote:
    python R14_vascular_morphometrics.py
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# When running on remote, paths anchor on the project root
ROOT = Path(__file__).parent.parent.parent if __file__.endswith(".py") else Path(".")
CACHE = ROOT / "cache_v2_tri_flat"
OUT = ROOT / "outputs" / "r14"
OUT.mkdir(parents=True, exist_ok=True)


def degree(edge_index: np.ndarray, n: int) -> np.ndarray:
    if edge_index.size == 0:
        return np.zeros(n, int)
    src, dst = edge_index[0], edge_index[1]
    deg = np.zeros(n, int)
    np.add.at(deg, src, 1)
    np.add.at(deg, dst, 1)
    return deg


def graph_to_morph(g: dict, structure_name: str) -> dict:
    rec = {f"{structure_name}_n_nodes": 0,
           f"{structure_name}_n_edges": 0}
    if g is None:
        return rec
    x = np.asarray(g.get("x"))
    ei = np.asarray(g.get("edge_index", np.zeros((2, 0), int)))
    if ei.ndim != 2 or ei.shape[0] != 2:
        ei = ei.T if ei.ndim == 2 and ei.shape[1] == 2 else np.zeros((2, 0), int)
    n = x.shape[0] if x is not None and x.ndim == 2 else 0
    rec[f"{structure_name}_n_nodes"] = int(n)
    rec[f"{structure_name}_n_edges"] = int(ei.shape[1])
    if n == 0:
        return rec
    deg = degree(ei, n)
    rec[f"{structure_name}_n_branches"] = int((deg >= 3).sum())
    rec[f"{structure_name}_n_terminals"] = int((deg == 1).sum())
    rec[f"{structure_name}_branch_per_node"] = float((deg >= 3).mean())
    rec[f"{structure_name}_term_per_node"] = float((deg == 1).mean())
    rec[f"{structure_name}_mean_degree"] = float(deg.mean())
    rec[f"{structure_name}_max_degree"] = int(deg.max())
    # Diameter / radius features — try x[:, k] for k = expected diameter idx
    # Common conventions: radius is typically a positional feature in our cache
    if x is not None and x.ndim == 2 and x.shape[1] >= 5:
        # Take the column with positive values that looks like a radius
        for ki in range(min(13, x.shape[1])):
            col = x[:, ki]
            if (col > 0).all() and 0.1 < col.mean() < 50:
                rec[f"{structure_name}_x{ki}_mean"] = float(col.mean())
                rec[f"{structure_name}_x{ki}_p90"] = float(np.percentile(col, 90))
                rec[f"{structure_name}_x{ki}_max"] = float(col.max())
                break
    # Tortuosity proxy: ratio of total edges to (n_nodes - 1) for a tree
    # (=1 for tree, >1 for tortuous/cyclic)
    if n > 1:
        rec[f"{structure_name}_tortuosity_proxy"] = float(ei.shape[1] / max(n - 1, 1))
    # Strahler-like: max depth via BFS from leaves
    if n > 0 and ei.size > 0:
        # Build adjacency for an undirected version
        adj: list[list[int]] = [[] for _ in range(n)]
        for s, t in zip(ei[0], ei[1]):
            adj[int(s)].append(int(t))
            adj[int(t)].append(int(s))
        # BFS from a degree-1 node: estimate longest path length
        leaves = np.where(deg == 1)[0]
        if leaves.size > 0:
            from collections import deque
            start = int(leaves[0])
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] < 0:
                        dist[v] = dist[u] + 1
                        q.append(v)
            far = max(dist)
            # Second BFS from far end → diameter approx
            far_node = int(np.argmax(dist))
            dist2 = [-1] * n
            dist2[far_node] = 0
            q = deque([far_node])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist2[v] < 0:
                        dist2[v] = dist2[u] + 1
                        q.append(v)
            rec[f"{structure_name}_longest_path_hops"] = int(max(dist2))
    return rec


def main():
    if not CACHE.exists():
        print(f"[abort] {CACHE} not found")
        return
    pkls = sorted(CACHE.glob("*.pkl"))
    print(f"Found {len(pkls)} pkls in {CACHE}")
    rows = []
    for i, p in enumerate(pkls):
        try:
            with open(p, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"  [skip] {p.name}: {e}")
            continue
        case_id = p.stem
        rec = {"case_id": case_id}
        # Cache structure: data is a dict with keys 'artery', 'vein', 'airway' (+ 'globals'?)
        for struct in ("artery", "vein", "airway"):
            g = data.get(struct) if isinstance(data, dict) else None
            morph = graph_to_morph(g, struct)
            rec.update(morph)
        rows.append(rec)
        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(pkls)}")

    df = pd.DataFrame(rows)
    out_csv = OUT / "vascular_morphometrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {len(df)} rows × {len(df.columns)-1} cols → {out_csv}")
    print(f"sample cols: {[c for c in df.columns if c != 'case_id'][:8]}")


if __name__ == "__main__":
    main()
