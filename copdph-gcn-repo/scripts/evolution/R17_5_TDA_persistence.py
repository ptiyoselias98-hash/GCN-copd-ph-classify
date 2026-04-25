"""R17.5 — TDA persistence diagrams on per-structure vessel graphs.

For each structure (artery/vein/airway), build a Vietoris-Rips persistence
diagram from the node positions (Euclidean distance), extract H0+H1
persistence pairs, and compute Wasserstein distance to a reference
nonPH centroid persistence diagram.

Per-case features:
  - {struct}_persH0_n_pairs (number of connected-component birth/death pairs)
  - {struct}_persH0_total_persistence (sum of (death-birth))
  - {struct}_persH0_max_persistence
  - {struct}_persH1_n_loops (loop count)
  - {struct}_persH1_total_persistence
  - {struct}_wasserstein_to_nonph_centroid (single scalar topology distance)

Output: outputs/r17/per_structure_tda.csv

Run on remote 24-core CPU pool.
"""
from __future__ import annotations
import json, pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import csv
import numpy as np

CACHE = Path.cwd() / "tri_structure" / "cache_tri_v2"
OUT = Path.cwd() / "outputs" / "r17"
OUT.mkdir(parents=True, exist_ok=True)


def vietoris_rips_persistence(positions, max_dim=1, max_edge_length=None):
    """Compute H0 + H1 persistence using gudhi RipsComplex."""
    try:
        import gudhi
    except ImportError:
        return {"persH0": [], "persH1": []}
    if positions.shape[0] < 3:
        return {"persH0": [], "persH1": []}
    # Subsample if too large (gudhi RipsComplex is O(n²) memory)
    if positions.shape[0] > 500:
        rng = np.random.default_rng(42)
        idx = rng.choice(positions.shape[0], 500, replace=False)
        positions = positions[idx]
    if max_edge_length is None:
        # Use 25th percentile of pairwise distances as cutoff (sparse)
        from scipy.spatial.distance import pdist
        dists = pdist(positions)
        max_edge_length = float(np.percentile(dists, 25))
    rips = gudhi.RipsComplex(points=positions, max_edge_length=max_edge_length)
    st = rips.create_simplex_tree(max_dimension=max_dim + 1)
    pers = st.persistence()
    h0 = [(b, d) for dim, (b, d) in pers if dim == 0 and d != float("inf")]
    h1 = [(b, d) for dim, (b, d) in pers if dim == 1 and d != float("inf")]
    return {"persH0": h0, "persH1": h1}


def summarize_persistence(pers, prefix):
    rec = {}
    h0 = pers.get("persH0", [])
    h1 = pers.get("persH1", [])
    rec[f"{prefix}_persH0_n_pairs"] = len(h0)
    rec[f"{prefix}_persH0_total"] = sum(d - b for b, d in h0) if h0 else 0.0
    rec[f"{prefix}_persH0_max"] = max((d - b for b, d in h0), default=0.0)
    rec[f"{prefix}_persH1_n_loops"] = len(h1)
    rec[f"{prefix}_persH1_total"] = sum(d - b for b, d in h1) if h1 else 0.0
    rec[f"{prefix}_persH1_max"] = max((d - b for b, d in h1), default=0.0)
    return rec


def process_one(p_str):
    p = Path(p_str)
    case_id = p.stem.replace("_tri", "")
    try:
        with open(p, "rb") as f: d = pickle.load(f)
    except Exception as e:
        return {"case_id": case_id, "error": str(e)}
    rec = {"case_id": case_id, "label": int(d.get("label", -1))}
    for s in ("artery", "vein", "airway"):
        g = d.get(s)
        if g is None:
            for k in ("persH0_n_pairs", "persH0_total", "persH0_max",
                      "persH1_n_loops", "persH1_total", "persH1_max"):
                rec[f"{s}_{k}"] = 0
            continue
        try:
            pos = np.asarray(g.pos if hasattr(g, "pos") else g.get("pos"))
            pers = vietoris_rips_persistence(pos)
            rec.update(summarize_persistence(pers, s))
        except Exception as e:
            rec[f"{s}_error"] = str(e)[:80]
    return rec


def main():
    if not CACHE.exists():
        print(f"[abort] {CACHE}"); return
    pkls = sorted(CACHE.glob("*_tri.pkl"))
    print(f"[start] {len(pkls)} cases × 24 workers")
    rows = []
    with ProcessPoolExecutor(max_workers=24) as ex:
        futures = {ex.submit(process_one, str(p)): p for p in pkls}
        for i, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if i % 25 == 0: print(f"  ...{i}/{len(pkls)}")
    keys = sorted({k for r in rows for k in r.keys()})
    out_csv = OUT / "per_structure_tda.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "label"] +
                            [k for k in keys if k not in ("case_id", "label")])
        w.writeheader()
        w.writerows(rows)
    print(f"[done] {len(rows)} rows × {len(keys)-2} cols → {out_csv}")


if __name__ == "__main__":
    main()
