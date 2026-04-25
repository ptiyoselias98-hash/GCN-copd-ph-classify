"""R14.A2 — Convert graph_stats_v2.json to a per-case CSV for clustering.

The 47-feature graph aggregates per case live in
`outputs/r5/graph_stats_v2.json` as a dict-of-dicts. Flatten to CSV
matching `case_id` column.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
INP = ROOT / "outputs" / "r5" / "graph_stats_v2.json"
OUT = ROOT / "outputs" / "r5" / "cache_v2_tri_flat_aggregates.csv"


def main():
    raw = json.loads(INP.read_text(encoding="utf-8"))
    rows = []
    for cid, stats in raw.items():
        if not isinstance(stats, dict):
            continue
        flat = {"case_id": cid}
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                flat[k] = v
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, (int, float)):
                        flat[f"{k}_{kk}"] = vv
        rows.append(flat)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} rows × {len(df.columns) - 1} feature cols → {OUT}")
    # Print a few feature names
    feats = [c for c in df.columns if c != "case_id"]
    print(f"sample feats: {feats[:8]}")


if __name__ == "__main__":
    main()
