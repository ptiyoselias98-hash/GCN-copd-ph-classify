## R17 schema discovery (cron fire 2026-04-25 17:43)

**cache_v2_tri_flat schema** is artery+vein (and possibly +airway) concatenated into a single PyG `Data`:
- Source builder: `_remote_v2_flat_convert.py` does `x = torch.cat([art.x, vein.x], dim=0)` and `vein.edge_index += n_artery`
- **Structure id is implicit from node-index range**, NOT a one-hot in node features
- For per-structure split, need: `n_artery_nodes`, `n_vein_nodes`, `n_airway_nodes` recoverable from the source per-structure tri pkls (`cache_tri_v2/{case_id}_tri.pkl` on remote with `artery: Data, vein: Data, airway: Data` keys)

**Fix path for R17**: rewrite morphometrics extractor to:
1. Load source `cache_tri_v2/{case_id}_tri.pkl` (per-structure schema), NOT `cache_v2_tri_flat/{case_id}.pkl` (merged schema)
2. For each of artery / vein / airway separately, run the same `graph_metrics()` we have in R14_morph_v2.py
3. Output CSV with `case_id, artery_n_branches, artery_radius_p90, vein_n_branches, vein_radius_p90, airway_n_branches, ...` etc.

This is the right approach. Run on remote (243+100 = 343 cases × 24-core parallel pool) in next cron fire.
