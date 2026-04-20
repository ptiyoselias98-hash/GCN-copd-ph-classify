#!/usr/bin/env python3
"""qa_cache_tri.py -- Sprint 7 Task 2: cache quality-assurance gate.

Hard gate before any training. Any FAIL -> fix rebuild_cache_tri.py and re-run.
WARNs are logged but non-blocking.

Outputs (into --output_dir):
    qa_report.json    per-patient/per-structure result
    qa_summary.txt    pass/fail decision + population stats
    node_count_hist.png
    diameter_dist.png
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import sys
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Tuple

import numpy as np

log = logging.getLogger(__name__)


FAIL = "FAIL"
WARN = "WARN"
OK = "OK"

STRUCTS = ("artery", "vein", "airway")


def _per_graph_checks(struct: str, data) -> List[Tuple[str, str, str]]:
    """Return list of (check_name, severity, OK|FAIL|WARN) triples.

    Severity is the highest severity that this check can produce; the third
    element records the actual outcome. A check that returns OK is treated as
    passed regardless of its severity.
    """
    out: List[Tuple[str, str, str]] = []

    num_nodes = int(data.num_nodes)
    out.append(("num_nodes>0", FAIL, OK if num_nodes > 0 else FAIL))

    x = data.x.numpy() if hasattr(data.x, "numpy") else np.asarray(data.x)
    finite_x = np.isfinite(x).all()
    out.append(("x_no_nan_inf", FAIL, OK if finite_x else FAIL))

    ei = data.edge_index.numpy() if hasattr(data.edge_index, "numpy") else np.asarray(data.edge_index)
    if ei.size == 0:
        out.append(("edge_idx_in_range", FAIL, OK))
    else:
        in_range = (ei >= 0).all() and (ei < num_nodes).all()
        out.append(("edge_idx_in_range", FAIL, OK if in_range else FAIL))

    pos = data.pos.numpy() if hasattr(data.pos, "numpy") else np.asarray(data.pos)
    out.append(("pos_finite", FAIL, OK if np.isfinite(pos).all() else FAIL))

    # diameter dim 0 non-negative
    diam_ok = (x[:, 0] >= 0).all() if x.size else True
    out.append(("diameter_nonneg", WARN, OK if diam_ok else WARN))

    # strahler dim 10 non-negative and integer-like
    if x.shape[1] > 10 and x.size:
        col = x[:, 10]
        strahler_ok = (col >= 0).all() and np.allclose(col, np.round(col))
        out.append(("strahler_valid", WARN, OK if strahler_ok else WARN))
    else:
        out.append(("strahler_valid", WARN, OK))

    # single large connected component (>=80%)
    if num_nodes > 1 and ei.size:
        # union-find
        parent = list(range(num_nodes))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for s, d in zip(ei[0], ei[1]):
            union(int(s), int(d))
        comp_sizes: Dict[int, int] = {}
        for i in range(num_nodes):
            r = find(i)
            comp_sizes[r] = comp_sizes.get(r, 0) + 1
        largest = max(comp_sizes.values())
        frac = largest / num_nodes
        out.append(("largest_cc_pct",
                    WARN,
                    OK if frac >= 0.80 else WARN))
    else:
        # single-node / no-edge graph -- treat as vacuously OK (or placeholder)
        out.append(("largest_cc_pct", WARN, OK))

    return out


def _median_diameter(data) -> float:
    x = data.x.numpy() if hasattr(data.x, "numpy") else np.asarray(data.x)
    if x.size == 0:
        return 0.0
    diam = x[:, 0]
    diam = diam[diam > 0]
    return float(np.median(diam)) if diam.size else 0.0


def run_qa(cache_dir: Path, labels_csv: Path, output_dir: Path,
           old_unified_cache: Path = None) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_ids: List[str] = []
    with labels_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row.get("case_id") or row.get("patient_id")
            if cid:
                expected_ids.append(cid)

    per_patient: Dict[str, Any] = {}
    node_counts = {s: [] for s in STRUCTS}
    diameters = {s: [] for s in STRUCTS}
    airway_real_count = 0
    missing_ids = []
    per_patient_fail_flags = {}

    for cid in expected_ids:
        pkl = cache_dir / f"{cid}_tri.pkl"
        if not pkl.exists():
            missing_ids.append(cid)
            continue
        with pkl.open("rb") as f:
            payload = pickle.load(f)

        patient_entry = {}
        any_fail = False
        for struct in STRUCTS:
            data = payload.get(struct)
            if data is None:
                patient_entry[struct] = {"status": "MISSING"}
                any_fail = True
                continue
            checks = _per_graph_checks(struct, data)
            failed = [c for c, sev, outcome in checks if outcome == FAIL]
            warned = [c for c, sev, outcome in checks if outcome == WARN]
            patient_entry[struct] = {
                "num_nodes": int(data.num_nodes),
                "num_edges": int(data.edge_index.shape[1]) if data.edge_index.numel() else 0,
                "failed": failed, "warned": warned,
                "median_diameter": _median_diameter(data),
            }
            if failed:
                any_fail = True
            node_counts[struct].append(int(data.num_nodes))
            diameters[struct].append(_median_diameter(data))
            if struct == "airway" and int(data.num_nodes) > 1:
                airway_real_count += 1
        per_patient[cid] = patient_entry
        per_patient_fail_flags[cid] = any_fail

    total_expected = len(expected_ids)
    total_loaded = len(per_patient)
    total_fail = sum(1 for v in per_patient_fail_flags.values() if v)

    # population checks
    pop: List[Tuple[str, str, Any]] = []

    chk8_ok = len(missing_ids) == 0 and total_loaded == total_expected
    pop.append(("all_gold_present", FAIL if not chk8_ok else OK,
                f"{total_loaded}/{total_expected}"))

    med_artery = median(node_counts["artery"]) if node_counts["artery"] else 0
    med_vein = median(node_counts["vein"]) if node_counts["vein"] else 0
    med_airway = median(node_counts["airway"]) if node_counts["airway"] else 0
    pop.append(("median_artery>median_vein",
                WARN if med_artery <= med_vein else OK,
                f"artery={med_artery} vein={med_vein}"))

    airway_rate = airway_real_count / max(1, total_loaded)
    pop.append(("airway_real_rate>=0.80",
                WARN if airway_rate < 0.80 else OK,
                f"{airway_real_count}/{total_loaded} ({airway_rate:.2%})"))

    # population diameter ranges
    def _range_check(name, vals, lo, hi):
        med = float(np.median(vals)) if vals else 0.0
        in_range = lo <= med <= hi
        pop.append((name,
                    WARN if not in_range else OK,
                    f"median={med:.2f} expected [{lo},{hi}]"))

    _range_check("artery_median_diam_mm", diameters["artery"], 2.0, 8.0)
    _range_check("vein_median_diam_mm",   diameters["vein"],   1.5, 6.0)
    _range_check("airway_median_diam_mm", diameters["airway"], 3.0, 15.0)

    # cross-cache consistency -- optional, WARN-only
    if old_unified_cache and old_unified_cache.is_dir():
        pairs = []
        for cid in expected_ids:
            old_pkl = old_unified_cache / f"{cid}.pkl"
            if not old_pkl.exists():
                continue
            with old_pkl.open("rb") as f:
                old_p = pickle.load(f)
            old_graph = old_p.get("graph") or old_p.get("data")
            if old_graph is None or cid not in per_patient:
                continue
            old_n = int(getattr(old_graph, "num_nodes", 0))
            new_n = (per_patient[cid].get("artery", {}).get("num_nodes", 0)
                     + per_patient[cid].get("vein", {}).get("num_nodes", 0))
            if old_n > 0 and new_n > 0:
                pairs.append((old_n, new_n))
        if len(pairs) >= 5:
            arr = np.array(pairs, dtype=float)
            r = float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])
            pop.append(("cross_cache_r>0.6",
                        WARN if r <= 0.60 else OK,
                        f"r={r:.3f} n={len(pairs)}"))
        else:
            pop.append(("cross_cache_r>0.6", WARN, "insufficient pairs -- skipped"))
    else:
        pop.append(("cross_cache_r>0.6", WARN, "old cache not provided -- skipped"))

    # final decision
    has_pop_fail = any(sev == FAIL for _, sev, _ in pop if sev == FAIL)
    has_graph_fail = total_fail > 0
    overall = FAIL if (has_pop_fail or has_graph_fail) else OK

    report = {
        "overall": overall,
        "n_expected": total_expected,
        "n_loaded": total_loaded,
        "n_missing": len(missing_ids),
        "missing_ids": missing_ids,
        "per_patient_fail_count": total_fail,
        "population_checks": [{"check": n, "severity": s, "detail": d} for n, s, d in pop],
        "per_patient": per_patient,
    }
    (output_dir / "qa_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # summary
    lines = []
    lines.append(f"QA decision: {overall}")
    lines.append(f"loaded {total_loaded}/{total_expected} patients "
                 f"(missing: {len(missing_ids)})")
    lines.append(f"per-patient FAILs: {total_fail}")
    lines.append("")
    lines.append("Population checks:")
    for n, s, d in pop:
        lines.append(f"  [{s:4}] {n}: {d}")
    lines.append("")
    lines.append("Median node counts:")
    for s in STRUCTS:
        if node_counts[s]:
            lines.append(f"  {s:6}: median={median(node_counts[s]):.0f} "
                         f"min={min(node_counts[s])} max={max(node_counts[s])}")
    lines.append("")
    lines.append(f"Airway real-graph rate: {airway_real_count}/{total_loaded} "
                 f"({airway_rate:.2%})")
    (output_dir / "qa_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    # plots (best-effort -- skip if matplotlib missing)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        bins = np.linspace(0, max([max(v, default=0) for v in node_counts.values()] + [1]), 40)
        for s, color in zip(STRUCTS, ("#c0392b", "#2980b9", "#27ae60")):
            if node_counts[s]:
                ax.hist(node_counts[s], bins=bins, alpha=0.5, label=s, color=color)
        ax.set_xlabel("num_nodes")
        ax.set_ylabel("count")
        ax.set_title("Node count per structure (QA)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "node_count_hist.png", dpi=110)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        data = [diameters[s] for s in STRUCTS]
        ax.violinplot(data, showmedians=True)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(STRUCTS)
        ax.set_ylabel("median diameter (mm)")
        ax.set_title("Per-structure median diameter (QA)")
        fig.tight_layout()
        fig.savefig(output_dir / "diameter_dist.png", dpi=110)
        plt.close(fig)
    except Exception as e:
        log.warning("plot generation skipped: %s", e)

    print("\n".join(lines))
    return 0 if overall == OK else 1


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--old_unified_cache", default=None,
                   help="Optional: path to Phase 1 unified cache for cross-check.")
    args = p.parse_args()

    rc = run_qa(
        cache_dir=Path(args.cache_dir),
        labels_csv=Path(args.labels),
        output_dir=Path(args.output_dir),
        old_unified_cache=Path(args.old_unified_cache) if args.old_unified_cache else None,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
