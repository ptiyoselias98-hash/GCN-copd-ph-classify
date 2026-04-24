"""Remote v2 graph cache builder — kimimaro-based, drop-in for cache_tri schema.

Design (per codex review of expanded pilot 2026-04-21):
  * Skeleton: kimimaro TEASAR (anisotropy=spacing) — handles blob masks
    by forcing tree-only output (no spurious cycles).
  * Contract degree-2 chains; nodes = junctions+endpoints (deg != 2).
  * Per-edge geometry: physical length (mm), median radius (DT, mm), tortuosity.
  * Per-node features: aggregate of incident edges + Strahler order + degree.
  * QC fields: vox/key, p90_edge_len, max_edge_len, n_components, n_skeletons,
    largest_component_vox_frac, dropped_component_vox_frac.
  * Per-structure suspect tag (artery vox/key>=850, vein>=950, airway separate).

Output: {case_id}_tri.pkl with same schema as rebuild_cache_separate.py:
  {
    "label": int,
    "patient_id": str,
    "artery": Data, "vein": Data, "airway": Data,
    "qc": {
      "artery": {...metrics + suspect bool},
      "vein":   {...},
      "airway": {...},
    },
    "builder_version": "v2_kimimaro",
  }

Each Data object:
  x          (N, 12)  float32   — node features
  edge_index (2, E)   int64     — edges (both directions)
  edge_attr  (E, 3)   float32   — [diameter_mm, length_mm, tortuosity]
  pos        (N, 3)   float32   — node spatial position (voxel index space)
  y          (1,)     int64     — graph label
  num_nodes  scalar   int

This script is uploaded to remote and run there via the orchestrator.
"""
import argparse, csv, json, logging, os, pickle, sys, time, traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

log = logging.getLogger("v2builder")


# ---------------------------------------------------------------------------
# Lazy imports — only loaded inside worker so multiprocessing fork is cheap
# ---------------------------------------------------------------------------
def _imports():
    import nibabel as nib
    from scipy import ndimage
    import torch
    from torch_geometric.data import Data
    import kimimaro
    return nib, ndimage, torch, Data, kimimaro


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _contract_skeleton(sk, dt_mm, vox_idx):
    """Walk deg-2 chains; return list of contracted edges with geometry.

    Args:
        sk: kimimaro Skeleton (vertices in physical mm coords).
        dt_mm: 3D distance transform of the WHOLE mask (mm units).
        vox_idx: (V, 3) int — voxel indices of each kimimaro vertex.

    Returns:
        nodes_idx: list[int] — kimimaro vertex IDs that are key points (deg!=2).
        edges: list[dict] — each {a, b, length_mm, radius_mean_mm,
               radius_med_mm, radius_max_mm, tortuosity, n_chain}.
    """
    V = sk.vertices.shape[0]
    if V == 0:
        return [], []
    adj = [[] for _ in range(V)]
    for a, b in sk.edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    deg = np.array([len(a) for a in adj])
    key = [i for i in range(V) if deg[i] != 2]
    key_set = set(key)

    edges = []
    seen = set()
    for k in key:
        for nbr in adj[k]:
            path = [k, nbr]
            prev, cur = k, nbr
            safety = 0
            while cur not in key_set and safety < 1_000_000:
                nxt = [n for n in adj[cur] if n != prev]
                if not nxt:
                    break
                prev = cur
                cur = nxt[0]
                path.append(cur)
                safety += 1
            if cur == k or cur not in key_set:
                continue
            edge_id = tuple(sorted((k, cur)))
            if edge_id in seen:
                continue
            seen.add(edge_id)
            pts = sk.vertices[path]
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            length_mm = float(seg.sum())
            chord_mm = float(np.linalg.norm(pts[-1] - pts[0]))
            tort = length_mm / max(chord_mm, 1e-6)
            # radius lookup: for each vertex, dt_mm at its voxel index
            ids = vox_idx[path]
            rads = dt_mm[ids[:, 0], ids[:, 1], ids[:, 2]]
            edges.append({
                "a": int(k), "b": int(cur),
                "length_mm": length_mm,
                "radius_med_mm": float(np.median(rads)),
                "radius_mean_mm": float(np.mean(rads)),
                "radius_max_mm": float(np.max(rads)),
                "tortuosity": tort,
                "n_chain": len(path),
            })
    return key, edges


def _strahler_orders(num_nodes, edges):
    """Compute Horton-Strahler order on the contracted graph.

    Strahler is defined for trees rooted at a leaf. For our (mostly tree-like)
    graphs we approximate: any node with degree==1 = leaf (order 1); inner
    node = max(child_orders) if all distinct, else max+1.
    """
    if num_nodes == 0:
        return {}
    adj = defaultdict(list)
    for e in edges:
        adj[e["a"]].append(e["b"])
        adj[e["b"]].append(e["a"])
    deg = {n: len(adj[n]) for n in range(num_nodes)}
    leaves = [n for n in range(num_nodes) if deg.get(n, 0) <= 1]
    order = {n: 1 for n in leaves}
    # BFS from leaves inward
    queue = list(leaves)
    seen = set(leaves)
    while queue:
        nxt = []
        for n in queue:
            for nb in adj[n]:
                if nb in seen:
                    continue
                # check if all but one neighbor of nb is already ordered
                ordered_nbs = [adj_n for adj_n in adj[nb] if adj_n in order]
                if len(ordered_nbs) >= deg[nb] - 1:
                    cs = sorted([order[c] for c in ordered_nbs], reverse=True)
                    if len(cs) >= 2 and cs[0] == cs[1]:
                        order[nb] = cs[0] + 1
                    else:
                        order[nb] = cs[0]
                    seen.add(nb)
                    nxt.append(nb)
        queue = nxt
    # any disconnected node → order 1
    for n in range(num_nodes):
        order.setdefault(n, 1)
    return order


def _build_data(skels_by_label, mask, spacing, label, torch_mod, Data):
    """Combine all per-component skeletons → one Data object."""
    if not skels_by_label:
        return _empty_data(label, torch_mod, Data), {}

    # whole-mask DT (codex fix: per-component label mismatch caused 0 radii)
    from scipy import ndimage as ndi
    dt_mm = ndi.distance_transform_edt(mask > 0, sampling=spacing)
    sp = np.array(spacing)
    mask_shape = np.array(mask.shape)

    all_nodes = []          # voxel idx triples
    all_edges = []          # dicts with global-node a, b
    n_skeletons = 0

    for sk_label, sk in skels_by_label.items():
        V = sk.vertices.shape[0]
        if V == 0:
            continue
        n_skeletons += 1
        vox_idx = np.round(sk.vertices / sp).astype(int)
        vox_idx = np.clip(vox_idx, 0, mask_shape - 1)
        key, edges = _contract_skeleton(sk, dt_mm, vox_idx)
        # remap local keys → global node ids
        offset = len(all_nodes)
        local_to_global = {}
        for k in key:
            local_to_global[k] = offset + len(local_to_global)
            all_nodes.append(vox_idx[k].tolist())
        for e in edges:
            all_edges.append({
                "a": local_to_global[e["a"]],
                "b": local_to_global[e["b"]],
                "length_mm": e["length_mm"],
                "radius_med_mm": e["radius_med_mm"],
                "radius_mean_mm": e["radius_mean_mm"],
                "radius_max_mm": e["radius_max_mm"],
                "tortuosity": e["tortuosity"],
                "n_chain": e["n_chain"],
            })

    num_nodes = len(all_nodes)
    if num_nodes == 0:
        return _empty_data(label, torch_mod, Data), {
            "reason": "no_keys", "valid_structure": False,
            "missing_reason": "no_keys_after_contraction",
            "failed_hard_qc": True,
        }

    # node features (12-dim, matching legacy schema)
    x = np.zeros((num_nodes, 12), dtype=np.float32)
    incident_edge_idx = defaultdict(list)
    for ei, e in enumerate(all_edges):
        incident_edge_idx[e["a"]].append(ei)
        incident_edge_idx[e["b"]].append(ei)

    pos = np.array(all_nodes, dtype=np.float32)
    centroid = pos.mean(axis=0) if num_nodes > 0 else np.zeros(3)

    strahler = _strahler_orders(num_nodes, all_edges)

    for n in range(num_nodes):
        eis = incident_edge_idx[n]
        if eis:
            es = [all_edges[i] for i in eis]
            diameters = [e["radius_med_mm"] * 2 for e in es]
            lengths = [e["length_mm"] for e in es]
            torts = [e["tortuosity"] for e in es]
            x[n, 0] = float(np.mean(diameters))                 # diameter
            x[n, 1] = float(np.mean(lengths))                   # length
            x[n, 2] = float(np.mean(torts))                     # tortuosity
            # x[n, 3] ct_density — left at 0 (no CT lookup in v2)
            # x[n, 4:7] orientation — set to mean unit-vector along incident edges
            ovec = np.zeros(3)
            for e in es:
                # use stored chord direction proxy: use position diff to other endpoint
                other = e["b"] if e["a"] == n else e["a"]
                d = pos[other] - pos[n]
                norm = np.linalg.norm(d)
                if norm > 0:
                    ovec += d / norm
            if np.linalg.norm(ovec) > 0:
                ovec /= np.linalg.norm(ovec)
            x[n, 4:7] = ovec
        # x[n, 7:10] centroid — relative position
        x[n, 7:10] = pos[n] - centroid
        x[n, 10] = float(strahler.get(n, 1))                    # strahler
        x[n, 11] = float(len(eis))                              # degree

    # edge_index (both directions)
    src = []; dst = []; eattr = []
    for e in all_edges:
        src.append(e["a"]); dst.append(e["b"])
        ef = [e["radius_med_mm"] * 2, e["length_mm"], e["tortuosity"]]
        eattr.append(ef)
        src.append(e["b"]); dst.append(e["a"])
        eattr.append(ef)

    if src:
        edge_index = torch_mod.tensor([src, dst], dtype=torch_mod.long)
        edge_attr = torch_mod.tensor(np.array(eattr, dtype=np.float32))
    else:
        edge_index = torch_mod.zeros((2, 0), dtype=torch_mod.long)
        edge_attr = torch_mod.zeros((0, 3), dtype=torch_mod.float)

    data = Data(
        x=torch_mod.tensor(x, dtype=torch_mod.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=torch_mod.tensor(pos),
        y=torch_mod.tensor([label], dtype=torch_mod.long),
        num_nodes=num_nodes,
    )

    qc = {
        "n_skeletons": n_skeletons,
        "num_nodes": num_nodes,
        "num_edges": len(all_edges),
        "edge_len_mm_p50": float(np.median([e["length_mm"] for e in all_edges])) if all_edges else 0.0,
        "edge_len_mm_p90": float(np.percentile([e["length_mm"] for e in all_edges], 90)) if all_edges else 0.0,
        "edge_len_mm_max": float(max(e["length_mm"] for e in all_edges)) if all_edges else 0.0,
        "radius_mm_p50": float(np.median([e["radius_med_mm"] for e in all_edges])) if all_edges else 0.0,
    }
    return data, qc


def _empty_data(label, torch_mod, Data):
    return Data(
        x=torch_mod.zeros((1, 12), dtype=torch_mod.float),
        edge_index=torch_mod.zeros((2, 0), dtype=torch_mod.long),
        edge_attr=torch_mod.zeros((0, 3), dtype=torch_mod.float),
        pos=torch_mod.zeros((1, 3), dtype=torch_mod.float),
        y=torch_mod.tensor([label], dtype=torch_mod.long),
        num_nodes=1,
    )


# ---------------------------------------------------------------------------
# Per-structure builder
# ---------------------------------------------------------------------------
def build_structure(mask_path, label, struct, torch_mod, Data, kimimaro_mod, nib_mod):
    """Build one Data object for one structure of one case."""
    img = nib_mod.load(str(mask_path))
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
    # Robust mask extraction: NIfTI masks in this dataset use -2048 as the
    # background sentinel and carry raw HU inside the structure. A plain
    # `(arr > 0)` filter misses plain-scan airway/lung where lumen-air HU is
    # negative (~-900). Use `!= -2048` to catch any non-background voxel.
    raw = np.asarray(img.dataobj)
    arr = (raw != -2048).astype(np.uint32)
    mask_vox = int(arr.sum())

    if mask_vox < 100:
        return _empty_data(label, torch_mod, Data), {
            "skipped": True, "reason": "tiny_mask", "mask_vox": mask_vox,
            "spacing": spacing, "valid_structure": False,
            "missing_reason": "empty_mask", "failed_hard_qc": False,
        }

    # whole-mask connected components for QC
    from scipy import ndimage as ndi
    labeled, n_comp = ndi.label(arr > 0, structure=np.ones((3,3,3), np.uint8))
    if n_comp > 0:
        sizes = ndi.sum(arr, labeled, index=range(1, n_comp + 1))
        largest = int(max(sizes))
        sum_below_dust = int(sum(s for s in sizes if s < 1000))
    else:
        largest = 0; sum_below_dust = 0

    try:
        t0 = time.time()
        # Per-structure TEASAR tuning. Smaller scale/const + lower pdrf_exponent
        # give finer skeletons (more keypoints) on thin vessels.
        # Vessels (1-3 mm radius) need scale~1.0/const~5 to avoid over-pruning;
        # airway (larger radius) tolerates the slightly looser default.
        if struct in ("artery", "vein"):
            teasar = {
                "scale": 1.0, "const": 5,
                "pdrf_scale": 100000, "pdrf_exponent": 2,
                "soma_acceptance_threshold": 3500,
                "soma_detection_threshold": 750,
                "soma_invalidation_scale": 1.0,
                "soma_invalidation_const": 300,
                "max_paths": None,
            }
        else:  # airway
            teasar = {
                "scale": 1.5, "const": 10,
                "pdrf_scale": 100000, "pdrf_exponent": 4,
                "soma_acceptance_threshold": 3500,
                "soma_detection_threshold": 750,
                "soma_invalidation_scale": 1.0,
                "soma_invalidation_const": 300,
                "max_paths": None,
            }
        skels = kimimaro_mod.skeletonize(
            arr,
            teasar_params=teasar,
            anisotropy=spacing,
            dust_threshold=1000,
            progress=False,
            parallel=1,
        )
        elapsed = round(time.time() - t0, 1)
        # ROUND 7/8 — stamp provenance fields for auditable reproducibility
        try:
            _kv = kimimaro_mod.__version__
        except Exception:
            _kv = "unknown"
        try:
            import subprocess as _sp
            _git_sha = _sp.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(Path(__file__).parent),
                stderr=_sp.DEVNULL,
            ).decode().strip()
        except Exception:
            _git_sha = "unknown"
        provenance = {
            "builder_version": "v2_kimimaro",
            "git_sha": _git_sha,
            "kimimaro_version": _kv,
            "teasar_params": teasar,
            "dust_threshold": 1000,
            "mask_sentinel": -2048,
        }
    except Exception as ex:
        return _empty_data(label, torch_mod, Data), {
            "skipped": True, "reason": f"teasar_failed: {ex!r}",
            "mask_vox": mask_vox, "spacing": spacing,
            "elapsed_s": -1, "valid_structure": False,
            "missing_reason": "teasar_failed", "failed_hard_qc": True,
        }

    data, qc_extra = _build_data(skels, arr, spacing, label, torch_mod, Data)
    qc = {
        "skipped": False,
        "mask_vox": mask_vox,
        "spacing": spacing,
        "n_components_mask": n_comp,
        "largest_comp_vox": largest,
        "largest_comp_frac": round(largest / max(mask_vox, 1), 4),
        "dust_dropped_vox": sum_below_dust,
        "dust_dropped_frac": round(sum_below_dust / max(mask_vox, 1), 4),
        "elapsed_s": elapsed,
        **qc_extra,
    }
    qc["vox_per_key"] = round(mask_vox / max(qc.get("num_nodes", 1), 1), 1)
    # ROUND 7/8 — merge provenance into qc so downstream pkls carry it
    try:
        qc.update(provenance)
    except NameError:
        pass
    # per-structure suspect tag (warning, not fail)
    vk = qc["vox_per_key"]
    nn = qc.get("num_nodes", 0)
    if struct == "artery":
        qc["suspect"] = vk >= 850 or qc.get("edge_len_mm_p50", 0) >= 15
    elif struct == "vein":
        qc["suspect"] = vk >= 950 or qc.get("edge_len_mm_p50", 0) >= 15
    else:  # airway
        qc["suspect"] = False
    # hard-fail tier per codex: vox/key>2000 OR (mask_vox>100k AND num_nodes<100)
    if struct in ("artery", "vein"):
        qc["failed_hard_qc"] = bool(
            vk > 2000 or (mask_vox > 100_000 and nn < 100)
        )
    else:
        qc["failed_hard_qc"] = False
    qc["valid_structure"] = (not qc.get("skipped", False)) and (not qc["failed_hard_qc"])
    qc["missing_reason"] = None
    # top-3 longest edges with endpoint indices for spot-check overlay
    if data.edge_attr is not None and data.edge_attr.shape[0] > 0:
        # edge_attr columns: [diameter, length, tortuosity]; edges duplicated (a,b)+(b,a)
        # take unique edges (first half)
        n_edges = data.edge_attr.shape[0] // 2
        lens = [(i, float(data.edge_attr[2 * i, 1].item())) for i in range(n_edges)]
        lens.sort(key=lambda r: -r[1])
        top3 = []
        for i, L in lens[:3]:
            a = int(data.edge_index[0, 2 * i].item())
            b = int(data.edge_index[1, 2 * i].item())
            top3.append({"a": a, "b": b, "length_mm": round(L, 2)})
        qc["top3_longest_edges"] = top3
    else:
        qc["top3_longest_edges"] = []
    return data, qc


# ---------------------------------------------------------------------------
# Per-case worker (used by multiprocessing)
# ---------------------------------------------------------------------------
def _find_mask(case_dir, struct):
    for ext in (".nii.gz", ".nii"):
        p = case_dir / f"{struct}{ext}"
        if p.exists():
            return p
    return None


def _process_case(args):
    case_id, label, data_dirs, output_cache = args
    nib_mod, _, torch_mod, Data, kimimaro_mod = _imports()
    out_path = Path(output_cache) / f"{case_id}_tri.pkl"
    if out_path.exists():
        return (case_id, "skipped_exists", None)

    # pick the data dir that has at least artery
    case_dir = None
    for d in data_dirs:
        cd = Path(d) / case_id
        if cd.is_dir() and any((cd / f"artery{e}").exists() for e in (".nii.gz", ".nii")):
            case_dir = cd
            break
        if cd.is_dir() and any((cd / f"{s}{e}").exists()
                                for s in ("artery", "vein", "airway")
                                for e in (".nii.gz", ".nii")):
            case_dir = cd
            break
    if case_dir is None:
        return (case_id, "missing_dir", {"data_dirs_tried": data_dirs})

    # ROUND 7/8 — top-level provenance stamping for auditable reproducibility
    try:
        _kv_top = kimimaro_mod.__version__
    except Exception:
        _kv_top = "unknown"
    try:
        import subprocess as _sp
        _git_sha_top = _sp.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).parent),
            stderr=_sp.DEVNULL,
        ).decode().strip()
    except Exception:
        _git_sha_top = "unknown"
    result = {
        "label": int(label), "patient_id": case_id, "qc": {},
        "builder_version": "v2_kimimaro",
        "git_sha": _git_sha_top,
        "kimimaro_version": _kv_top,
        "mask_sentinel": -2048,
    }
    for struct in ("artery", "vein", "airway"):
        mp = _find_mask(case_dir, struct)
        if mp is None:
            result[struct] = _empty_data(int(label), torch_mod, Data)
            result["qc"][struct] = {
                "skipped": True, "reason": "no_mask",
                "valid_structure": False, "missing_reason": "no_mask",
                "failed_hard_qc": False,
            }
            continue
        try:
            data, qc = build_structure(mp, int(label), struct,
                                        torch_mod, Data, kimimaro_mod, nib_mod)
            result[struct] = data
            result["qc"][struct] = qc
        except Exception as e:
            result[struct] = _empty_data(int(label), torch_mod, Data)
            result["qc"][struct] = {
                "skipped": True, "reason": f"build_failed: {e!r}",
                "tb": traceback.format_exc()[:600],
                "valid_structure": False, "missing_reason": "build_failed",
                "failed_hard_qc": True,
            }

    with out_path.open("wb") as f:
        pickle.dump(result, f)
    summary = {s: result["qc"][s].get("num_nodes",
                                       result["qc"][s].get("mask_vox", 0))
               for s in ("artery", "vein", "airway")}
    return (case_id, "done", summary)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True, help="CSV file: case_id, label")
    p.add_argument("--data_dirs", nargs="+", required=True,
                   help="One or more directories containing per-case mask folders.")
    p.add_argument("--output_cache", required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--limit", type=int, default=0,
                   help="Process only first N cases (for sample runs).")
    p.add_argument("--cases", nargs="*",
                   help="Only process these case IDs (for sampling).")
    args = p.parse_args()

    labels = {}
    with open(args.labels, "r") as f:
        for row in csv.DictReader(f):
            cid = row.get("case_id") or row.get("patient_id")
            if cid:
                labels[cid] = int(row["label"])
    log.info("Loaded labels for %d cases from %s", len(labels), args.labels)

    if args.cases:
        keep = set(args.cases)
        labels = {k: v for k, v in labels.items() if k in keep}
        log.info("Filtered to %d cases via --cases", len(labels))
    if args.limit > 0:
        items = list(labels.items())[:args.limit]
        labels = dict(items)
        log.info("Limited to first %d cases", len(labels))

    os.makedirs(args.output_cache, exist_ok=True)
    tasks = [(cid, lbl, args.data_dirs, args.output_cache)
             for cid, lbl in labels.items()]
    total = len(tasks)
    log.info("Total: %d cases | workers: %d | output: %s",
             total, args.workers, args.output_cache)

    done = skipped = missing = errored = 0
    t0 = time.time()
    if args.workers <= 1:
        for t in tasks:
            cid, status, info = _process_case(t)
            if status == "done":
                done += 1
                log.info("[%d/%d done] %s  %s", done + skipped, total, cid, info)
            elif status == "skipped_exists":
                skipped += 1
            elif status == "missing_dir":
                missing += 1
                log.warning("[miss] %s", cid)
            else:
                errored += 1
                log.error("[err] %s  %s", cid, status)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        with ctx.Pool(args.workers) as pool:
            for cid, status, info in pool.imap_unordered(_process_case, tasks):
                if status == "done":
                    done += 1
                    log.info("[%d/%d done] %s  %s",
                             done + skipped + missing + errored, total, cid, info)
                elif status == "skipped_exists":
                    skipped += 1
                    log.info("[skip] %s exists", cid)
                elif status == "missing_dir":
                    missing += 1
                    log.warning("[miss] %s", cid)
                else:
                    errored += 1
                    log.error("[err] %s  %s", cid, status)

    elapsed = round(time.time() - t0, 1)
    log.info("Summary: done=%d skip=%d miss=%d err=%d (total=%d)  in %ss",
             done, skipped, missing, errored, total, elapsed)
    log.info("Cache written to %s", args.output_cache)


if __name__ == "__main__":
    main()
