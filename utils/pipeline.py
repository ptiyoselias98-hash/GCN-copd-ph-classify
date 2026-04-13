"""NIfTI loading + full per-patient processing pipeline."""
from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import ndimage

from .graph_builder import VascularGraphBuilder
from .quantification import (
    AirwayQuantifier,
    ParenchymaQuantifier,
    VascularQuantifier,
)
from .skeleton import VesselSkeleton

logger = logging.getLogger(__name__)


MASK_CANDIDATES = {
    "artery": ["artery.nii.gz", "artery.nii"],
    "vein": ["vein.nii.gz", "vein.nii"],
    "lung": ["lung.nii.gz", "lung.nii"],
    "airway": ["airway.nii.gz", "airway.nii"],
    "ct": ["ct.nii.gz", "ct.nii"],
}


def _load_nifti(path: Path) -> tuple[np.ndarray, tuple]:
    import nibabel as nib
    img = nib.load(str(path))
    arr = np.asarray(img.get_fdata(), dtype=np.float32)
    zooms = img.header.get_zooms()[:3]
    spacing = tuple(float(z) for z in zooms)
    return arr, spacing


def _find_file(folder: Path, candidates: List[str]) -> Optional[Path]:
    for name in candidates:
        p = folder / name
        if p.exists():
            return p
    return None


def _load_patient(folder: Path) -> Optional[dict]:
    data: Dict[str, np.ndarray] = {}
    spacing = (1.0, 1.0, 1.0)
    for key in ("artery", "vein", "lung", "airway"):
        path = _find_file(folder, MASK_CANDIDATES[key])
        if path is None:
            logger.warning("Missing %s mask for %s", key, folder.name)
            return None
        arr, sp = _load_nifti(path)
        data[key] = (arr > 0).astype(np.uint8)
        spacing = sp
    ct_path = _find_file(folder, MASK_CANDIDATES["ct"])
    data["ct"] = _load_nifti(ct_path)[0] if ct_path is not None else None
    data["spacing"] = spacing
    return data


def _build_graph_for_patient(masks: dict, config: dict, label: int) -> Optional[dict]:
    artery = masks["artery"]
    vein = masks["vein"]
    lung = masks["lung"]
    airway = masks["airway"]
    spacing = masks["spacing"]
    ct = masks.get("ct")

    vessel = ((artery + vein) > 0).astype(np.uint8)
    if vessel.sum() == 0:
        return None

    skel_cfg = config.get("skeleton", {})
    skeleton = VesselSkeleton(min_branch_length=skel_cfg.get("min_branch_length", 3))
    skel = skeleton.extract_skeleton(vessel)
    if skel.sum() == 0:
        return None

    classified = skeleton.classify_voxels(skel)
    branches = skeleton.trace_branches(skel, classified)
    if not branches:
        return None

    # Compute distance transform ONCE per-patient (shared across all branches)
    vessel_dt = ndimage.distance_transform_edt(vessel > 0, sampling=np.array(spacing))
    branch_features = [
        skeleton.compute_branch_features(b, vessel, ct_volume=ct, spacing=spacing, dt=vessel_dt)
        for b in branches
    ]

    gcfg = config.get("graph", {})
    builder = VascularGraphBuilder(
        spatial_edge_threshold=gcfg.get("spatial_edge_threshold", 15.0),
        add_spatial_edges=gcfg.get("add_spatial_edges", True),
        use_directed=gcfg.get("use_directed", False),
    )
    graph = builder.build_graph(branches, branch_features, label=label)

    qcfg = config.get("quantification", {})
    vq = VascularQuantifier(spacing=spacing)
    bv = vq.compute_blood_volume_metrics(
        vessel, lung,
        bv5_threshold=qcfg.get("bv5_threshold", 5.0),
        bv10_threshold=qcfg.get("bv10_threshold", 10.0),
    )
    pruning = vq.compute_pruning_index(branches, branch_features)
    av = vq.compute_artery_vein_ratio(artery, vein, lung)
    vascular = {**bv, **pruning, **av}

    if ct is not None:
        pq = ParenchymaQuantifier(spacing=spacing)
        parenchyma = pq.compute_laa(ct, lung, threshold=qcfg.get("laa_threshold", -950))
    else:
        parenchyma = {"laa_pct": 0.0, "mean_lung_density": 0.0}

    aq = AirwayQuantifier(spacing=spacing)
    ct_for_airway = ct if ct is not None else np.zeros_like(airway, dtype=np.float32)
    airway_feats = aq.compute_airway_metrics(airway, ct_for_airway, lung)

    return {
        "graph": graph,
        "features": {
            "vascular": vascular,
            "parenchyma": parenchyma,
            "airway": airway_feats,
        },
        "label": int(label),
    }


def process_dataset(
    data_dir: str,
    labels: Dict[str, int],
    config: dict,
    cache_dir: Optional[str] = None,
) -> List[dict]:
    root = Path(data_dir)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    out: List[dict] = []
    total = len(labels)
    for i, (case_id, label) in enumerate(labels.items()):
        folder = root / case_id
        if not folder.is_dir():
            logger.warning("Missing folder: %s", folder)
            continue

        cache_path = Path(cache_dir) / f"{case_id}.pkl" if cache_dir else None
        if cache_path and cache_path.exists():
            try:
                with cache_path.open("rb") as f:
                    entry = pickle.load(f)
                entry["patient_id"] = case_id
                out.append(entry)
                if (i + 1) % 10 == 0:
                    logger.info("[%d/%d] cache hit", i + 1, total)
                continue
            except Exception as exc:  # noqa: BLE001
                logger.warning("cache read failed %s: %s", case_id, exc)

        logger.info("[%d/%d] processing %s (label=%d)", i + 1, total, case_id, label)
        try:
            masks = _load_patient(folder)
            if masks is None:
                continue
            entry = _build_graph_for_patient(masks, config, label)
        except Exception as exc:  # noqa: BLE001
            logger.exception("failed %s: %s", case_id, exc)
            continue
        if entry is None:
            continue
        entry["patient_id"] = case_id

        if cache_path:
            try:
                with cache_path.open("wb") as f:
                    pickle.dump({k: v for k, v in entry.items() if k != "patient_id"}, f)
            except Exception as exc:  # noqa: BLE001
                logger.warning("cache write failed %s: %s", case_id, exc)

        out.append(entry)

    logger.info("processed %d/%d patients", len(out), total)
    return out
