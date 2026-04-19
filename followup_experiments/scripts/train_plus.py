#!/usr/bin/env python3
"""
train_plus.py — extended entry point covering the follow-up plan.

Variants
--------
  short         gold-subset 5-fold CV on Excel-confirmed ~105 patients
                (labels_gold.csv + splits_gold.json, vanilla classifier)
  medium        SAME gold subset WITH the Sprint 5 v2 triple
                (focal + node-drop + mPAP regression aux head, weight=0.1)
  medium_youden medium + per-fold Youden's J threshold calibration on val probs
                (primary metrics at Youden threshold; argmax-0.5 metrics
                preserved as *_argmax keys — Sprint 3 convention)

Note: mPAP only exists inside the Excel sheet, so every variant that uses
the mPAP aux head is restricted to the 105 Excel-matched cases. Running on
all 197 would leave 92 cases unsupervised for the aux head, which is why
the "full-197 with tricks" variant was removed.

Common flags that override variant presets:
  --labels PATH               labels.csv override
  --splits PATH               JSON list of {"train":[...],"val":[...]}
  --mpap_lookup PATH          JSON case_id -> mPAP
  --use_focal                 enable focal loss
  --node_drop FLOAT           augmentation prob (0 disables)
  --mpap_aux                  enable mPAP regression auxiliary head
  --mpap_weight FLOAT         aux loss weight (default 0.1)

Writes cv_results.json + history.json per fold under --output_dir.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pickle
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from gcn_models import build_model  # noqa: E402
from graph_builder import normalize_graph_features  # noqa: E402
from utils.training_plus import (  # noqa: E402
    cross_validate_plus,
    cross_validate_radiomics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_plus")


def load_labels(path: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row.get("case_id") or row.get("patient_id")
            if not cid:
                continue
            out[cid] = int(row["label"])
    return out


def load_cache(cache_dir: Path, labels: Dict[str, int]) -> List[dict]:
    out = []
    for case_id, label in labels.items():
        p = cache_dir / f"{case_id}.pkl"
        if not p.exists():
            log.warning("cache miss: %s", case_id)
            continue
        with p.open("rb") as f:
            entry = pickle.load(f)
        g = entry["graph"]
        if getattr(g, "y", None) is None or int(g.y.item()) != int(label):
            g.y = torch.tensor([int(label)], dtype=torch.long)
        entry["graph"] = g
        entry["label"] = int(label)
        entry["patient_id"] = case_id
        out.append(entry)
    return out


def load_json(path: Optional[Path]):
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def apply_variant(args) -> None:
    """Fill in variant presets without overriding explicit user flags."""
    v = args.variant
    if v == "short":
        args.labels = args.labels or "data/labels_gold.csv"
        args.splits = args.splits or "data/splits_gold.json"
    elif v in ("medium", "medium_youden", "medium_youden_rep"):
        args.labels = args.labels or "data/labels_gold.csv"
        args.splits = args.splits or "data/splits_gold.json"
        args.mpap_lookup = args.mpap_lookup or "data/mpap_lookup_gold.json"
        if not args.no_focal:
            args.use_focal = True
        if args.node_drop is None:
            args.node_drop = 0.1
        args.mpap_aux = True
        if v in ("medium_youden", "medium_youden_rep"):
            args.youden = True
        if v == "medium_youden_rep" and args.repeats <= 1:
            args.repeats = 3
    elif v in ("mode_gcn", "mode_hybrid", "mode_radiomics"):
        # Three-mode ablation: gold subset + Youden + multi-seed CV.
        args.labels = args.labels or "data/labels_gold.csv"
        args.splits = args.splits or "data/splits_gold.json"
        args.mpap_lookup = args.mpap_lookup or "data/mpap_lookup_gold.json"
        args.youden = True
        if args.repeats <= 1:
            args.repeats = 3
        if v == "mode_gcn":
            args.mode = "pulmonary_gcn"
            if not args.no_focal:
                args.use_focal = True
            if args.node_drop is None:
                args.node_drop = 0.1
            args.mpap_aux = True
        elif v == "mode_hybrid":
            args.mode = "hybrid"
            if not args.no_focal:
                args.use_focal = True
            if args.node_drop is None:
                args.node_drop = 0.1
            args.mpap_aux = True
        else:
            args.mode = "radiomics_only"
            if not args.no_focal:
                args.use_focal = True
            args.mpap_aux = False
            args.node_drop = 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant",
                    choices=["short", "medium", "medium_youden",
                             "medium_youden_rep",
                             "mode_gcn", "mode_hybrid", "mode_radiomics",
                             "none"],
                    default="none",
                    help="Preset (short=gold vanilla, medium=gold + v2 triple, "
                         "medium_youden=medium + Youden threshold, "
                         "medium_youden_rep=Youden+3×5-fold, "
                         "mode_*=ablation on encoder type, none=manual)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--splits", default=None)
    ap.add_argument("--mpap_lookup", default=None)
    ap.add_argument("--output_dir", default="outputs/followup")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--device", default="auto")

    ap.add_argument("--use_focal", action="store_true")
    ap.add_argument("--no_focal", action="store_true",
                    help="Disable focal loss (overrides variant preset)")
    ap.add_argument("--node_drop", type=float, default=None,
                    help="Node drop probability (0 disables)")
    ap.add_argument("--mpap_aux", action="store_true")
    ap.add_argument("--mpap_weight", type=float, default=0.1)
    ap.add_argument("--youden", action="store_true",
                    help="Per-fold Youden's J threshold on val probs (primary metrics)")
    ap.add_argument("--mode",
                    choices=["pulmonary_gcn", "hybrid", "radiomics_only"],
                    default="pulmonary_gcn",
                    help="Encoder mode: pulmonary_gcn (default), hybrid (GCN+radiomics), "
                         "radiomics_only (MLP on cached vascular+airway scalars)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="Number of distinct seeded k-fold partitions to aggregate")
    ap.add_argument("--seed_base", type=int, default=42,
                    help="First seed; subsequent repeats use seed_base+r")

    args = ap.parse_args()
    apply_variant(args)

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    log.info("device=%s", device)

    cfg_path = HERE / args.config if not os.path.isabs(args.config) else Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = dict(cfg.get("model_v2") or cfg["model"])
    training_cfg = dict(cfg["training"])
    if args.epochs is not None:
        training_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        training_cfg["batch_size"] = args.batch_size

    cache_dir = Path(args.cache_dir or cfg["data"]["cache_dir"])
    labels_csv = Path(args.labels or cfg["data"]["labels_csv"])
    labels = load_labels(labels_csv)
    log.info("labels: %d cases in %s", len(labels), labels_csv)

    dataset = load_cache(cache_dir, labels)
    if not dataset:
        log.error("no cached graphs — run rebuild_cache_joint.py first")
        sys.exit(2)
    log.info("loaded %d graphs from cache %s", len(dataset), cache_dir)

    feat_dim = int(dataset[0]["graph"].x.shape[1])
    if feat_dim != 15:
        log.error("feat dim %d != 15", feat_dim)
        sys.exit(3)
    graphs_norm = normalize_graph_features([d["graph"] for d in dataset])
    for d, g in zip(dataset, graphs_norm):
        d["graph"] = g

    splits = load_json(Path(args.splits)) if args.splits else None
    mpap_lookup = load_json(Path(args.mpap_lookup)) if args.mpap_lookup else None
    if mpap_lookup is not None:
        # Drop None/NaN strings — lookup values should be float | None.
        mpap_lookup = {
            k: (None if v is None else float(v)) for k, v in mpap_lookup.items()
        }

    # Mode → model.type override
    if args.mode == "hybrid":
        model_cfg["type"] = "HybridPulmonaryGCN"
        model_cfg.setdefault("mode", "residual")
        # Radiomics dim derived from cached features at runtime.
        from utils.training_plus import _stack_features  # noqa
        rad_arr = _stack_features(dataset)
        model_cfg["radiomics_dim"] = int(rad_arr.shape[1])
        log.info("hybrid mode: radiomics_dim=%d", model_cfg["radiomics_dim"])
    elif args.mode == "pulmonary_gcn":
        model_cfg["type"] = "PulmonaryGCN"

    model_cfg["in_channels"] = feat_dim
    model_cfg["out_channels"] = 2
    run_cfg = {
        "model": model_cfg,
        "training": training_cfg,
        "output_dir": args.output_dir,
    }

    shim = types.ModuleType("models")
    shim.build_model = build_model
    sys.modules["models"] = shim

    use_focal = args.use_focal and not args.no_focal
    node_drop_p = 0.0 if args.node_drop is None else float(args.node_drop)

    log.info("variant=%s mode=%s repeats=%d focal=%s node_drop=%.2f "
             "mpap_aux=%s (w=%.2f) youden=%s splits=%s",
             args.variant, args.mode, args.repeats, use_focal, node_drop_p,
             args.mpap_aux, args.mpap_weight, args.youden, bool(splits))

    if args.mode == "radiomics_only":
        results = cross_validate_radiomics(
            dataset, run_cfg,
            n_folds=args.n_folds, device=device,
            mpap_lookup=mpap_lookup,
            use_focal=use_focal,
            use_youden=args.youden,
            repeats=args.repeats,
            seed_base=args.seed_base,
        )
    else:
        # Hybrid path needs radiomics tensor attached to each Data object.
        if args.mode == "hybrid":
            from utils.training_plus import _stack_features  # noqa
            rad_arr = _stack_features(dataset)
            # Per-row z-score normalisation across the cohort to stabilise BatchNorm.
            mu = rad_arr.mean(axis=0, keepdims=True)
            sd = rad_arr.std(axis=0, keepdims=True) + 1e-6
            rad_norm = (rad_arr - mu) / sd
            for d, r in zip(dataset, rad_norm):
                # Shape (1, D) so PyG batches as (B, D) instead of flattening.
                d["graph"].radiomics = torch.tensor(r, dtype=torch.float).unsqueeze(0)
            log.info("attached radiomics tensor (1×%d) to %d graphs",
                     rad_arr.shape[1], len(dataset))
        results = cross_validate_plus(
            dataset, run_cfg,
            n_folds=args.n_folds, device=device,
            splits=splits, mpap_lookup=mpap_lookup,
            use_focal=use_focal, node_drop_p=node_drop_p,
            mpap_aux=args.mpap_aux, mpap_aux_weight=args.mpap_weight,
            use_youden=args.youden,
            repeats=args.repeats,
            seed_base=args.seed_base,
        )
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = Path(args.output_dir) / "cv_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "variant": args.variant,
            "mode": args.mode,
            "repeats": args.repeats,
            "seed_base": args.seed_base,
            "use_focal": use_focal,
            "node_drop_p": node_drop_p,
            "mpap_aux": args.mpap_aux,
            "mpap_aux_weight": args.mpap_weight,
            "use_youden": args.youden,
            "splits_file": args.splits,
            "labels_file": str(labels_csv),
            "n_cases": len(dataset),
            "results": results,
        }, f, indent=2)
    log.info("wrote %s", out_path)

    mean = results["mean"]
    order = ["auc", "accuracy", "precision", "sensitivity", "f1", "specificity"]
    print("\n====== 5-fold CV — six-metric summary ======")
    for k in order:
        m = mean.get(k, {}).get("mean", 0.0)
        s = mean.get(k, {}).get("std", 0.0)
        print(f"  {k:12s}: {m:.4f} ± {s:.4f}")


if __name__ == "__main__":
    main()
