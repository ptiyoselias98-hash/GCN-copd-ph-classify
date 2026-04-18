"""Generate mPAP-stratified 5-fold CV splits.

Reads the master xlsx, pulls mPAP + PH label per patient, buckets mPAP into
4 strata (<18, 18-22, 22-30, >30) and uses StratifiedKFold on a combined
(bucket × label) key so every fold sees a comparable borderline distribution.

Output: data/splits_mpap_stratified.json
Schema (matches load_splits consumer): list of 5 {"train": [...], "val": [...]} dicts
Patient ids are produced by case_to_pinyin(chinese_name) to match the cache/
labels convention.

Usage:
  python gen_mpap_stratified_splits.py \
    --xlsx "../copd-ph患者113例0331.xlsx" \
    --labels "../copdph-gcn-repo/... labels.csv" \
    --out    ./data/splits_mpap_stratified.json
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("gen_splits")


def bucketize_mpap(mpap: float) -> int:
    if np.isnan(mpap):
        return 0
    if mpap < 18:
        return 0
    if mpap < 22:
        return 1  # borderline
    if mpap < 30:
        return 2
    return 3


def case_to_pinyin_local(name: str) -> str:
    try:
        from run_hybrid import case_to_pinyin
        return case_to_pinyin(name)
    except Exception:
        pass
    # Fallback: use pypinyin directly
    try:
        from pypinyin import lazy_pinyin, Style
        parts = lazy_pinyin(str(name), style=Style.NORMAL)
        return "".join(p.capitalize() for p in parts)
    except Exception:
        return "".join(ch for ch in str(name) if ord(ch) < 128)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--xlsx", required=True)
    p.add_argument("--labels_csv", default="",
                   help="optional: labels.csv (patient_id,label). Used to "
                        "intersect the xlsx cohort with the cached cohort.")
    p.add_argument("--out", default="./data/splits_mpap_stratified.json")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name_col", default="name")
    args = p.parse_args()

    df = pd.read_excel(args.xlsx, sheet_name="Sheet1")
    if args.name_col not in df.columns:
        # Fall back to first column
        args.name_col = df.columns[0]
        logger.warning("name_col not found; using %s", args.name_col)

    mpap = pd.to_numeric(df.get("mPAP", pd.Series([np.nan] * len(df))),
                         errors="coerce").values
    y = (df["PH"] == "是").astype(int).values
    names = df[args.name_col].astype(str).tolist()
    pids = [case_to_pinyin_local(n) for n in names]

    # Intersect with labels.csv if provided
    if args.labels_csv and Path(args.labels_csv).exists():
        lab = pd.read_csv(args.labels_csv)
        valid = set(lab["patient_id"].astype(str).tolist())
        keep = [i for i, p in enumerate(pids) if p in valid]
        logger.info("intersect with labels.csv: %d / %d kept", len(keep), len(pids))
        mpap = mpap[keep]; y = y[keep]; pids = [pids[i] for i in keep]

    buckets = np.array([bucketize_mpap(v) for v in mpap])
    # Combined stratum key: bucket * 2 + label -> up to 8 unique strata
    strata = buckets * 2 + y
    logger.info("stratum counts: %s", dict(zip(*np.unique(strata, return_counts=True))))

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    splits = []
    for k, (tr, va) in enumerate(skf.split(pids, strata), 1):
        tr_ids = [pids[i] for i in tr]
        va_ids = [pids[i] for i in va]
        bd_val = int(((buckets[va] == 1)).sum())
        logger.info("  fold %d: train=%d val=%d borderline_val=%d",
                    k, len(tr_ids), len(va_ids), bd_val)
        splits.append({"train": tr_ids, "val": va_ids})

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
