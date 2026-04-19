#!/usr/bin/env python3
"""
gen_gold_labels.py — build labels_gold.csv + splits_gold.json for the
113-patient Excel-backed gold subset.

Purpose
-------
Sprint 5 Task 5 trained on server labels.csv (197 case_ids, labels derived
from folder prefix `ph_*` / `nonph_*`).  Excel `copd-ph患者113例0331.xlsx`
contains the 113 clinically-confirmed cases with mPAP.  Only ~105 of the 197
cache entries can be matched back to Excel rows; the remaining ~92 have
no clinical ground truth.

This script:
  1. Pulls cache case_ids + labels.csv from server.
  2. Matches Excel names (pypinyin) to case_ids.
  3. Emits:
       labels_gold.csv  — case_id,label (only matched rows)
       splits_gold.json — 5 folds, mPAP-bucket + label stratified
       mpap_lookup_gold.json — case_id -> mPAP (may be NaN for unknowns)
  4. Uploads all three to server under the standard data dir.

Run locally (Windows).  Requires paramiko, pypinyin, pandas, scikit-learn.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from pypinyin import lazy_pinyin, Style
from sklearn.model_selection import StratifiedKFold

try:
    import paramiko
except ImportError:
    print("[error] pip install paramiko", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------- constants
XLSX = r"C:\Users\cheng\Desktop\copd-ph患者113例0331.xlsx"
HOST = "10.60.147.117"
USER = "imss"
PASS = os.environ.get("IMSS_PASSWORD", "imsslab")

REMOTE_CACHE = "/home/imss/cw/GCN copdnoph copdph/cache"
REMOTE_LABELS = "/home/imss/cw/COPDnonPH COPD-PH /data/tables/labels.csv"
REMOTE_DATA = "/home/imss/cw/GCN copdnoph copdph/data"

LOCAL_OUT = Path(__file__).parent / "gold_data"


# ---------------------------------------------------------------- helpers
def case_to_pinyin(cid: str) -> str:
    """case_id `ph_caohujie_9001..._000` -> `caohujie`."""
    parts = cid.split("_")
    if len(parts) >= 3 and parts[0] in ("nonph", "ph"):
        return parts[1].lower()
    return cid.lower()


def name_to_pinyin(name: str) -> str:
    return "".join(lazy_pinyin(str(name).strip(), style=Style.NORMAL)).lower()


def bucketize(v: float) -> int:
    if np.isnan(v):
        return 0
    if v < 18:
        return 0
    if v < 22:
        return 1
    if v < 30:
        return 2
    return 3


# ---------------------------------------------------------------- main
def main():
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)

    # (1) pull cache ids + labels.csv from server
    print(f"[1/5] pulling cache list + labels from {HOST} ...")
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, 22, USER, PASS, look_for_keys=False,
                allow_agent=False, timeout=20)

    _, out, _ = cli.exec_command(
        f"ls '{REMOTE_CACHE}' | sed 's/\\.pkl$//'")
    cache_ids: List[str] = [l.strip() for l in out.read().decode().splitlines() if l.strip()]
    print(f"      cache_ids: {len(cache_ids)}")

    _, out, _ = cli.exec_command(f"cat '{REMOTE_LABELS}'")
    labels_text = out.read().decode()
    caseid_to_label: Dict[str, int] = {}
    for line in labels_text.strip().splitlines()[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0]:
            try:
                caseid_to_label[parts[0]] = int(parts[1])
            except ValueError:
                continue
    print(f"      labels.csv entries: {len(caseid_to_label)}")

    # (2) read Excel (113 patients)
    print(f"[2/5] loading Excel {XLSX}")
    df = pd.read_excel(XLSX, sheet_name="Sheet1")
    name_col = None
    for c in df.columns:
        cs = str(c)
        if "name" in cs.lower() or "姓名" in cs:
            name_col = c
            break
    if name_col is None:
        name_col = df.columns[0]
    print(f"      name column: '{name_col}'  rows: {len(df)}")

    mpap_col = None
    for c in df.columns:
        if str(c).strip().lower() in ("mpap", "mean_pap", "pap"):
            mpap_col = c
            break
    if mpap_col is None:
        mpap_col = "mPAP"
    mpap_vals = pd.to_numeric(df.get(mpap_col, pd.Series([np.nan] * len(df))),
                              errors="coerce").values
    if "PH" in df.columns:
        ph_labels = (df["PH"].astype(str).str.strip() == "是").astype(int).values
    else:
        ph_labels = np.zeros(len(df), dtype=int)

    # (3) pinyin -> cache_ids index
    pinyin_to_caseids: Dict[str, List[str]] = {}
    for cid in cache_ids:
        pinyin_to_caseids.setdefault(case_to_pinyin(cid), []).append(cid)

    excel_match: List[List[str]] = []
    n_matched = 0
    unmatched: List[str] = []
    for i, row in df.iterrows():
        name = str(row[name_col]).strip()
        py = name_to_pinyin(name)
        cids = pinyin_to_caseids.get(py, [])
        if not cids:
            for cache_py in pinyin_to_caseids:
                if py and (py in cache_py or cache_py in py):
                    cids = pinyin_to_caseids[cache_py]
                    break
        excel_match.append(cids)
        if cids:
            n_matched += 1
        else:
            unmatched.append(f"{name} ({py})")
    print(f"[3/5] matched {n_matched}/{len(df)} Excel rows to cache case_ids")
    if unmatched:
        print("      unmatched (first 10):")
        for u in unmatched[:10]:
            print("        -", u)

    # (4) emit labels_gold.csv + mpap_lookup_gold.json + splits_gold.json
    print("[4/5] emitting gold artifacts")
    gold_caseids: List[str] = []
    gold_labels: List[int] = []
    gold_strata: List[int] = []
    mpap_lookup: Dict[str, float | None] = {}

    for i, cids in enumerate(excel_match):
        if not cids:
            continue
        m = float(mpap_vals[i]) if not np.isnan(mpap_vals[i]) else float("nan")
        y_excel = int(ph_labels[i])  # clinical PH label from Excel
        bucket = bucketize(m)
        for cid in cids:
            # Prefer Excel clinical label (mPAP-backed) over folder-prefix label.
            y = y_excel
            gold_caseids.append(cid)
            gold_labels.append(y)
            gold_strata.append(bucket * 2 + y)
            mpap_lookup[cid] = None if np.isnan(m) else m

    # Cross-check Excel label vs folder-prefix label — flag disagreements.
    mismatches = []
    for cid, y_excel in zip(gold_caseids, gold_labels):
        y_folder = caseid_to_label.get(cid)
        if y_folder is not None and y_folder != y_excel:
            mismatches.append((cid, y_folder, y_excel))
    if mismatches:
        print(f"      [WARN] {len(mismatches)} case_ids have folder-prefix label "
              f"!= Excel PH column (first 5):")
        for cid, yf, ye in mismatches[:5]:
            print(f"        {cid}  folder={yf}  excel={ye}")

    gold_csv = LOCAL_OUT / "labels_gold.csv"
    with gold_csv.open("w", encoding="utf-8", newline="") as f:
        f.write("case_id,label\n")
        for cid, y in zip(gold_caseids, gold_labels):
            f.write(f"{cid},{y}\n")
    print(f"      wrote {gold_csv} ({len(gold_caseids)} rows, "
          f"{sum(gold_labels)} PH / {len(gold_labels) - sum(gold_labels)} nonPH)")

    mpap_json = LOCAL_OUT / "mpap_lookup_gold.json"
    with mpap_json.open("w", encoding="utf-8") as f:
        json.dump(mpap_lookup, f, ensure_ascii=False, indent=2)
    n_with_mpap = sum(1 for v in mpap_lookup.values() if v is not None)
    print(f"      wrote {mpap_json} ({len(mpap_lookup)} entries, "
          f"{n_with_mpap} with mPAP)")

    # 5-fold stratified splits
    strata_arr = np.array(gold_strata)
    # Fold label = bucket*2+y; merge rare strata to avoid StratifiedKFold blowup.
    unique, counts = np.unique(strata_arr, return_counts=True)
    rare = set(unique[counts < 5].tolist())
    if rare:
        strata_arr = np.array([
            s if s not in rare else (s % 2) for s in strata_arr
        ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = []
    for k, (tr_idx, va_idx) in enumerate(
            skf.split(np.arange(len(gold_caseids)), strata_arr), 1):
        tr = [gold_caseids[i] for i in tr_idx]
        va = [gold_caseids[i] for i in va_idx]
        pos_tr = sum(gold_labels[i] for i in tr_idx)
        pos_va = sum(gold_labels[i] for i in va_idx)
        print(f"      fold {k}: train={len(tr)} (pos={pos_tr})  val={len(va)} (pos={pos_va})")
        splits.append({"train": tr, "val": va})

    splits_json = LOCAL_OUT / "splits_gold.json"
    with splits_json.open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    print(f"      wrote {splits_json}")

    # (5) upload to server
    print("[5/5] uploading to server ...")
    sftp = cli.open_sftp()
    for local in (gold_csv, mpap_json, splits_json):
        remote = f"{REMOTE_DATA}/{local.name}"
        sftp.put(str(local), remote)
        print(f"      -> {remote}")
    sftp.close()
    cli.close()
    print("[done]")


if __name__ == "__main__":
    main()
