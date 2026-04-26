"""R21.B — Generate 5-fold stratified splits for unified-301 cohort.

Stratify on (label × protocol) to ensure each fold has both PH/nonPH and
contrast/plain-scan representation. Output schema matches existing
`data/splits_contrast_only/fold_K/{train,val}.txt`.

Run on remote inside `/home/imss/cw/GCN copdnoph copdph/`.

Outputs: data/splits_unified_301/fold_{1..5}/{train,val}.txt
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/imss/cw/GCN copdnoph copdph")
CACHE = ROOT / "cache_tri_v2_unified301"
LABELS = ROOT / "data" / "labels_extended_382.csv"
PROTO = ROOT / "data" / "case_protocol_extended.csv"
OUT = ROOT / "data" / "splits_unified_301"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    cached = sorted([p.stem.replace("_tri", "") for p in CACHE.glob("*_tri.pkl")])
    print(f"cached: {len(cached)}")
    labels = {}
    with LABELS.open() as f:
        for r in csv.DictReader(f):
            cid = r.get("case_id") or r.get("patient_id")
            if cid:
                labels[cid] = int(r["label"])
    proto = {}
    with PROTO.open() as f:
        for r in csv.DictReader(f):
            proto[r["case_id"]] = r["protocol"]

    cases = [c for c in cached if c in labels]
    y_label = np.array([labels[c] for c in cases])
    y_proto = np.array([proto.get(c, "unknown") for c in cases])
    # stratify on label*protocol (4 groups): label0_contrast, label0_plain,
    # label1_contrast, label1_plain (label1_plain ~empty so collapses to 3 groups)
    y_combined = np.array([f"L{l}_{p}" for l, p in zip(y_label, y_proto)])
    print(f"effective: {len(cases)} cases")
    for grp in sorted(set(y_combined.tolist())):
        print(f"  {grp}: {(y_combined == grp).sum()}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(cases, y_combined), 1):
        fold_dir = OUT / f"fold_{fold_idx}"
        fold_dir.mkdir(exist_ok=True)
        with (fold_dir / "train.txt").open("w") as f:
            for i in tr_idx: f.write(cases[i] + "\n")
        with (fold_dir / "val.txt").open("w") as f:
            for i in va_idx: f.write(cases[i] + "\n")
        print(f"fold_{fold_idx}: train={len(tr_idx)} val={len(va_idx)}")
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    main()
