"""R6.4 — Build per-case OOF table with case_id for the primary endpoint.

The patched run_sprint6_v2_probs.py saved ensemble_y_true and ensemble_y_score
as flat arrays in fold-iteration order. We reconstruct the case_id ordering
by reading the same data/splits_contrast_only/fold_*/val.txt that the
training run consumed.

Output: outputs/r6/primary_endpoint_oof.csv with columns:
  case_id, fold, label, p_arm_a, p_arm_c, protocol
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
SPLITS = ROOT / "data" / "splits_contrast_only"
ARM_A = ROOT / "outputs" / "r5" / "arm_a_contrast_only_probs.json"
ARM_C = ROOT / "outputs" / "r5" / "arm_c_contrast_only_probs.json"
PROTO = ROOT / "data" / "case_protocol.csv"
OUT = ROOT / "outputs" / "r6" / "primary_endpoint_oof.csv"


def fold_case_order(splits_dir: Path) -> list[tuple[int, str]]:
    out = []
    for k in range(1, 6):
        val_file = splits_dir / f"fold_{k}" / "val.txt"
        for c in val_file.read_text().splitlines():
            c = c.strip()
            if c:
                out.append((k, c))
    return out


CACHE_LIST = ROOT / "outputs" / "r5" / "cache_v2_tri_flat_list.txt"


def main() -> None:
    if not SPLITS.exists():
        print(f"WARN: {SPLITS} missing locally")
        return
    cached = set(c.strip() for c in CACHE_LIST.read_text().splitlines() if c.strip())
    raw = fold_case_order(SPLITS)
    fold_cids = [(k, c) for (k, c) in raw if c in cached]
    print(f"Total val case_ids: raw={len(raw)}, cached={len(fold_cids)}")

    arm_a = json.loads(ARM_A.read_text())["baseline"]["gcn_only"]
    arm_c = json.loads(ARM_C.read_text())["baseline"]["gcn_only"]
    yt_a = arm_a["ensemble_y_true"]
    ys_a = arm_a["ensemble_y_score"]
    yt_c = arm_c["ensemble_y_true"]
    ys_c = arm_c["ensemble_y_score"]
    assert len(fold_cids) == len(yt_a) == len(yt_c), \
        f"length mismatch: cids={len(fold_cids)} a={len(yt_a)} c={len(yt_c)}"
    assert yt_a == yt_c, "label order differs between arm_a and arm_c"

    proto = {r["case_id"]: r["protocol"] for r in csv.DictReader(PROTO.open(encoding="utf-8"))}

    rows = []
    for (k, cid), y, pa, pc in zip(fold_cids, yt_a, ys_a, ys_c):
        rows.append({
            "case_id": cid,
            "fold": k,
            "label": int(y),
            "protocol": proto.get(cid, "unknown"),
            "p_arm_a": float(pa),
            "p_arm_c": float(pc),
        })
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
