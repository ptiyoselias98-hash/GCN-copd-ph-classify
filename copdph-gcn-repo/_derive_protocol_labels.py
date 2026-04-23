"""Derive per-case protocol (contrast-enhanced vs plain-scan) from original DCM folders.

Folder conventions (user-provided, 2026-04-23):
  H:/官方数据data/COPDnonPH_seg/           → 27 contrast-enhanced nonPH
  H:/官方数据data/COPDPH_seg/              → 170 contrast-enhanced PH (plus .rar dupes)
  H:/官方数据data/New folder-COPD（PH概率小）/ → 85 plain-scan nonPH

DCM folders are CamelCase like `BaoXiaoPing__9001193765__Thursday, January 2, 2020_000`.
Case ids are lowercase `nonph_baoxiaoping_9001193765_thursday_january_2_2020_000`.

We match by (pinyin_lower, id_lower) — i.e. the first two `__`-separated tokens,
lowercased and id stripped of non-alphanumeric padding.

Output `data/case_protocol.csv` with columns `case_id,protocol` where
protocol = 1 for contrast, 0 for plain-scan.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

DCM_ROOTS = {
    "contrast": [
        Path("H:/官方数据data/COPDnonPH_seg"),
        Path("H:/官方数据data/COPDPH_seg"),
    ],
    "plain_scan": [
        Path("H:/官方数据data/New folder-COPD（PH概率小）"),
    ],
}
LABELS = Path(__file__).parent / "data" / "labels_expanded_282.csv"
OUT = Path(__file__).parent / "data" / "case_protocol.csv"


def folder_key(folder_name: str) -> tuple[str, str] | None:
    """Extract (pinyin, id) from DCM folder name like `BaoXiaoPing__9001193765__...`."""
    if folder_name.endswith(".rar"):
        return None
    parts = folder_name.split("__")
    if len(parts) < 2:
        return None
    pinyin = re.sub(r"[^a-zA-Z]", "", parts[0]).lower()
    pid = re.sub(r"[^a-zA-Z0-9]", "", parts[1]).lower()
    return (pinyin, pid)


def case_key(case_id: str) -> tuple[str, str] | None:
    """Extract (pinyin, id) from case_id — first 2 tokens after label prefix."""
    toks = case_id.split("_")
    if len(toks) < 3:
        return None
    # Multi-pinyin handling (haliba_ahengbieke) → we join tokens until an all-digit/alphanum id-like token.
    weekday_idx = next(
        (
            i
            for i, t in enumerate(toks)
            if t
            in {
                "monday", "tuesday", "wednesday", "thursday",
                "friday", "saturday", "sunday",
            }
        ),
        -1,
    )
    if weekday_idx < 3:
        return None
    # ID is the token immediately before weekday
    pid = toks[weekday_idx - 1]
    pinyin = "".join(toks[1 : weekday_idx - 1])
    return (pinyin, pid)


def main() -> None:
    # Build DCM folder → protocol mapping
    dcm_map: dict[tuple[str, str], str] = {}
    for protocol, dirs in DCM_ROOTS.items():
        for d in dirs:
            if not d.exists():
                print(f"WARN: {d} does not exist")
                continue
            for entry in d.iterdir():
                k = folder_key(entry.name)
                if k is None:
                    continue
                if k in dcm_map and dcm_map[k] != protocol:
                    print(f"WARN: {k} appears in both protocols, keeping first")
                    continue
                dcm_map.setdefault(k, protocol)
    print(f"DCM keys indexed: {len(dcm_map)} "
          f"(contrast={sum(1 for v in dcm_map.values() if v=='contrast')}, "
          f"plain={sum(1 for v in dcm_map.values() if v=='plain_scan')})")

    # Match case_ids
    with LABELS.open() as f:
        reader = csv.DictReader(f)
        cases = [(r["case_id"], int(r["label"])) for r in reader]
    rows = []
    unmatched = []
    for cid, lbl in cases:
        k = case_key(cid)
        protocol = dcm_map.get(k) if k else None
        if protocol is None:
            unmatched.append(cid)
            protocol = "contrast" if lbl == 1 else "unknown"
        rows.append({"case_id": cid, "label": lbl, "protocol": protocol})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "label", "protocol"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {OUT} ({len(rows)} rows)")

    from collections import Counter
    crosstab: dict[tuple[int, str], int] = Counter()
    for r in rows:
        crosstab[(r["label"], r["protocol"])] += 1
    print("label × protocol cross-tab:")
    for (l, p), n in sorted(crosstab.items()):
        print(f"  label={l} protocol={p}: {n}")
    if unmatched:
        print(f"Unmatched ({len(unmatched)}, fallback applied):")
        for c in unmatched[:10]:
            print(f"  {c}")


if __name__ == "__main__":
    main()
