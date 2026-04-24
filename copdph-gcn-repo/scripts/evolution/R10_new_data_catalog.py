"""R10 — Catalog the 100 new plain-scan nonPH cases (H:/424/*).

Sources:
  H:/424/copdnoph平扫性/               — 24 re-uploads of previously-placeholder cases
  H:/424新增copdnoph平扫性/             — 76 brand-new cases

Outputs:
  data/case_protocol_r10_additions.csv  — case_id,source,status (new|refill)
  outputs/r10/new_data_catalog.md       — human-readable summary
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
REFILL = Path("H:/424/copdnoph平扫性")
NEW = Path("H:/424新增copdnoph平扫性")
PROTO = ROOT / "data" / "case_protocol.csv"
OUT_CSV = ROOT / "data" / "case_protocol_r10_additions.csv"
OUT_MD = ROOT / "outputs" / "r10" / "new_data_catalog.md"
OUT_MD.parent.mkdir(parents=True, exist_ok=True)


def folder_to_case_id(folder_name: str) -> str | None:
    if folder_name.endswith(".rar") or folder_name.startswith("."):
        return None
    parts = folder_name.split("__")
    if len(parts) < 3:
        return None
    pinyin = re.sub(r"[^a-zA-Z]", "", parts[0]).lower()
    pid = re.sub(r"[^a-zA-Z0-9]", "", parts[1]).lower()
    date_scan = parts[2].strip()
    date_scan = re.sub(r"[^a-zA-Z0-9, ]", "", date_scan).lower()
    date_scan = date_scan.replace(",", "").replace(" ", "_")
    # Append trailing "_000" if not present (scan index)
    if not re.search(r"_\d+$", date_scan):
        date_scan += "_000"
    return f"nonph_{pinyin}_{pid}_{date_scan}"


def main() -> None:
    existing = set(row["case_id"] for row in csv.DictReader(PROTO.open(encoding="utf-8")))
    rows = []
    for src_root, src_tag in [(REFILL, "refill"), (NEW, "new")]:
        if not src_root.exists():
            continue
        for entry in sorted(src_root.iterdir()):
            cid = folder_to_case_id(entry.name)
            if cid is None:
                continue
            rows.append({
                "case_id": cid,
                "folder_name": entry.name,
                "source_tag": src_tag,
                "already_in_protocol_table": int(cid in existing),
            })

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    src_counts = Counter((r["source_tag"], r["already_in_protocol_table"]) for r in rows)
    lines = [
        "# R10 new-data catalog",
        "",
        f"Refill (previously placeholder) folder: `{REFILL}` → {sum(1 for r in rows if r['source_tag']=='refill')} cases",
        f"New-patient folder: `{NEW}` → {sum(1 for r in rows if r['source_tag']=='new')} cases",
        f"Total new entries: **{len(rows)}**",
        "",
        "## Cross-tab source × already-in-protocol-table",
        "",
        "| source_tag | already_in_protocol_table | count |",
        "|---|---|---|",
    ]
    for (tag, inp), n in sorted(src_counts.items()):
        lines.append(f"| {tag} | {inp} | {n} |")

    refill_matches = sum(
        1 for r in rows
        if r["source_tag"] == "refill" and r["already_in_protocol_table"] == 1
    )
    new_matches = sum(
        1 for r in rows
        if r["source_tag"] == "new" and r["already_in_protocol_table"] == 1
    )
    lines += [
        "",
        f"**Refill matching existing cohort**: {refill_matches}/24 — these replace the",
        "placeholder-vessel cases flagged in `project_v2_cache_missing_segmentations.md`",
        "(expected 27 such cases; matching 24 is plausible given pinyin variant spellings).",
        "",
        f"**New-patient matching existing cohort**: {new_matches}/76 — these should all",
        "be 0 if the new folder is truly disjoint.",
        "",
        "## Impact on cohort statistics (projected)",
        "",
        "- Current: 170 PH + 112 nonPH = 282 cases (55% contrast, 45% plain-scan).",
        "- After ingesting these 100 additions + ~24 refills that replace placeholders:",
        "  ~170 PH (unchanged) + ~210 nonPH = **~380 total**, 45% contrast / 55% plain-scan.",
        "",
        "**Pipeline required** (Round 10+):",
        "1. DCM → NIfTI per case (dcm2niix).",
        "2. Lung/artery/vein/airway segmentation — HiPaS-style unified model ideal.",
        "3. Build v2 cache with kimimaro (existing `_remote_build_v2_cache.py`).",
        "4. Rerun protocol-prediction tests — expected gain: tighter CIs on within-nonPH LR,",
        "   potentially enabling domain-adversarial training to reach ≤0.6 target.",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
