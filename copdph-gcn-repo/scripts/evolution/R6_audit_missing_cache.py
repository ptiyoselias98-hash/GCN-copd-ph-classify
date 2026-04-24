"""R6.2 — Audit which cases are missing from cache_v2_tri_flat.

Inputs:
  - data/case_protocol.csv (282 cases with label + protocol)
  - outputs/r5/cache_v2_tri_flat_list.txt (244 case_ids that have a pkl)

Output:
  outputs/r6/missing_cache_audit.{md,csv}

Cross-tab missing cases by (label, protocol) and probable cause
(placeholder vessels, etc.) per project memory.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
PROTO = ROOT / "data" / "case_protocol.csv"
CACHE_LIST = ROOT / "outputs" / "r5" / "cache_v2_tri_flat_list.txt"
LUNG_V2 = ROOT / "outputs" / "lung_features_v2.csv"
OUT_DIR = ROOT / "outputs" / "r6"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_MD = OUT_DIR / "missing_cache_audit.md"
OUT_CSV = OUT_DIR / "missing_cache_audit.csv"


def main() -> None:
    cached = set(c.strip() for c in CACHE_LIST.read_text().splitlines() if c.strip())
    proto_rows = list(csv.DictReader(PROTO.open(encoding="utf-8")))
    lung_rows = list(csv.DictReader(LUNG_V2.open(encoding="utf-8")))
    placeholder_status = {
        r["case_id"]: (
            r.get("artery_placeholder", ""), r.get("vein_placeholder", ""),
            r.get("airway_placeholder", ""), r.get("error", ""),
        )
        for r in lung_rows
    }

    out_rows = []
    for r in proto_rows:
        cid = r["case_id"]
        in_cache = cid in cached
        ap, vp, wp, err = placeholder_status.get(cid, ("", "", "", "lung_v2_missing"))
        out_rows.append({
            "case_id": cid,
            "label": r["label"],
            "protocol": r["protocol"],
            "in_cache_v2_tri_flat": int(in_cache),
            "artery_placeholder": ap,
            "vein_placeholder": vp,
            "airway_placeholder": wp,
            "lung_v2_error": err,
        })

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    # Cross-tab
    from collections import Counter
    in_tab = Counter()
    out_tab = Counter()
    for r in out_rows:
        key = (r["label"], r["protocol"])
        if r["in_cache_v2_tri_flat"]:
            in_tab[key] += 1
        else:
            out_tab[key] += 1

    lines = [
        "# R6.2 — Cache_v2_tri_flat coverage audit",
        "",
        f"Total: {len(out_rows)} cases in protocol table.",
        f"In cache: **{sum(in_tab.values())}**, missing: **{sum(out_tab.values())}**.",
        "",
        "| label | protocol | in cache | missing |",
        "|---|---|---|---|",
    ]
    for k in sorted(set(in_tab) | set(out_tab)):
        lines.append(f"| {k[0]} | {k[1]} | {in_tab.get(k, 0)} | {out_tab.get(k, 0)} |")
    lines += ["", "## Missing cases by reason", ""]
    reasons = Counter()
    for r in out_rows:
        if r["in_cache_v2_tri_flat"]:
            continue
        if r["lung_v2_error"] and r["lung_v2_error"] != "":
            reasons["lung_v2_error"] += 1
        elif r["artery_placeholder"] == "1" or r["vein_placeholder"] == "1":
            reasons["placeholder_vessel"] += 1
        else:
            reasons["other_unaccounted"] += 1
    for k, v in reasons.most_common():
        lines.append(f"- **{k}**: {v}")
    lines += [
        "",
        "## Reading",
        "",
        "Per project memory `project_v2_cache_missing_segmentations.md`: 27 nonPH",
        "have placeholder vessel masks (upstream segmentation failed on plain-scan",
        "CT) and ~7 PH have absent vessel files. The R6.2 audit confirms which",
        "cases the v2 cache builder dropped vs which it kept (with degraded inputs).",
        "",
        "If the missing 39 cases are all label-correlated (mostly nonPH), then the",
        "243-case sample is biased toward PH (163/243=67%) vs the 282-case truth",
        "(170/282=60%). Round 6 will rebuild a small subset of placeholder cases",
        "with degraded-graph handling to test exclusion sensitivity at the GCN level.",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
