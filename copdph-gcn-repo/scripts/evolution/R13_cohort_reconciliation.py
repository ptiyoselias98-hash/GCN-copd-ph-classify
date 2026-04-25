"""R13.1 — 345-cohort case_id-level reconciliation + DCM-count audit.

Walks the five authoritative source folders on H:, enumerates each
case subfolder, audits 00000001..00000005 DCM child directory counts
for slice-count consistency, and writes:

  outputs/r13/cohort_345_manifest.csv     # case_id, group, src, status, n_dcm_per_phase
  outputs/r13/cohort_345_summary.md       # human-readable counts + diff vs legacy 282
  outputs/r13/cohort_audit_failures.md    # cases excluded with reasons

Per user 2026-04-25 the user already manually excluded cases with
mismatched 00000001..5 DCM counts in COPDPH_seg (originally ~170+ → 160
PH) and in New folder-COPDNOPH (→ 58 plain-scan). This script
re-confirms by the same audit to surface any remaining mismatches
plus segmentation-quality flags.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
OUT = ROOT / "outputs" / "r13"
OUT.mkdir(parents=True, exist_ok=True)

# Authoritative source folders (Chinese paths quoted directly)
SOURCES = [
    ("ph_contrast",   r"H:\官方数据data\COPDPH_seg（160例增强性CT）",          "PH contrast (post-prune)"),
    ("nonph_contrast",r"H:\官方数据data\COPDnonPH_seg（27例增强性CT）",         "nonPH contrast"),
    ("nonph_plain",   r"H:\官方数据data\New folder-COPDNOPH 58例平扫性",         "nonPH plain-scan (post-prune)"),
    ("nonph_refill",  r"H:\4月24号-新增24个copdnoph平扫性\（24个完整）copdnoph平扫性", "nonPH plain refill (24 complete)"),
    ("nonph_new",     r"H:\4月24号-新增76个copdnoph平扫性",                    "nonPH plain new"),
]

# Try also without full-width parentheses since FS may store ASCII
ALT = {
    r"H:\官方数据data\COPDPH_seg（160例增强性CT）": [r"H:\官方数据data\COPDPH_seg(160例增强性CT)",
                                                       r"H:\官方数据data\COPDPH_seg"],
    r"H:\官方数据data\COPDnonPH_seg（27例增强性CT）": [r"H:\官方数据data\COPDnonPH_seg(27例增强性CT)",
                                                          r"H:\官方数据data\COPDnonPH_seg"],
}


def find_existing(p: str) -> Path | None:
    cand = Path(p)
    if cand.exists():
        return cand
    for alt in ALT.get(p, []):
        c = Path(alt)
        if c.exists():
            return c
    # If parent exists, try fuzzy match on dirs that start with the prefix
    parent = cand.parent
    if parent.exists():
        prefix = cand.name.split("（")[0].split("(")[0]
        for child in parent.iterdir():
            if child.is_dir() and child.name.startswith(prefix):
                return child
    return None


def slugify(name: str, group: str) -> str:
    s = name.lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        return ""
    label = "ph" if group == "ph_contrast" else "nonph"
    if not s.startswith(label + "_"):
        s = f"{label}_{s}"
    return s


def audit_case(case_dir: Path) -> dict:
    rec = {"path": str(case_dir), "subfolders": {}, "status": "ok", "issue": ""}
    for k in range(1, 6):
        sub = case_dir / f"0000000{k}"
        if not sub.exists():
            rec["subfolders"][f"0000000{k}"] = -1
            continue
        try:
            n = sum(1 for _ in sub.iterdir())
        except OSError:
            n = -2
        rec["subfolders"][f"0000000{k}"] = n
    counts = [v for v in rec["subfolders"].values() if v >= 0]
    if len(counts) < 5:
        rec["status"] = "missing_phase"
        rec["issue"] = f"only {len(counts)}/5 phase subfolders exist"
    elif len(set(counts)) > 1:
        rec["status"] = "dcm_count_mismatch"
        rec["issue"] = f"DCM counts {rec['subfolders']} differ across phases"
    elif counts and counts[0] < 50:
        rec["status"] = "low_slice_count"
        rec["issue"] = f"only {counts[0]} slices/phase (suspicious for full chest CT)"
    return rec


def main():
    rows = []
    failures = []
    counts = {grp: 0 for grp, _, _ in SOURCES}
    failures_count = {grp: 0 for grp, _, _ in SOURCES}

    for grp, raw, desc in SOURCES:
        p = find_existing(raw)
        if p is None:
            failures.append({"src": raw, "reason": "PATH_NOT_FOUND"})
            print(f"[MISS] {grp}: {raw}")
            continue
        print(f"[OK]   {grp}: {p}")
        for child in sorted(p.iterdir()):
            if not child.is_dir():
                continue
            # Skip nested helper folders (e.g. "（24个完整）..." parent)
            if grp == "nonph_refill" and child.name.startswith("(") or child.name.startswith("（"):
                continue
            rec = audit_case(child)
            # Skip user-quarantine folders like 有缺失病例 / 有缺失数据
            quarantine_markers = ("有缺失", "废弃", "drop", "exclude", "invalid", "无效")
            if any(m in child.name for m in quarantine_markers):
                failures_count[grp] += 1
                failures.append({"case_id": "(quarantine_folder)", "group": grp,
                                 "src": child.name, "issue": "user-quarantine subfolder (skipped)"})
                continue
            case_id = slugify(child.name, grp)
            if not case_id:
                continue
            row = {
                "case_id": case_id,
                "group": grp,
                "src_dir": child.name,
                "src_root": str(p),
                "status": rec["status"],
                "issue": rec["issue"],
                "n_dcm_per_phase": json.dumps(rec["subfolders"], ensure_ascii=False),
            }
            rows.append(row)
            if rec["status"] == "ok":
                counts[grp] += 1
            else:
                failures_count[grp] += 1
                failures.append({"case_id": case_id, "group": grp,
                                 "src": child.name, "issue": rec["issue"]})

    # Write manifest CSV
    manifest = OUT / "cohort_345_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "group", "src_dir", "src_root",
                                          "status", "issue", "n_dcm_per_phase"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nManifest: {manifest} ({len(rows)} rows)")

    # Compare against legacy 282
    legacy = ROOT / "data" / "labels_expanded_282.csv"
    legacy_ids: set[str] = set()
    if legacy.exists():
        import csv as _csv
        with legacy.open(encoding="utf-8") as f:
            r = _csv.DictReader(f)
            for row in r:
                legacy_ids.add(row["case_id"].strip())

    new_ids = {r["case_id"] for r in rows if r["status"] == "ok"}
    only_new = new_ids - legacy_ids
    only_legacy = legacy_ids - new_ids
    common = new_ids & legacy_ids

    summary_md = [
        "# R13.1 — 345-cohort reconciliation + DCM-count audit",
        "",
        "Source folders (post-user-prune):",
        "",
        "| group | source | n_ok | n_failed |",
        "|---|---|---|---|",
    ]
    for grp, raw, desc in SOURCES:
        summary_md.append(f"| `{grp}` | {desc} | {counts.get(grp,0)} | {failures_count.get(grp,0)} |")
    summary_md += [
        "",
        f"**Total OK cases**: {len(new_ids)} | **Total failed**: "
        f"{sum(failures_count.values())} (see `cohort_audit_failures.md`)",
        "",
        "## Diff vs legacy 282-cohort (`data/labels_expanded_282.csv`)",
        "",
        f"- legacy_ids: **{len(legacy_ids)}**",
        f"- 345_ids (status=ok): **{len(new_ids)}**",
        f"- common: **{len(common)}**",
        f"- only in legacy (10-case PH overcount candidates): **{len(only_legacy)}**",
        f"- only in 345 (newly ingestible): **{len(only_new)}**",
        "",
        "Cases only in legacy (likely duplicates/re-enrollments to be dropped):",
        "",
    ]
    summary_md += [f"- `{c}`" for c in sorted(only_legacy)[:50]]
    if len(only_legacy) > 50:
        summary_md.append(f"- ... and {len(only_legacy) - 50} more")

    summary_md += [
        "",
        "Cases only in 345 (queued for ingestion):",
        "",
    ]
    summary_md += [f"- `{c}`" for c in sorted(only_new)[:50]]
    if len(only_new) > 50:
        summary_md.append(f"- ... and {len(only_new) - 50} more")

    (OUT / "cohort_345_summary.md").write_text("\n".join(summary_md), encoding="utf-8")

    fail_md = ["# R13.1 — Cohort audit failures (DCM-count mismatches)", ""]
    if not failures:
        fail_md.append("None — all subfolders pass the 00000001..5 DCM-count audit.")
    else:
        fail_md += ["| group | case_id | src_dir | issue |", "|---|---|---|---|"]
        for f in failures:
            if "case_id" in f:
                fail_md.append(f"| {f['group']} | `{f['case_id']}` | `{f['src']}` | {f['issue']} |")
            else:
                fail_md.append(f"| - | (path) | `{f['src']}` | {f['reason']} |")
    (OUT / "cohort_audit_failures.md").write_text("\n".join(fail_md), encoding="utf-8")

    diff = {
        "n_ok_345": len(new_ids),
        "n_legacy_282": len(legacy_ids),
        "n_common": len(common),
        "n_only_legacy": len(only_legacy),
        "n_only_345": len(only_new),
        "only_legacy_examples": sorted(only_legacy)[:20],
        "only_new_examples": sorted(only_new)[:20],
        "counts_per_group": counts,
        "failures_per_group": failures_count,
    }
    (OUT / "cohort_345_diff.json").write_text(json.dumps(diff, indent=2, ensure_ascii=False),
                                                encoding="utf-8")
    print(f"\nDiff vs legacy: 345_ok={len(new_ids)} legacy={len(legacy_ids)} "
          f"common={len(common)} only_legacy={len(only_legacy)} only_new={len(only_new)}")
    for grp, raw, _ in SOURCES:
        print(f"  {grp}: ok={counts.get(grp,0)} failed={failures_count.get(grp,0)}")


if __name__ == "__main__":
    main()
