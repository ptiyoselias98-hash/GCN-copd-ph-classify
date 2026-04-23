"""W2 audit: verify fold splits are patient-disjoint.

Case-id format: `<label>_<pinyin>_<id>_<weekday>_<month>_<day>_<year>_<scan_idx>`.
Patient key = (label, pinyin, id). Same patient may have multiple scans
(different date and/or scan_idx). If the same patient_key appears in both
train.txt and val.txt of any fold, we have label leakage.

Outputs `outputs/_patient_leakage_audit.json` and a markdown summary.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
SPLITS = ROOT / "data" / "splits_expanded_282"
LABELS = ROOT / "data" / "labels_expanded_282.csv"
OUT_JSON = ROOT / "outputs" / "_patient_leakage_audit.json"
OUT_MD = ROOT / "outputs" / "_patient_leakage_audit.md"

WEEKDAYS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}


def parse_case(case_id: str) -> tuple[str, ...] | None:
    """Patient key = everything up to (but not including) the weekday token.

    Case format: `<label>_<pinyin_parts...>_<id>_<weekday>_<month>_<day>_<year>_<scan_idx>`.
    Multi-word pinyin names (e.g. `haliba_ahengbieke`) are captured correctly
    because we terminate the patient key at the first weekday token.
    """
    parts = case_id.split("_")
    if not parts or parts[0] not in {"nonph", "ph"}:
        return None
    for i, tok in enumerate(parts):
        if tok in WEEKDAYS:
            if i < 2:
                return None
            return tuple(parts[:i])
    return None


def load_split(fold: int, which: str) -> list[str]:
    path = SPLITS / f"fold_{fold}" / f"{which}.txt"
    return [c.strip() for c in path.read_text().splitlines() if c.strip()]


def load_labels() -> dict[str, int]:
    out: dict[str, int] = {}
    with LABELS.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["case_id"]] = int(row["label"])
    return out


def audit() -> dict:
    labels = load_labels()
    all_cases = sorted(labels)
    patient_of_case: dict[str, tuple[str, str, str]] = {}
    unparsed: list[str] = []
    for c in all_cases:
        pk = parse_case(c)
        if pk is None:
            unparsed.append(c)
        else:
            patient_of_case[c] = pk

    # Patient-level summary
    cases_per_patient: dict[tuple, list[str]] = defaultdict(list)
    for c, pk in patient_of_case.items():
        cases_per_patient[pk].append(c)
    multi_scan_patients = {
        pk: cs for pk, cs in cases_per_patient.items() if len(cs) > 1
    }

    # Per-fold leakage check
    fold_reports = []
    total_leaks = 0
    for fold in range(1, 6):
        train = load_split(fold, "train")
        val = load_split(fold, "val")
        train_pks = {patient_of_case[c] for c in train if c in patient_of_case}
        val_pks = {patient_of_case[c] for c in val if c in patient_of_case}
        leaked_pks = train_pks & val_pks
        leaked_cases = {
            pk: {
                "train_cases": [c for c in train if patient_of_case.get(c) == pk],
                "val_cases": [c for c in val if patient_of_case.get(c) == pk],
                "label": labels[next(iter([c for c in train + val if patient_of_case.get(c) == pk]))],
            }
            for pk in leaked_pks
        }
        total_leaks += len(leaked_pks)
        fold_reports.append(
            {
                "fold": fold,
                "train_cases": len(train),
                "val_cases": len(val),
                "train_patients": len(train_pks),
                "val_patients": len(val_pks),
                "leaked_patients": len(leaked_pks),
                "leaked": [
                    {
                        "patient": "_".join(pk),
                        "label": info["label"],
                        "train_cases": info["train_cases"],
                        "val_cases": info["val_cases"],
                    }
                    for pk, info in leaked_cases.items()
                ],
            }
        )

    summary = {
        "n_cases": len(all_cases),
        "n_cases_parsed": len(patient_of_case),
        "n_unparsed": len(unparsed),
        "unparsed_examples": unparsed[:10],
        "n_patients": len(cases_per_patient),
        "n_multi_scan_patients": len(multi_scan_patients),
        "scan_counts": Counter(len(v) for v in cases_per_patient.values()),
        "total_leaked_patient_occurrences": total_leaks,
        "fold_reports": fold_reports,
    }
    return summary


def main() -> None:
    summary = audit()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=list))
    lines = [
        "# Patient-level fold leakage audit (W2)",
        "",
        f"- Total cases: **{summary['n_cases']}**",
        f"- Case-IDs parsed: **{summary['n_cases_parsed']}** (unparsed: {summary['n_unparsed']})",
        f"- Unique patients: **{summary['n_patients']}**",
        f"- Multi-scan patients: **{summary['n_multi_scan_patients']}**",
        f"- Scan-count histogram: {dict(summary['scan_counts'])}",
        f"- **Total leaked patient occurrences across 5 folds: {summary['total_leaked_patient_occurrences']}**",
        "",
        "## Per-fold",
        "",
        "| Fold | #train | #val | train_pts | val_pts | leaked |",
        "|---|---|---|---|---|---|",
    ]
    for r in summary["fold_reports"]:
        lines.append(
            f"| {r['fold']} | {r['train_cases']} | {r['val_cases']} | "
            f"{r['train_patients']} | {r['val_patients']} | {r['leaked_patients']} |"
        )
    if summary["total_leaked_patient_occurrences"]:
        lines += ["", "## Leaked patient details", ""]
        for r in summary["fold_reports"]:
            if not r["leaked"]:
                continue
            lines.append(f"### fold_{r['fold']}")
            for leak in r["leaked"]:
                lines.append(
                    f"- **{leak['patient']}** (label={leak['label']})\n"
                    f"  - train: {leak['train_cases']}\n"
                    f"  - val:   {leak['val_cases']}"
                )
    OUT_MD.write_text("\n".join(lines))
    print(OUT_MD.read_text())


if __name__ == "__main__":
    main()
