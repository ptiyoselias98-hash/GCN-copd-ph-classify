"""R15.E — Extend labels + protocol manifest with the 100 new plain-scan nonPH cases.

Reads the dcm_conversion_log.json (R15.0) for the 100 case_ids, joins with
the legacy labels_expanded_282.csv + case_protocol.csv, and emits unified
labels_extended_382.csv + case_protocol_extended.csv where the 100 new
cases are tagged label=0, protocol=plain_scan.

Note: 22 of these 100 case_ids may already exist in the legacy 282 manifest
(refill of earlier placeholders) — those are detected and de-duplicated.

Outputs: data/labels_extended_382.csv  (max ~382 rows, possibly fewer
                                         after de-dup with refills)
        data/case_protocol_extended.csv
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
DATA = ROOT / "data"
DCM_LOG = ROOT / "outputs" / "r15" / "dcm_conversion_log.json"


def main():
    if not DCM_LOG.exists():
        raise SystemExit(f"missing {DCM_LOG}")
    log = json.loads(DCM_LOG.read_text(encoding="utf-8"))
    new_ids = [c["case_id"] for c in log["cases"]
               if c.get("status") == "ok" and c.get("case_id")]
    print(f"R15.E: {len(new_ids)} new ok cases")

    legacy_lab = pd.read_csv(DATA / "labels_expanded_282.csv")
    legacy_pro = pd.read_csv(DATA / "case_protocol.csv")
    print(f"  legacy: {len(legacy_lab)} labels, {len(legacy_pro)} protocols")

    new_lab = pd.DataFrame({"case_id": new_ids, "label": [0] * len(new_ids)})
    new_pro = pd.DataFrame({
        "case_id": new_ids,
        "label": [0] * len(new_ids),
        "protocol": ["plain_scan"] * len(new_ids),
    })

    overlap = set(new_ids) & set(legacy_lab["case_id"].tolist())
    print(f"  overlap with legacy 282: {len(overlap)}")

    # Outer-merge: keep legacy rows unchanged; add new ones not in legacy
    new_only = new_lab[~new_lab["case_id"].isin(legacy_lab["case_id"])]
    ext_lab = pd.concat([legacy_lab, new_only], ignore_index=True)
    new_only_p = new_pro[~new_pro["case_id"].isin(legacy_pro["case_id"])]
    ext_pro = pd.concat([legacy_pro, new_only_p], ignore_index=True)

    out_lab = DATA / "labels_extended_382.csv"
    out_pro = DATA / "case_protocol_extended.csv"
    ext_lab.to_csv(out_lab, index=False)
    ext_pro.to_csv(out_pro, index=False)
    print(f"  wrote {len(ext_lab)} → {out_lab}")
    print(f"  wrote {len(ext_pro)} → {out_pro}")
    print(f"  added {len(new_only)} new labels, {len(new_only_p)} new protocols")
    # Audit
    print(f"  ext_pro protocol breakdown:")
    print(ext_pro["protocol"].value_counts().to_string())
    print(f"  ext_lab label breakdown:")
    print(ext_lab["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
