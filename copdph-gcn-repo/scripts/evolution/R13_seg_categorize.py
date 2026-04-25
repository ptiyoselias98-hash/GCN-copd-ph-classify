"""R13.2b — Categorize seg-quality findings into real failures vs false-positives.

Reads outputs/r13/seg_quality_report.json (raw) and emits:

  outputs/r13/seg_failures_real.{json,md}  — cases needing re-segmentation
  outputs/r13/seg_findings_summary.md      — top-level breakdown

Categories:
  - REAL_FAIL: any mask EMPTY (segmentation pipeline failed on that mask)
  - LUNG_ANOMALY: lung mask has unusual component count (>5 lobes, suspicious)
  - VESSEL_FRAGMENTED_ONLY: only artery/vein flagged >100 components
       (vasculature naturally fragmented — likely a false positive of the
       100-component threshold; tracked but NOT excluded)
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
INP = ROOT / "outputs" / "r13" / "seg_quality_report.json"
OUT = ROOT / "outputs" / "r13"


def main():
    d = json.loads(INP.read_text(encoding="utf-8"))
    real_fails = []
    lung_anom = []
    vessel_only = []

    for c in d["cases"]:
        if c["status"] != "QUALITY_ISSUE":
            continue
        bads = {n: m.get("status") for n, m in c["masks"].items()
                if m.get("status", "ok") not in ("ok", "?")}
        if any(s == "EMPTY" for s in bads.values()):
            real_fails.append({"case_id": c["case_id"], "group": c["group"],
                               "src_dir": c["src_dir"], "bads": bads})
        elif any(s == "LUNG_COMPONENT_ANOMALY" for s in bads.values()):
            lung_anom.append({"case_id": c["case_id"], "group": c["group"],
                              "src_dir": c["src_dir"], "bads": bads})
        elif all(n in ("artery", "vein") for n in bads):
            vessel_only.append({"case_id": c["case_id"], "group": c["group"],
                                "bads": bads})
        else:
            real_fails.append({"case_id": c["case_id"], "group": c["group"],
                               "src_dir": c["src_dir"], "bads": bads})

    # Write outputs
    (OUT / "seg_failures_real.json").write_text(
        json.dumps({"summary": {"n_real_fails": len(real_fails),
                                "n_lung_anomaly": len(lung_anom),
                                "n_vessel_fragmented_only": len(vessel_only)},
                    "real_fails": real_fails, "lung_anomaly": lung_anom},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# R13.2b — Segmentation-quality findings (categorized)",
          "",
          "Three-way categorisation of the 80 cases flagged by",
          "`R13_seg_quality_audit.py`:",
          "",
          f"- **{len(real_fails)} REAL FAILURES** — at least one mask empty",
          f"  (lung/airway/artery/vein). These cases must be re-segmented or",
          f"  excluded from any downstream analysis. Per-case list below.",
          "",
          f"- **{len(lung_anom)} LUNG-COMPONENT ANOMALY** — lung mask has",
          f"  unusual component count (expected 1-2 lobes, observed >5 or 0).",
          f"  Inspect visually before re-segmenting.",
          "",
          f"- **{len(vessel_only)} VESSEL-FRAGMENTED ONLY** — only artery/vein",
          f"  flagged with >100 connected components. Vasculature is naturally",
          f"  fragmented (many vessel branches), so this is **likely a false",
          f"  positive** of the 100-component threshold. The R13 audit threshold",
          f"  for vessel components should be raised (e.g., to 1000+) in the",
          f"  next iteration.",
          "",
          "## REAL FAILURES (re-segment or exclude)",
          "",
          "| group | case_id | failure pattern |",
          "|---|---|---|"]

    for r in sorted(real_fails, key=lambda x: x["case_id"]):
        empty_masks = [n for n, s in r["bads"].items() if s == "EMPTY"]
        pattern = f"all 4 EMPTY" if len(empty_masks) == 4 else f"{','.join(empty_masks)} EMPTY"
        md.append(f"| {r['group']} | `{r['case_id']}` | {pattern} |")

    md += ["", "## LUNG-COMPONENT ANOMALY",
           "",
           "| group | case_id | issue |",
           "|---|---|---|"]
    for r in lung_anom:
        md.append(f"| {r['group']} | `{r['case_id']}` | "
                  f"{r['bads'].get('lung', 'lung-anom')} |")

    md += ["", "## Implications",
           "",
           f"- The legacy 282-cohort effective denominator drops by {len(real_fails)}",
           "  cases for any analysis requiring all 4 mask types. Most failures are",
           "  in the `nonph_plain` stratum (pre-pipeline ingestion of plain-scan CT).",
           "- This invalidates earlier R1-R12 within-nonPH analyses that assumed",
           "  all 80 nonPH-plain were graphable. Effective n=80 may be closer to",
           f"  80 - (failures in nonph_plain) ≈ {80 - sum(1 for r in real_fails if r['group'] == 'nonph_plain')}.",
           "- For R14: re-segment failed cases via HiPaS-style unified pipeline",
           "  (`reference_hipas_paper.md`) before claiming protocol-invariance.",
           ""]

    (OUT / "seg_findings_summary.md").write_text("\n".join(md), encoding="utf-8")

    print(f"REAL FAILURES: {len(real_fails)}")
    print(f"LUNG ANOMALY:  {len(lung_anom)}")
    print(f"VESSEL ONLY (likely FP): {len(vessel_only)}")
    print(f"\nSaved {OUT}/seg_failures_real.json + seg_findings_summary.md")


if __name__ == "__main__":
    main()
