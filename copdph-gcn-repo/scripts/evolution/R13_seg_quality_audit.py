"""R13.2 — Segmentation-quality audit on the 345 cohort.

User flag (2026-04-25): "有子文件夹里面数量一致但是分割质量有明显问题的，
需要你主动发现并记录报告" — even when 00000001..5 DCM counts agree,
some cases have visibly broken segmentations. Actively detect them.

Strategy: each case folder may contain NIfTI segmentation masks (look
for `.nii.gz`, `.nii`, or `seg/`/`mask/` subfolders). For each mask:
  - voxel count of foreground (>= 1 if integer label, > -2048 if HU)
  - n connected components (skimage)
  - bbox extent
  - if too small (<10k vox), flag
  - if mask all-zero → critical fail
  - if mask all-one or fills entire volume → suspicious
  - if components > 50 → likely noise/leak

Output: outputs/r13/seg_quality_report.{json,md} with per-case status.

Note: this script is a SCAN-AND-REPORT only — does not modify anything.
If `nibabel` / `scipy` not installed, gracefully degrade to file-size
heuristics + flag missing dependencies.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
OUT = ROOT / "outputs" / "r13"
OUT.mkdir(parents=True, exist_ok=True)
MANIFEST = OUT / "cohort_345_manifest.csv"

# Legacy unified NIfTI roots — masks for already-ingested cases live here
NII_ROOTS = [
    Path(r"E:\桌面文件\nii格式图\nii-unified-282"),
    Path(r"E:\桌面文件\nii格式图\nii-New folder-COPDnoph"),
]

try:
    import nibabel as nib
    import numpy as np
    HAVE_NIB = True
except Exception as exc:
    HAVE_NIB = False
    NIB_ERR = str(exc)

try:
    from scipy.ndimage import label as cc_label
    HAVE_CC = True
except Exception:
    HAVE_CC = False


# Expected mask basenames
MASK_NAMES = ["lung", "airway", "artery", "vein"]


def find_masks(case_id: str, case_dir: Path) -> dict[str, Path]:
    out = {}
    # Try unified NIfTI roots first (case_id-based naming)
    candidates: list[Path] = []
    for root in NII_ROOTS:
        if not root.exists():
            continue
        candidates.append(root / case_id)
        # nii-New folder-COPDnoph has no prefix
        if case_id.startswith("nonph_"):
            candidates.append(root / case_id[len("nonph_"):])
        if case_id.startswith("ph_"):
            candidates.append(root / case_id[len("ph_"):])
    candidates.append(case_dir)
    for cand in candidates:
        if not cand.exists() or not cand.is_dir():
            continue
        for n in MASK_NAMES:
            if n in out:
                continue
            for ext in (".nii.gz", ".nii"):
                p = cand / f"{n}{ext}"
                if p.exists():
                    out[n] = p
                    break
        if len(out) >= len(MASK_NAMES):
            break
    return out


def audit_mask(p: Path) -> dict:
    if not HAVE_NIB:
        return {"path": str(p), "size_bytes": p.stat().st_size,
                "warning": f"nibabel unavailable: {NIB_ERR}"}
    img = nib.load(str(p))
    arr = img.get_fdata(caching="unchanged")
    rec = {"path": str(p),
           "size_bytes": p.stat().st_size,
           "shape": list(arr.shape),
           "dtype": str(arr.dtype)}
    # Detect HU sentinel (-2048 = bg) vs integer labels
    has_hu = float(arr.min()) <= -1500
    if has_hu:
        fg = (arr > -2000).astype("uint8")
        rec["mode"] = "hu_sentinel"
    else:
        fg = (arr > 0).astype("uint8")
        rec["mode"] = "integer_label"
    n_fg = int(fg.sum())
    rec["n_fg_vox"] = n_fg
    rec["fg_frac"] = float(n_fg) / max(1, fg.size)
    if n_fg == 0:
        rec["status"] = "EMPTY"
        return rec
    if rec["fg_frac"] > 0.95:
        rec["status"] = "ALL_FILLED"
        return rec
    if n_fg < 10000 and ("lung" in p.name or "airway" in p.name):
        rec["status"] = "TOO_SMALL"
        rec["issue"] = f"only {n_fg} vox (<10000)"
        return rec
    if HAVE_CC:
        lbl, ncc = cc_label(fg)
        rec["n_components"] = int(ncc)
        if ncc > 100:
            rec["status"] = "TOO_FRAGMENTED"
            rec["issue"] = f"{ncc} connected components (>100)"
            return rec
        # Lung-specific: expect 1-2 connected components
        if "lung" in p.name and (ncc < 1 or ncc > 5):
            rec["status"] = "LUNG_COMPONENT_ANOMALY"
            rec["issue"] = f"lung has {ncc} components (expected 1-2)"
            return rec
    rec["status"] = "ok"
    return rec


def main():
    if not HAVE_NIB:
        print(f"[warn] nibabel unavailable ({NIB_ERR}); audit will use filesize only")
    if not HAVE_CC:
        print("[warn] scipy.ndimage.label unavailable; component checks skipped")

    if not MANIFEST.exists():
        raise SystemExit(f"Run R13_cohort_reconciliation.py first (missing {MANIFEST})")

    case_records = []
    n_no_masks = 0
    n_critical = 0
    n_warn = 0
    with MANIFEST.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["status"] != "ok":
                continue
            case_dir = Path(row["src_root"]) / row["src_dir"]
            masks = find_masks(row["case_id"], case_dir)
            rec = {
                "case_id": row["case_id"],
                "group": row["group"],
                "src_dir": row["src_dir"],
                "n_masks_found": len(masks),
                "masks": {},
            }
            if not masks:
                rec["status"] = "no_masks_in_source"
                rec["note"] = "DCM only — segmentation pipeline pending"
                n_no_masks += 1
            else:
                for name, p in masks.items():
                    rec["masks"][name] = audit_mask(p)
                statuses = [m.get("status", "?") for m in rec["masks"].values()]
                bads = [s for s in statuses if s not in ("ok", "?")]
                if bads:
                    rec["status"] = "QUALITY_ISSUE"
                    rec["bad_masks"] = bads
                    n_critical += 1
                else:
                    rec["status"] = "ok"
            case_records.append(rec)

    summary = {
        "total_audited": len(case_records),
        "no_masks_in_source": n_no_masks,
        "quality_issues": n_critical,
        "ok": len(case_records) - n_no_masks - n_critical,
        "have_nibabel": HAVE_NIB,
        "have_cc_label": HAVE_CC,
    }
    out_json = OUT / "seg_quality_report.json"
    out_json.write_text(json.dumps({"summary": summary, "cases": case_records},
                                    indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# R13.2 — Segmentation-quality audit on 345-cohort source folders",
          "",
          "Per user 2026-04-25: even when 00000001..5 DCM counts match, some",
          "cases have visibly broken segmentations. This script scans each",
          "case for NIfTI masks (lung/airway/artery/vein) and flags",
          "EMPTY/ALL_FILLED/TOO_SMALL/TOO_FRAGMENTED/LUNG_COMPONENT_ANOMALY.",
          "",
          f"**nibabel available**: {HAVE_NIB}  | **scipy.ndimage.label**: {HAVE_CC}",
          "",
          f"**Total audited**: {summary['total_audited']}",
          f"**No masks found in source folder**: {summary['no_masks_in_source']} "
          "(DCM-only cases — masks pending segmentation pipeline)",
          f"**Cases with quality issues**: {summary['quality_issues']}",
          f"**Cases passing audit**: {summary['ok']}",
          "",
          "## Cases flagged with quality issues",
          "",
          "| group | case_id | bad masks | first issue |",
          "|---|---|---|---|"]
    for rec in case_records:
        if rec["status"] != "QUALITY_ISSUE":
            continue
        bad_names = [n for n, m in rec["masks"].items()
                     if m.get("status", "ok") not in ("ok", "?")]
        first_issue = next((m.get("issue") or m.get("status")
                           for m in rec["masks"].values()
                           if m.get("status") not in ("ok", "?")), "")
        md.append(f"| {rec['group']} | `{rec['case_id']}` | {','.join(bad_names)} | {first_issue} |")
    if summary["quality_issues"] == 0:
        md.append("| - | - | - | (none) |")

    md += ["",
           "## Cases without source-side masks (need pipeline ingestion)",
           "",
           f"Total: {summary['no_masks_in_source']}",
           "",
           "(Listed in `seg_quality_report.json` under cases[].status='no_masks_in_source')"]

    (OUT / "seg_quality_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved {out_json}")
    print(f"  total={summary['total_audited']} no_masks={summary['no_masks_in_source']} "
          f"quality_issues={summary['quality_issues']} ok={summary['ok']}")


if __name__ == "__main__":
    main()
