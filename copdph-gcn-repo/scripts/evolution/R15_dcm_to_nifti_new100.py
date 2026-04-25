"""R15.0 — DCM → NIfTI conversion for the 100 NEW plain-scan nonPH cases.

Sources (per project_copdph_cohort_protocols memory):
  - H:\\4月24号-新增24个copdnoph平扫性\\（24个完整）copdnoph平扫性    (24 cases)
  - H:\\4月24号-新增76个copdnoph平扫性                                  (76 cases)

Picks the first phase subfolder (00000001) — same convention as the
existing convert_ct.py for the legacy 282-cohort. Skips quarantine
folders (有缺失 / 无效 / etc).

Output: E:\\桌面文件\\nii格式图\\nii-new100\\{case_id}\\ct.nii.gz
        outputs/r15/dcm_conversion_log.json

Adapts E:\\桌面文件\\nii格式图\\convert_ct.py (the existing reusable
converter on this machine).
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import SimpleITK as sitk

ROOT = Path(__file__).parent.parent.parent
OUT_LOG = ROOT / "outputs" / "r15"
OUT_LOG.mkdir(parents=True, exist_ok=True)

SOURCES = [
    (Path(r"H:\4月24号-新增24个copdnoph平扫性\（24个完整）copdnoph平扫性"), "nonph_"),
    (Path(r"H:\4月24号-新增76个copdnoph平扫性"), "nonph_"),
]
TARGET_ROOT = Path(r"E:\桌面文件\nii格式图\nii-new100")
QUARANTINE_MARKERS = ("有缺失", "废弃", "drop", "exclude", "invalid", "无效")


def slugify(name: str, prefix: str) -> str:
    s = name.lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if s and not s.startswith(prefix):
        s = prefix + s
    return s


def convert_dcm_to_nii(dicom_dir: Path, out_path: Path) -> dict:
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        return {"status": "no_series"}
    best_files, best_count = None, -1
    for sid in series_ids:
        files = reader.GetGDCMSeriesFileNames(str(dicom_dir), sid)
        if len(files) > best_count:
            best_files, best_count = files, len(files)
    reader.SetFileNames(best_files)
    image = reader.Execute()
    spacing = image.GetSpacing()
    size = image.GetSize()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False,
                                      dir=r"C:\Windows\Temp") as tmp:
        tmp_path = tmp.name
    try:
        sitk.WriteImage(image, tmp_path, useCompression=True)
        shutil.move(tmp_path, str(out_path))
    finally:
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()
    return {"status": "ok", "n_slices": best_count,
            "shape": list(size), "spacing": list(spacing)}


def main():
    log = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "sources": [str(s[0]) for s in SOURCES],
           "target_root": str(TARGET_ROOT),
           "cases": [], "summary": {}}
    n_total = n_ok = n_quarantine = n_skip = n_fail = 0
    for src_root, prefix in SOURCES:
        if not src_root.exists():
            log["cases"].append({"src": str(src_root), "status": "MISSING_SRC"})
            continue
        for subj in sorted(src_root.iterdir()):
            if not subj.is_dir():
                continue
            n_total += 1
            cid = slugify(subj.name, prefix)
            if any(m in subj.name for m in QUARANTINE_MARKERS) or not cid:
                log["cases"].append({"src": subj.name, "case_id": cid,
                                      "status": "QUARANTINE_SKIPPED"})
                n_quarantine += 1
                continue
            dicom_dir = subj / "00000001"
            if not dicom_dir.is_dir():
                log["cases"].append({"src": subj.name, "case_id": cid,
                                      "status": "NO_PHASE_00000001"})
                n_skip += 1
                continue
            target_dir = TARGET_ROOT / cid
            out_path = target_dir / "ct.nii.gz"
            if out_path.exists() and out_path.stat().st_size > 1000:
                log["cases"].append({"src": subj.name, "case_id": cid,
                                      "status": "EXISTS",
                                      "n_bytes": out_path.stat().st_size})
                n_ok += 1
                print(f"[exists] {cid}", flush=True)
                continue
            try:
                t0 = time.time()
                rec = convert_dcm_to_nii(dicom_dir, out_path)
                rec["src"] = subj.name; rec["case_id"] = cid
                rec["wall_seconds"] = round(time.time() - t0, 2)
                log["cases"].append(rec)
                if rec.get("status") == "ok":
                    n_ok += 1
                    print(f"[ok] {cid}: {rec['n_slices']} slices, "
                          f"{rec['wall_seconds']}s", flush=True)
                else:
                    n_fail += 1
                    print(f"[fail/{rec.get('status')}] {cid}", flush=True)
            except Exception as exc:
                log["cases"].append({"src": subj.name, "case_id": cid,
                                      "status": "EXCEPTION", "error": str(exc)})
                n_fail += 1
                print(f"[fail/exc] {cid}: {exc}", flush=True)

    log["summary"] = {"total": n_total, "ok": n_ok,
                      "quarantine": n_quarantine, "skip": n_skip, "fail": n_fail,
                      "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (OUT_LOG / "dcm_conversion_log.json").write_text(
        json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== total={n_total} ok={n_ok} quarantine={n_quarantine} "
          f"skip={n_skip} fail={n_fail} ===", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
