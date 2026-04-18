"""Build {patient_id: mPAP} JSON from xlsx for sprint5 aux regression.

Usage:
  python gen_mpap_lookup.py --xlsx ../copd-ph患者113例0331.xlsx \
                            --out ./data/mpap_lookup.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from run_hybrid import case_to_pinyin
except Exception:
    def case_to_pinyin(n: str) -> str:
        return "".join(ch for ch in str(n) if ord(ch) < 128)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--xlsx", required=True)
    p.add_argument("--out", default="./data/mpap_lookup.json")
    p.add_argument("--name_col", default="姓名")
    args = p.parse_args()

    df = pd.read_excel(args.xlsx, sheet_name="Sheet1")
    name_col = args.name_col if args.name_col in df.columns else df.columns[0]
    mpap = pd.to_numeric(df.get("mPAP", pd.Series([np.nan] * len(df))),
                         errors="coerce")

    out: dict[str, float | None] = {}
    for name, m in zip(df[name_col].astype(str), mpap):
        pid = case_to_pinyin(name)
        out[pid] = None if pd.isna(m) else float(m)

    dst = Path(args.out); dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"wrote {dst} ({len(out)} entries, "
          f"{sum(v is not None for v in out.values())} non-null mPAP)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
