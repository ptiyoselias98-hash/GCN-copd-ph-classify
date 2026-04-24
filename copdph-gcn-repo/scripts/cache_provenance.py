"""Read cache pkls and print builder provenance (version, git SHA, kimimaro).

Usage:
    python scripts/cache_provenance.py cache_v2_tri_flat/<case>.pkl ...
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path


def inspect(path: Path) -> dict:
    with path.open("rb") as f:
        d = pickle.load(f)
    out = {"file": path.name}
    for key in (
        "builder_version",
        "git_sha",
        "kimimaro_version",
        "mask_sentinel",
        "patient_id",
        "label",
    ):
        if key in d:
            out[key] = d[key]
    # Per-structure TEASAR params live under qc[struct]
    qc = d.get("qc", {})
    for struct in ("artery", "vein", "airway"):
        s = qc.get(struct, {})
        if isinstance(s, dict) and "teasar_params" in s:
            out[f"{struct}_teasar_params"] = s["teasar_params"]
    out["top_level_keys"] = list(d.keys())
    return out


def verify_expected(path: Path, expected: dict) -> tuple[bool, list[str]]:
    """Compare pkl provenance fields against expected dict; return (ok, mismatches)."""
    info = inspect(path)
    issues = []
    for k, v in expected.items():
        got = info.get(k)
        if got != v:
            issues.append(f"{k}: expected {v!r}, got {got!r}")
    return (len(issues) == 0), issues


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print("Usage: python scripts/cache_provenance.py <pkl> [<pkl> ...]")
        sys.exit(1)
    for p in paths:
        info = inspect(p)
        for k, v in info.items():
            if isinstance(v, list) and len(v) > 10:
                v = v[:10] + ["…"]
            print(f"  {k}: {v}")
        print()


if __name__ == "__main__":
    main()
