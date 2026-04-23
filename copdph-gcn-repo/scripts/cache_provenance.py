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
        "teasar_params",
        "dust_threshold",
        "mask_sentinel",
        "case_id",
        "label",
    ):
        if key in d:
            out[key] = d[key]
    out["top_level_keys"] = list(d.keys())
    return out


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
