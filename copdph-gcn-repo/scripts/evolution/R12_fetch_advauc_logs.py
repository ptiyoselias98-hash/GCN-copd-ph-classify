"""R12.3 — Fetch R11 GRL-fix per-epoch adversary AUC logs from remote and
parse into per-(lambda,seed) JSON artifacts.

Saves outputs/r11/embeddings/l{lam}_s{seed}/run.log (raw) plus
outputs/r11/embeddings/adv_auc_per_epoch.json (parsed summary).
Addresses R11 must_fix item #2: save per-epoch adversary AUC + run logs.

Usage:
    python scripts/evolution/R12_fetch_advauc_logs.py
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

REMOTE = "imss@10.60.147.117"
REMOTE_BASE = "/home/imss/cw/GCN copdnoph copdph/outputs"
ROOT = Path(__file__).parent.parent.parent
OUT_BASE = ROOT / "outputs" / "r11" / "embeddings"
SUMMARY_OUT = ROOT / "outputs" / "r11" / "adv_auc_per_epoch.json"

LAMBDAS = [0.0, 1.0, 5.0, 10.0]
SEEDS = [42, 1042, 2042]
ADV_RE = re.compile(r"\[adv\]\s+epoch=(\d+)\s+batch_auc_mean=([\d.]+)\s+n_batches=(\d+)")
FOLD_RE = re.compile(r"fold\s+(\d+)\s+thr=")


def fetch_log(lam: float, seed: int) -> Path | None:
    out_dir = OUT_BASE / f"l{lam}_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_log = out_dir / "run.log"
    remote = f"{REMOTE_BASE}/sprint6_arm_a_grlfix_l{lam}_s{seed}/run.log"
    cmd = ["scp", "-q", f"{REMOTE}:{remote}", str(out_log)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[skip] {lam}/{seed}: {r.stderr.strip()[:120]}")
        return None
    return out_log


def parse_log(p: Path) -> dict:
    fold_idx = 0
    fold_records: list[list[dict]] = [[]]
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        m = ADV_RE.search(line)
        if m:
            fold_records[-1].append({
                "epoch": int(m.group(1)),
                "batch_auc_mean": float(m.group(2)),
                "n_batches": int(m.group(3)),
            })
            continue
        if FOLD_RE.search(line):
            fold_idx += 1
            fold_records.append([])
    fold_records = [f for f in fold_records if f]
    return {
        "n_folds": len(fold_records),
        "folds": fold_records,
    }


def main():
    summary = {}
    for lam in LAMBDAS:
        for seed in SEEDS:
            log = fetch_log(lam, seed)
            if log is None:
                continue
            parsed = parse_log(log)
            key = f"l{lam}_s{seed}"
            summary[key] = parsed
            n_folds = parsed["n_folds"]
            first = len(parsed["folds"][0]) if parsed["folds"] else 0
            note = " (no adversary epochs — λ=0 baseline)" if lam == 0.0 and first == 0 else ""
            print(f"[ok] {key}: {n_folds} folds, first-fold {first} adv-epochs{note}")
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved summary: {SUMMARY_OUT}")
    print(f"  configs covered: {len(summary)} / {len(LAMBDAS)*len(SEEDS)}")


if __name__ == "__main__":
    main()
