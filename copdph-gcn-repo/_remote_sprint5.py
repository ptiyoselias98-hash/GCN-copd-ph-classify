"""Orchestrator for sprint5: push + launch on remote, with 3 arms.

Arms (serial, one nohup session):
  A: full       — node_drop=0.10  mpap_aux=0.10   (all 3 improvements)
  B: ndrop_only — node_drop=0.10  mpap_aux=0.0    (ablate mpap aux)
  C: mpap_only  — node_drop=0.0   mpap_aux=0.10   (ablate node drop)

All arms use the same mPAP-stratified splits JSON.

Each arm writes outputs/sprint5_<arm>/sprint5_results.json;
sentinel outputs/sprint5_done.flag is touched when all done.

Usage:
  python _remote_sprint5.py push      # push scripts + data
  python _remote_sprint5.py launch    # push + nohup launch (idempotent)
  python _remote_sprint5.py status    # tail logs + check flag
  python _remote_sprint5.py fetch     # sftp results back
"""
from __future__ import annotations
import io
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os
import paramiko

HOST, PORT, USER = "10.60.147.117", 22, "imss"
PASS = "imsslab"
REMOTE = "/home/imss/cw/GCN copdnoph copdph"
LOCAL_REPO = Path(r"C:\Users\cheng\Desktop\图卷积-肺小血管演化规律探索\copdph-gcn-repo")
LOCAL_OUT = Path(r"C:\Users\cheng\Desktop\图卷积-肺小血管演化规律探索\outputs")
ENV = "source /home/imss/miniconda3/etc/profile.d/conda.sh && conda activate pulmonary_bv5_py39"
LABELS = "/home/imss/cw/COPDnonPH COPD-PH /data/tables/labels.csv"

ARMS = [
    # (name, node_drop_p, mpap_aux_weight)
    ("full",       0.10, 0.10),
    ("ndrop_only", 0.10, 0.00),
    ("mpap_only",  0.00, 0.10),
]
SENTINEL = "outputs/sprint5_done.flag"
SPLITS_JSON_REMOTE = "data/splits_mpap_stratified.json"
MPAP_LOOKUP_REMOTE = "data/mpap_lookup.json"

PUSH_FILES = [
    "run_sprint5.py",
    "run_sprint3.py",
    "run_sprint2.py",
    "run_hybrid.py",
    "hybrid_gcn.py",
    "enhance_features.py",
    "gen_mpap_stratified_splits.py",
    "gen_mpap_lookup.py",
]
DATA_PUSH = [
    ("data/splits_mpap_stratified.json", SPLITS_JSON_REMOTE),
    ("data/mpap_lookup.json", MPAP_LOOKUP_REMOTE),
]


def _cli() -> paramiko.SSHClient:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, USER, PASS, timeout=15,
              allow_agent=False, look_for_keys=False)
    return c


def _run(c, cmd, t=60):
    _, o, e = c.exec_command(cmd, timeout=t)
    return o.read().decode("utf-8", "replace"), e.read().decode("utf-8", "replace")


def push(c):
    sftp = c.open_sftp()
    for f in PUSH_FILES:
        lp = LOCAL_REPO / f
        if not lp.exists():
            print(f"[push] SKIP {f} (missing locally)")
            continue
        sftp.put(str(lp), f"{REMOTE}/{f}")
        print(f"[push] {f}")
    for local_rel, remote_rel in DATA_PUSH:
        lp = LOCAL_REPO / local_rel
        if not lp.exists():
            print(f"[push-data] SKIP {local_rel} (generate locally first)")
            continue
        # Ensure remote dir exists
        remote_dir = "/".join(remote_rel.split("/")[:-1])
        _run(c, f"mkdir -p '{REMOTE}/{remote_dir}'")
        sftp.put(str(lp), f"{REMOTE}/{remote_rel}")
        print(f"[push-data] {local_rel}")
    sftp.close()


def launch(c):
    body = [f"cd '{REMOTE}'", "rm -f outputs/sprint5_done.flag"]
    for name, ndp, maw in ARMS:
        out = f"outputs/sprint5_{name}"
        body += [
            f"mkdir -p '{out}'",
            f"echo '=== arm={name} node_drop={ndp} mpap_aux={maw} ===' "
            f"> '{out}/run.log'",
            f"python -u run_sprint5.py "
            f"--cache_dir ./cache --radiomics ./data/copd_ph_radiomics.csv "
            f"--labels '{LABELS}' --splits_json '{SPLITS_JSON_REMOTE}' "
            f"--mpap_lookup '{MPAP_LOOKUP_REMOTE}' "
            f"--output_dir './{out}' "
            f"--epochs 300 --batch_size 8 --lr 1e-3 "
            f"--loss focal --globals_keep local4 "
            f"--node_drop_p {ndp} --mpap_aux_weight {maw} "
            f">> '{out}/run.log' 2>&1",
        ]
    body.append(f"touch '{SENTINEL}'")
    joined = " && ".join(body)
    launch_cmd = (
        f"nohup bash -lc \"{ENV} && {joined}\" "
        f"< /dev/null > '{REMOTE}/outputs/sprint5_launcher.log' 2>&1 "
        f"& disown; echo launched"
    )
    try:
        c.exec_command(launch_cmd, timeout=5)
    except Exception:
        pass
    time.sleep(4)
    o, _ = _run(c, "pgrep -fa 'python.*run_sprint5' || true")
    print("[pgrep]", o.strip() or "(none yet; may still be starting)")
    o, _ = _run(c, f"tail -n 8 '{REMOTE}/outputs/sprint5_launcher.log' "
                   f"2>/dev/null || true")
    print("[launcher.log]\n" + o)


def status(c):
    print("--- sentinel ---")
    o, _ = _run(c, f"ls -la '{REMOTE}/{SENTINEL}' 2>&1")
    print(o.strip())
    print("\n--- pgrep ---")
    o, _ = _run(c, "pgrep -fa 'python.*run_sprint5' || true")
    print(o.strip() or "(none)")
    print("\n--- nvidia-smi ---")
    o, _ = _run(c, "nvidia-smi --query-gpu=index,utilization.gpu,memory.used "
                   "--format=csv,noheader 2>/dev/null || true")
    print(o.strip())
    for name, _, _ in ARMS:
        out = f"outputs/sprint5_{name}"
        print(f"\n--- {name} tail ---")
        o, _ = _run(c, f"tail -n 8 '{REMOTE}/{out}/run.log' 2>&1 | tail -n 8")
        print(o)
        o, _ = _run(c, f"ls -la '{REMOTE}/{out}/sprint5_results.json' 2>&1")
        print(o.strip())


def is_done(c) -> bool:
    o, _ = _run(c, f"test -f '{REMOTE}/{SENTINEL}' && echo YES || echo NO")
    return "YES" in o


def fetch(c):
    sftp = c.open_sftp()
    for name, _, _ in ARMS:
        local = LOCAL_OUT / f"sprint5_{name}"
        local.mkdir(parents=True, exist_ok=True)
        for fn in ("sprint5_results.json", "run.log"):
            rp = f"{REMOTE}/outputs/sprint5_{name}/{fn}"
            try:
                sftp.get(rp, str(local / fn))
                print(f"OK {name}/{fn}  ({(local / fn).stat().st_size} B)")
            except Exception as e:
                print(f"MISS {name}/{fn}: {e}")
    try:
        sftp.get(f"{REMOTE}/outputs/sprint5_launcher.log",
                 str(LOCAL_OUT / "sprint5_launcher.log"))
    except Exception:
        pass
    sftp.close()


def main() -> int:
    action = sys.argv[1] if len(sys.argv) > 1 else "launch"
    c = _cli()
    try:
        if action == "push":
            push(c)
        elif action == "launch":
            push(c); launch(c)
        elif action == "status":
            status(c)
        elif action == "fetch":
            fetch(c)
        elif action == "is_done":
            print("DONE" if is_done(c) else "RUNNING")
        else:
            print(f"unknown action: {action}")
            return 1
    finally:
        c.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
