"""Push compute_pa_ao_from_masks.py to server, run it in background,
then sftp the resulting CSV back to local data/.

Usage:
  python _remote_pa_ao.py run    # push + nohup run + wait + fetch
  python _remote_pa_ao.py fetch  # just fetch the CSV if already computed
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
NII_ROOT = "/home/imss/cw/COPDnonPH COPD-PH /data/nii"
LOCAL_REPO = Path(r"C:\Users\cheng\Desktop\图卷积-肺小血管演化规律探索\copdph-gcn-repo")
ENV = "source /home/imss/miniconda3/etc/profile.d/conda.sh && conda activate pulmonary_bv5_py39"

REMOTE_CSV = "data/pa_ao_measurements.csv"
REMOTE_LOG = "outputs/pa_ao.log"
REMOTE_DONE = "outputs/pa_ao_done.flag"


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
    sftp.put(str(LOCAL_REPO / "compute_pa_ao_from_masks.py"),
             f"{REMOTE}/compute_pa_ao_from_masks.py")
    sftp.close()
    print("[push] compute_pa_ao_from_masks.py")


def launch(c):
    _run(c, f"cd '{REMOTE}' && mkdir -p data outputs && rm -f '{REMOTE_DONE}'")
    cmd = (
        f"cd '{REMOTE}' && "
        f"nohup bash -lc \"{ENV} && python -u compute_pa_ao_from_masks.py "
        f"--nii_root '{NII_ROOT}' --out '{REMOTE_CSV}' "
        f"&& touch '{REMOTE_DONE}'\" "
        f"< /dev/null > '{REMOTE_LOG}' 2>&1 & disown; echo launched"
    )
    try:
        c.exec_command(cmd, timeout=5)
    except Exception:
        pass
    time.sleep(3)
    o, _ = _run(c, f"tail -n 6 '{REMOTE}/{REMOTE_LOG}' 2>/dev/null || true")
    print("[log]\n" + o)


def wait_done(c, max_polls: int = 120, interval: int = 30):
    for i in range(max_polls):
        o, _ = _run(c, f"test -f '{REMOTE}/{REMOTE_DONE}' && echo YES || echo NO")
        if "YES" in o:
            print(f"[done] after {i * interval}s")
            return True
        o2, _ = _run(c, f"tail -n 2 '{REMOTE}/{REMOTE_LOG}' 2>/dev/null || true")
        print(f"[poll {i + 1}/{max_polls}] {o2.strip()[-120:]}")
        time.sleep(interval)
    return False


def fetch(c):
    sftp = c.open_sftp()
    local = LOCAL_REPO / "data" / "pa_ao_measurements.csv"
    local.parent.mkdir(parents=True, exist_ok=True)
    try:
        sftp.get(f"{REMOTE}/{REMOTE_CSV}", str(local))
        print(f"[fetch] {local} ({local.stat().st_size} B)")
    except Exception as e:
        print(f"MISS: {e}")
    try:
        sftp.get(f"{REMOTE}/{REMOTE_LOG}",
                 str(LOCAL_REPO / "data" / "pa_ao.log"))
    except Exception:
        pass
    sftp.close()


def main() -> int:
    action = sys.argv[1] if len(sys.argv) > 1 else "run"
    c = _cli()
    try:
        if action == "run":
            push(c); launch(c)
            if wait_done(c):
                fetch(c)
            else:
                print("TIMEOUT — rerun `python _remote_pa_ao.py fetch` later")
                return 1
        elif action == "push":
            push(c)
        elif action == "launch":
            launch(c)
        elif action == "fetch":
            fetch(c)
        elif action == "status":
            o, _ = _run(c, f"ls -la '{REMOTE}/{REMOTE_DONE}' '{REMOTE}/{REMOTE_CSV}' "
                            f"2>&1; tail -n 8 '{REMOTE}/{REMOTE_LOG}' 2>/dev/null")
            print(o)
        else:
            print(f"unknown: {action}")
            return 1
    finally:
        c.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
