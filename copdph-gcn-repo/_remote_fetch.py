"""Fetch sprint2 v2 results (JSON + log) to local outputs subfolder."""
from __future__ import annotations
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
import os
import paramiko

HOST = "10.60.147.117"; USER = "imss"; PASS = "imsslab"
REPO_R = "/home/imss/cw/GCN copdnoph copdph"
OUT_R = f"{REPO_R}/outputs/sprint2_enhanced_v2"
LOCAL = Path(r"C:\Users\cheng\Desktop\图卷积-肺小血管演化规律探索\outputs\sprint2_v2")
LOCAL.mkdir(parents=True, exist_ok=True)

FILES = [
    f"{OUT_R}/sprint2_results.json",
    f"{OUT_R}/run.log",
]

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(HOST, 22, USER, PASS, timeout=15, allow_agent=False, look_for_keys=False)
sftp = cli.open_sftp()
for rp in FILES:
    name = Path(rp).name
    lp = LOCAL / name
    try:
        sftp.get(rp, str(lp))
        print(f"OK {name}  ({lp.stat().st_size} B)")
    except Exception as e:
        print(f"FAIL {rp}: {e}")
sftp.close(); cli.close()
print(f"\nLocal dir: {LOCAL}")
