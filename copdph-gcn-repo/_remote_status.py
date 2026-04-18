"""Tail run.log + show pid info for sprint2 training."""
from __future__ import annotations
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import os
import paramiko

HOST, PORT, USER = "10.60.147.117", 22, "imss"
PASS = "imsslab"
REMOTE_REPO = "/home/imss/cw/GCN copdnoph copdph"
OUT = "outputs/sprint2_enhanced_v2"

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(HOST, PORT, USER, PASS, timeout=15, allow_agent=False, look_for_keys=False)

def run(cmd):
    _, o, e = cli.exec_command(cmd, timeout=60)
    return o.read().decode("utf-8", "replace"), e.read().decode("utf-8", "replace")

for cmd in [
    "pgrep -fa 'python.*run_sprint2\\.py' || true",
    f"ls -la '{REMOTE_REPO}/{OUT}/' 2>&1 | head",
    f"wc -l '{REMOTE_REPO}/{OUT}/run.log' 2>/dev/null",
    f"tail -n 40 '{REMOTE_REPO}/{OUT}/run.log' 2>&1",
    "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || true",
]:
    o, e = run(cmd)
    print(f"\n$ {cmd}\n{o}", end="")
    if e.strip():
        print("ERR:", e[-300:])
cli.close()
