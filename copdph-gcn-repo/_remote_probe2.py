from __future__ import annotations
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import os
import paramiko

HOST, PORT, USER = "10.60.147.117", 22, "imss"
PASS = "imsslab"
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(HOST, PORT, USER, PASS, timeout=15,
            allow_agent=False, look_for_keys=False)

def run(cmd):
    _, o, e = cli.exec_command(cmd, timeout=60)
    return o.read().decode("utf-8", "replace"), e.read().decode("utf-8", "replace")

for cmd in [
    "ls /home/imss/cw",
    "find /home/imss/cw -maxdepth 5 -type f -iname 'labels*.csv' 2>/dev/null",
    "find /home/imss/cw -maxdepth 6 -type d -iname 'folds*' 2>/dev/null",
    "find /home/imss/cw -maxdepth 6 -type d -iname 'splits*' 2>/dev/null",
    "ls '/home/imss/cw/GCN copdnoph copdph/data/' 2>&1 | head",
    "ls '/home/imss/cw/GCN copdnoph copdph/outputs/' 2>&1 | head -30",
    "grep -l 'labels.csv' '/home/imss/cw/GCN copdnoph copdph/outputs/' -r 2>/dev/null | head",
]:
    o, e = run(cmd)
    print(f"\n$ {cmd}\n{o}")
    if e.strip():
        print("ERR:", e[-500:])
cli.close()
