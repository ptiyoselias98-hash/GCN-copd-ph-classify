"""Thin remote exec helper: prints stdout/err. Usage: python _remote_exec.py '<cmd>'"""
import os
import sys
import paramiko
HOST, PORT, USER = "10.60.147.117", 22, "imss"
PASS = "imsslab"
cmd = sys.argv[1]
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, PORT, USER, PASS, timeout=15, allow_agent=False, look_for_keys=False)
_, o, e = c.exec_command(cmd, timeout=120)
print(o.read().decode("utf-8", "replace"))
err = e.read().decode("utf-8", "replace")
if err.strip():
    print("ERR:", err)
c.close()
