"""Find the right Python interpreter on remote server."""
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
    "which python python3 python3.10 python3.11 python3.12 2>&1",
    "python --version; python3 --version",
    "ls /opt/conda/envs 2>/dev/null; ls ~/miniconda3/envs 2>/dev/null; ls ~/anaconda3/envs 2>/dev/null",
    "cat '/home/imss/cw/GCN copdnoph copdph/REMOTE_RUN.md' 2>/dev/null",
    "cat '/home/imss/cw/GCN copdnoph copdph/requirements.txt' 2>/dev/null",
    "ls /home/imss/cw/",
    "ps -ef | grep -E 'run_(sprint2|real|cv_full|hybrid)' | grep -v grep",
    "nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv 2>&1 | head",
]:
    print(f"\n$ {cmd}")
    o, e = run(cmd)
    if o.strip():
        print(o.rstrip())
    if e.strip():
        print("STDERR:", e.rstrip())
cli.close()
