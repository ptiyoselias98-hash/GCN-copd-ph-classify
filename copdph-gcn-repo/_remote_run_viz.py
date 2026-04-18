"""Launch visualize.py on remote (uses 25-D enhanced features + Top-8 boxplots)."""
from __future__ import annotations

import sys
import os
import paramiko

HOST = "10.60.147.117"; USER = "imss"; PASS = os.environ.get("IMSS_SSH_PASSWORD")
if not PASS:
    raise RuntimeError("Set IMSS_SSH_PASSWORD before running this script.")
PY = "/home/imss/miniconda3/envs/pulmonary_bv5_py39/bin/python"
REPO = "/home/imss/cw/GCN copdnoph copdph"
LABELS = "/home/imss/cw/COPDnonPH COPD-PH /data/tables/labels.csv"
SPLITS = "/home/imss/cw/COPDnonPH COPD-PH /data/splits/folds"
CACHE = f"{REPO}/cache"
RAD = f"{REPO}/data/copd_ph_radiomics.csv"
OUT = f"{REPO}/outputs/viz_top10"
LOG = f"{REPO}/outputs/viz_top10.log"


def run(cli, cmd, timeout=180):
    _, o, e = cli.exec_command(cmd, timeout=timeout)
    return o.read().decode("utf-8", "replace"), e.read().decode("utf-8", "replace"), o.channel.recv_exit_status()


def main() -> int:
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, 22, USER, PASS, timeout=15, allow_agent=False, look_for_keys=False)

    # check no viz running
    o, _, _ = run(cli, "pgrep -af visualize.py | grep -v grep || true")
    if o.strip():
        print("[abort] viz already running:\n" + o); return 2

    # inspect visualize.py CLI args
    o, _, _ = run(cli, f"grep -E 'add_argument' '{REPO}/visualize.py'")
    print("[viz cli args]\n" + o)

    cmd = (
        f"cd '{REPO}' && mkdir -p '{OUT}' && rm -f '{LOG}' && "
        f"CUDA_VISIBLE_DEVICES=0 nohup '{PY}' -u visualize.py "
        f"--cache_dir '{CACHE}' "
        f"--labels \"{LABELS}\" --splits \"{SPLITS}\" "
        f"--radiomics '{RAD}' --output_dir '{OUT}' "
        f"--epochs 120 "
        f"> '{LOG}' 2>&1 & echo PID=$!; disown"
    )
    o, e, _ = run(cli, cmd)
    print("[launch]", o.strip(), e.strip())

    import time; time.sleep(4)
    o, _, _ = run(cli, f"head -n 30 '{LOG}' 2>/dev/null; echo '---PGREP---'; pgrep -af visualize.py")
    print("[initial]\n" + o)

    print(f"\nLOG={LOG}\nOUT={OUT}")
    cli.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
