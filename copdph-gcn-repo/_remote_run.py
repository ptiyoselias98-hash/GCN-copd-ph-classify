"""Compile-check + launch 5-fold sprint2 in background; print PID + log path."""
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
OUT = f"{REPO}/outputs/sprint2_top10"
LOG = f"{REPO}/outputs/sprint2_top10.log"


def run(cli, cmd, timeout=180):
    _, o, e = cli.exec_command(cmd, timeout=timeout)
    out = o.read().decode("utf-8", "replace")
    err = e.read().decode("utf-8", "replace")
    rc = o.channel.recv_exit_status()
    return rc, out, err


def main() -> int:
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, 22, USER, PASS, timeout=15,
                allow_agent=False, look_for_keys=False)

    rc, o, e = run(cli,
        f"cd '{REPO}' && '{PY}' -m py_compile enhance_features.py "
        f"run_sprint2.py visualize.py && echo COMPILE_OK")
    print("[compile]", o.strip() or e.strip())
    if "COMPILE_OK" not in o:
        return 1

    rc, o, _ = run(cli,
        f"cd '{REPO}' && '{PY}' -c \"from enhance_features import "
        f"EXPECTED_OUT_DIM, BASELINE_IN_DIM; print(BASELINE_IN_DIM, EXPECTED_OUT_DIM)\"")
    print("[dim]", o.strip())

    rc, o, _ = run(cli,
        "ps -ef | grep -E 'run_sprint2|run_real|run_cv_full' | grep -v grep || true")
    if o.strip():
        print("[abort] existing jobs:\n" + o); return 2

    _, o, _ = run(cli, "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader")
    print("[gpu]", o.strip())

    cmd = (
        f"cd '{REPO}' && mkdir -p '{OUT}' && rm -f '{LOG}' && "
        f"CUDA_VISIBLE_DEVICES=0 nohup '{PY}' -u run_sprint2.py "
        f"--cache_dir '{CACHE}' "
        f"--labels \"{LABELS}\" --splits \"{SPLITS}\" "
        f"--radiomics '{RAD}' --output_dir '{OUT}' "
        f"--epochs 300 --batch_size 8 --lr 1e-3 --wd 5e-4 --seed 42 "
        f"--modes radiomics_only,gcn_only,hybrid "
        f"--feature_sets baseline,enhanced "
        f"> '{LOG}' 2>&1 & echo PID=$!; disown"
    )
    rc, o, e = run(cli, cmd)
    print("[launch]", o.strip(), e.strip())

    import time; time.sleep(4)
    _, o, _ = run(cli, f"head -n 20 '{LOG}' 2>/dev/null; echo '---'; "
                        f"pgrep -f run_sprint2.py | head -3")
    print("[initial log]\n" + o)
    cli.close()
    print(f"\nLOG={LOG}\nOUT={OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
