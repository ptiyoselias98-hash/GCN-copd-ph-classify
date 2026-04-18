"""Push a single file to remote and relaunch training."""
from __future__ import annotations
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
import os
import paramiko

HOST, PORT, USER = "10.60.147.117", 22, "imss"
PASS = os.environ.get("IMSS_SSH_PASSWORD")
if not PASS:
    raise RuntimeError("Set IMSS_SSH_PASSWORD before running this script.")
REMOTE = "/home/imss/cw/GCN copdnoph copdph"
LOCAL_REPO = Path(r"C:\Users\cheng\Desktop\图卷积-肺小血管演化规律探索\copdph-gcn-repo")
FILE = "run_sprint2.py"
OUT = "outputs/sprint2_enhanced_v2"
ENV = "source /home/imss/miniconda3/etc/profile.d/conda.sh && conda activate pulmonary_bv5_py39"
LABELS = "/home/imss/cw/COPDnonPH COPD-PH /data/tables/labels.csv"
SPLITS = "/home/imss/cw/COPDnonPH COPD-PH /data/splits/folds"

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(HOST, PORT, USER, PASS, timeout=15, allow_agent=False, look_for_keys=False)

def run(cmd, t=60):
    _, o, e = cli.exec_command(cmd, timeout=t)
    return o.read().decode("utf-8", "replace"), e.read().decode("utf-8", "replace")

# 1) push
sftp = cli.open_sftp()
sftp.put(str(LOCAL_REPO / FILE), f"{REMOTE}/{FILE}")
sftp.close()
print(f"[push] {FILE}")

# 2) ensure no prior run still alive
o, _ = run("pgrep -fa 'python.*run_sprint2\\.py' || true")
print("[pgrep]", o.strip() or "(none)")

# 3) launch
launch = (
    f"cd '{REMOTE}' && mkdir -p '{OUT}' && "
    f"nohup python -u run_sprint2.py "
    f"  --cache_dir ./cache "
    f"  --radiomics ./data/copd_ph_radiomics.csv "
    f"  --labels '{LABELS}' --splits '{SPLITS}' "
    f"  --output_dir './{OUT}' "
    f"  --epochs 300 --batch_size 8 --lr 1e-3 "
    f"  < /dev/null > '{OUT}/run.log' 2>&1 & PID=$!; disown; echo $PID"
)
o, e = run(f"bash -lc \"{ENV} && {launch}\"", t=60)
print("[launch]", o.strip())
if e.strip():
    print("STDERR:", e[-500:])

# 4) quick tail
import time
time.sleep(3)
o, _ = run(f"tail -n 10 '{REMOTE}/{OUT}/run.log' 2>&1")
print("[tail]\n" + o)
cli.close()
