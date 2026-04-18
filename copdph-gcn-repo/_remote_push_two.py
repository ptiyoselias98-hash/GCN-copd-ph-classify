"""Push run_sprint2.py + hybrid_gcn.py and relaunch."""
from __future__ import annotations
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
import os
import paramiko

HOST, PORT, USER = "10.60.147.117", 22, "imss"
PASS = "imsslab"
REMOTE = "/home/imss/cw/GCN copdnoph copdph"
LOCAL = Path(r"C:\Users\cheng\Desktop\图卷积-肺小血管演化规律探索\copdph-gcn-repo")
FILES = ["run_sprint2.py", "hybrid_gcn.py"]
OUT = "outputs/sprint2_enhanced_v2"
ENV = "source /home/imss/miniconda3/etc/profile.d/conda.sh && conda activate pulmonary_bv5_py39"
LABELS = "/home/imss/cw/COPDnonPH COPD-PH /data/tables/labels.csv"
SPLITS = "/home/imss/cw/COPDnonPH COPD-PH /data/splits/folds"

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(HOST, PORT, USER, PASS, timeout=15, allow_agent=False, look_for_keys=False)

def run(cmd, t=30):
    _, o, e = cli.exec_command(cmd, timeout=t)
    return o.read().decode("utf-8", "replace"), e.read().decode("utf-8", "replace")

sftp = cli.open_sftp()
for f in FILES:
    sftp.put(str(LOCAL / f), f"{REMOTE}/{f}")
    print(f"[push] {f}")
sftp.close()

# truncate old run.log so we can diff cleanly
run(f"bash -lc \"> '{REMOTE}/{OUT}/run.log'\"")

launch = (
    f"cd '{REMOTE}' && mkdir -p '{OUT}' && "
    f"nohup python -u run_sprint2.py "
    f"  --cache_dir ./cache --radiomics ./data/copd_ph_radiomics.csv "
    f"  --labels '{LABELS}' --splits '{SPLITS}' "
    f"  --output_dir './{OUT}' "
    f"  --epochs 300 --batch_size 8 --lr 1e-3 "
    f"  < /dev/null > '{OUT}/run.log' 2>&1 & disown; echo launched"
)
# Fire-and-forget — don't wait for stdout (nohup keeps stdio open via bash wrapper).
try:
    cli.exec_command(f"bash -lc \"{ENV} && {launch}\"", timeout=5)
except Exception:
    pass
time.sleep(4)
o, _ = run("pgrep -fa 'python.*run_sprint2\\.py' || true")
print("[pgrep]", o.strip())
o, _ = run(f"tail -n 15 '{REMOTE}/{OUT}/run.log'")
print("[tail]\n" + o)
cli.close()
