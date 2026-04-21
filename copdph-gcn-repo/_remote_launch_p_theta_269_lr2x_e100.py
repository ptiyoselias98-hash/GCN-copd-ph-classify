"""Launch p_theta_269_lr2x_e100 on idle GPU0 — half epochs (100) variant of
the lr=2e-3 n=269 cell, to get a fast estimate without waiting for the
slow-converging --epochs 200 original on GPU1.
"""
import paramiko, time
HOST, PORT, USER, PASS = "10.60.147.117", 22, "imss", "imsslab"

NAME = "p_theta_269_lr2x_e100"
SCRIPT = f'''#!/usr/bin/env bash
set -uo pipefail
PROJ="/home/imss/cw/GCN copdnoph copdph"
OUT="${{PROJ}}/outputs/{NAME}"
LOG="${{OUT}}/{NAME}.log"
FLAG="${{OUT}}/{NAME}_done.flag"
mkdir -p "${{OUT}}"
rm -f "${{FLAG}}"

source /home/imss/miniconda3/etc/profile.d/conda.sh
conda activate pulmonary_bv5_py39
cd "${{PROJ}}/tri_structure"
export CUDA_VISIBLE_DEVICES=0

echo "===== {NAME} start $(date -Is) (GPU 0, lr=2e-3, epochs=100, SOLO) =====" > "${{LOG}}"
python -u tri_structure_pipeline.py \\
    --cache_dir "${{PROJ}}/cache_tri_converted" \\
    --labels "${{PROJ}}/data/labels_expanded_282.csv" \\
    --output_dir "${{OUT}}" \\
    --epochs 100 --repeats 3 \\
    --lr 2e-3 \\
    --mpap_aux \\
    >> "${{LOG}}" 2>&1
RC=$?
echo "===== {NAME} rc=$RC end $(date -Is) =====" >> "${{LOG}}"
echo "$RC" > "${{FLAG}}"
'''

c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, PORT, USER, PASS, timeout=15, allow_agent=False, look_for_keys=False)
sftp = c.open_sftp()
rpath = f"/tmp/_launch_{NAME}.sh"
with sftp.file(rpath, "w") as f: f.write(SCRIPT)
sftp.close()

cmd = (f"chmod +x {rpath} && "
       f"setsid bash -c 'bash {rpath} > /tmp/_{NAME}_nohup.out 2>&1' "
       f"< /dev/null > /dev/null 2>&1 &")
c.exec_command(cmd, timeout=5)
print(f"launched {NAME} on GPU0")

time.sleep(5)
_, o, _ = c.exec_command(
    "pgrep -af tri_structure_pipeline.py; echo ---; "
    "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader",
    timeout=10)
print(o.read().decode())
c.close()
