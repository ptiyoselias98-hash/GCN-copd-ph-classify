"""P-γ: re-run tri_phase1 on GPU0 to validate ~0.88 AUC baseline reproducible."""
import paramiko, time
HOST, PORT, USER, PASS = "10.60.147.117", 22, "imss", "imsslab"

REMOTE_SCRIPT = "/tmp/_p_gamma_launch.sh"
SCRIPT = '''#!/usr/bin/env bash
# P-γ: tri_phase1 validation rerun on GPU0.
set -uo pipefail
PROJ="/home/imss/cw/GCN copdnoph copdph"
OUT="${PROJ}/outputs/p_gamma_tri_phase1_rerun"
FLAG="${OUT}/p_gamma_done.flag"
LOG="${OUT}/p_gamma.log"
mkdir -p "${OUT}"
rm -f "${FLAG}"

source /home/imss/miniconda3/etc/profile.d/conda.sh
conda activate pulmonary_bv5_py39

cd "${PROJ}/tri_structure"

export CUDA_VISIBLE_DEVICES=0

echo "===== P-γ tri_phase1 rerun start $(date -Is) =====" > "${LOG}"
echo "GPU=${CUDA_VISIBLE_DEVICES}" >> "${LOG}"
nvidia-smi --query-gpu=index,name,memory.used --format=csv >> "${LOG}"

python -u tri_structure_pipeline.py \\
    --cache_dir "${PROJ}/tri_structure/cache_tri" \\
    --labels "${PROJ}/data/labels_gold.csv" \\
    --mpap "${PROJ}/data/mpap_lookup_gold.json" \\
    --output_dir "${OUT}" \\
    --epochs 200 --repeats 3 \\
    --mpap_aux \\
    >> "${LOG}" 2>&1

RC=$?
echo "===== P-γ rc=${RC} end $(date -Is) =====" >> "${LOG}"
echo "${RC}" > "${FLAG}"
'''

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, PORT, USER, PASS, timeout=15, allow_agent=False, look_for_keys=False)
sftp = c.open_sftp()
with sftp.file(REMOTE_SCRIPT, "w") as f:
    f.write(SCRIPT)
sftp.close()

# launch nohup in background
cmd = f"chmod +x {REMOTE_SCRIPT} && nohup bash {REMOTE_SCRIPT} > /tmp/_p_gamma_nohup.out 2>&1 &"
_, o, e = c.exec_command(cmd, timeout=30)
print("launch stdout:", o.read().decode("utf-8","replace"))
print("launch stderr:", e.read().decode("utf-8","replace"))

# quick sanity check
time.sleep(3)
_, o, _ = c.exec_command("pgrep -af tri_structure_pipeline.py || echo NONE", timeout=10)
print("proc check:", o.read().decode("utf-8","replace"))
c.close()
