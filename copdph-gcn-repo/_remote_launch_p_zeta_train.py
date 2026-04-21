"""P-ζ: tri_structure training on expanded n=282 cache (269 usable)."""
import paramiko, time
HOST, PORT, USER, PASS = "10.60.147.117", 22, "imss", "imsslab"

REMOTE_SCRIPT = "/tmp/_p_zeta_train.sh"
SCRIPT = '''#!/usr/bin/env bash
set -uo pipefail
PROJ="/home/imss/cw/GCN copdnoph copdph"
OUT="${PROJ}/outputs/p_zeta_tri_282"
FLAG="${OUT}/p_zeta_done.flag"
LOG="${OUT}/p_zeta.log"
mkdir -p "${OUT}"
rm -f "${FLAG}"

source /home/imss/miniconda3/etc/profile.d/conda.sh
conda activate pulmonary_bv5_py39

cd "${PROJ}/tri_structure"

export CUDA_VISIBLE_DEVICES=1

echo "===== P-ζ tri_phase1 on expanded 282 start $(date -Is) =====" > "${LOG}"
echo "GPU=${CUDA_VISIBLE_DEVICES}" >> "${LOG}"
echo "cache=${PROJ}/cache_tri" >> "${LOG}"
echo "cache file count: $(ls \"${PROJ}/cache_tri\" 2>/dev/null | wc -l)" >> "${LOG}"

# tri_structure_pipeline expects tri format + _tri.pkl suffix by default. Let's
# probe: tri_structure/cache_tri has _tri.pkl (106). Root cache_tri has .pkl (269).
# Check if pipeline supports naming variant — if not, symlink.

python -u tri_structure_pipeline.py \\
    --cache_dir "${PROJ}/cache_tri_converted" \\
    --labels "${PROJ}/data/labels_expanded_282.csv" \\
    --output_dir "${OUT}" \\
    --epochs 200 --repeats 3 \\
    --mpap_aux \\
    >> "${LOG}" 2>&1

RC=$?
echo "===== P-ζ rc=${RC} end $(date -Is) =====" >> "${LOG}"
echo "${RC}" > "${FLAG}"
'''

c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, PORT, USER, PASS, timeout=15, allow_agent=False, look_for_keys=False)
sftp = c.open_sftp()
with sftp.file(REMOTE_SCRIPT, "w") as f: f.write(SCRIPT)
sftp.close()
cmd = f"chmod +x {REMOTE_SCRIPT} && nohup bash {REMOTE_SCRIPT} > /tmp/_p_zeta_train_nohup.out 2>&1 &"
_, o, e = c.exec_command(cmd, timeout=30)
print("launch:", o.read().decode(), e.read().decode())
time.sleep(5)
_, o, _ = c.exec_command("pgrep -af tri_structure_pipeline.py | head -5 || echo NONE", timeout=10)
print("procs:", o.read().decode())
c.close()
