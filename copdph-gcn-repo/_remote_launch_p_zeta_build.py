"""P-ζ prep: resume cache_tri build (CPU-only, 8 workers) for the 14 missing cases."""
import paramiko, time
HOST, PORT, USER, PASS = "10.60.147.117", 22, "imss", "imsslab"

REMOTE_SCRIPT = "/tmp/_p_zeta_build.sh"
SCRIPT = '''#!/usr/bin/env bash
set -uo pipefail
PROJ="/home/imss/cw/GCN copdnoph copdph"
OUT="${PROJ}/outputs/p_zeta_build_log"
FLAG="${OUT}/p_zeta_build_done.flag"
LOG="${OUT}/p_zeta_build.log"
mkdir -p "${OUT}"
rm -f "${FLAG}"

source /home/imss/miniconda3/etc/profile.d/conda.sh
conda activate pulmonary_bv5_py39

cd "${PROJ}"

echo "===== P-ζ build resume start $(date -Is) =====" > "${LOG}"
echo "pre-build file count: $(ls cache_tri 2>/dev/null | wc -l)" >> "${LOG}"

python -u build_cache_tristructure.py --workers 8 >> "${LOG}" 2>&1

RC=$?
echo "post-build file count: $(ls cache_tri 2>/dev/null | wc -l)" >> "${LOG}"
echo "===== P-ζ build rc=${RC} end $(date -Is) =====" >> "${LOG}"
echo "${RC}" > "${FLAG}"
'''

c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, PORT, USER, PASS, timeout=15, allow_agent=False, look_for_keys=False)
sftp = c.open_sftp()
with sftp.file(REMOTE_SCRIPT, "w") as f: f.write(SCRIPT)
sftp.close()

cmd = f"chmod +x {REMOTE_SCRIPT} && nohup bash {REMOTE_SCRIPT} > /tmp/_p_zeta_build_nohup.out 2>&1 &"
_, o, e = c.exec_command(cmd, timeout=30)
print("launch:", o.read().decode(), e.read().decode())

time.sleep(4)
_, o, _ = c.exec_command("pgrep -af build_cache_tristructure.py | head -20 || echo NONE", timeout=10)
print("proc:", o.read().decode())
c.close()
