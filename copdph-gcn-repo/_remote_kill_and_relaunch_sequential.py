"""Kill 2 LR-sweep jobs that share GPUs with kept jobs, then deploy a new
sequential watcher that re-launches them solo after the kept jobs finish.

Plan:
  Kill: p_theta_106_lr2x  (GPU0)   — wasted ~7 min
        p_theta_269_lr2x  (GPU1)   — wasted ~5 min
  Keep running: p_theta_106_lrhalf (GPU0), p_theta_269_lrhalf (GPU1)
  After each lrhalf done.flag appears → solo-launch the matching lr2x on
  the same GPU.
"""
import paramiko, time
HOST, PORT, USER, PASS = "10.60.147.117", 22, "imss", "imsslab"

KILL_AND_DEPLOY = r'''#!/usr/bin/env bash
set -uo pipefail
PROJ="/home/imss/cw/GCN copdnoph copdph"
WLOG="/tmp/_seq_relaunch_watcher.log"

echo "===== kill+seq-watcher start $(date -Is) =====" > "${WLOG}"

# 1) kill the 2 jobs that contend for GPU
for PAT in p_theta_106_lr2x p_theta_269_lr2x; do
  echo "[$(date -Is)] killing ${PAT}" >> "${WLOG}"
  pkill -9 -f "${PAT}" || true
done
sleep 2

# 2) wipe their partial output dirs so flags never appear from old runs
for j in p_theta_106_lr2x p_theta_269_lr2x; do
  rm -rf "${PROJ}/outputs/${j}"
  echo "[$(date -Is)] wiped outputs/${j}" >> "${WLOG}"
done

# 3) sanity: report remaining tri_structure_pipeline procs
echo "[$(date -Is)] surviving procs:" >> "${WLOG}"
pgrep -af tri_structure_pipeline.py >> "${WLOG}" 2>/dev/null || echo "    (none)" >> "${WLOG}"

# Helper to start one job solo and wait synchronously
launch_solo_and_wait() {
  local name="$1" gpu="$2" lr="$3" cache="$4" labels="$5"
  local out="${PROJ}/outputs/${name}"
  local log="${out}/${name}.log"
  local flag="${out}/${name}_done.flag"
  mkdir -p "${out}"
  rm -f "${flag}"
  local script="/tmp/_launch_${name}.sh"
  cat > "${script}" <<EOF
#!/usr/bin/env bash
set -uo pipefail
source /home/imss/miniconda3/etc/profile.d/conda.sh
conda activate pulmonary_bv5_py39
cd "${PROJ}/tri_structure"
export CUDA_VISIBLE_DEVICES=${gpu}
echo "===== ${name} start \$(date -Is) (GPU ${gpu}, lr=${lr}, SOLO) =====" > "${log}"
python -u tri_structure_pipeline.py \\
    --cache_dir "${cache}" \\
    --labels "${labels}" \\
    --output_dir "${out}" \\
    --epochs 200 --repeats 3 \\
    --lr ${lr} \\
    --mpap_aux \\
    >> "${log}" 2>&1
RC=\$?
echo "===== ${name} rc=\$RC end \$(date -Is) =====" >> "${log}"
echo "\$RC" > "${flag}"
EOF
  chmod +x "${script}"
  setsid bash -c "bash ${script} > /tmp/_${name}_nohup.out 2>&1" < /dev/null > /dev/null 2>&1 &
  echo "[$(date -Is)] solo-launched ${name} on GPU${gpu} lr=${lr}" >> "${WLOG}"
}

CACHE_GOLD="${PROJ}/tri_structure/cache_tri"
LABELS_GOLD="${PROJ}/data/labels_gold.csv"
CACHE_EXP="${PROJ}/cache_tri_converted"
LABELS_EXP="${PROJ}/data/labels_expanded_282.csv"

# 4) wait for each "lrhalf" flag, then launch the matching "lr2x" solo on the same GPU
launched_106=0
launched_269=0
for i in $(seq 1 360); do
  if [ ${launched_106} -eq 0 ] && ls "${PROJ}/outputs/p_theta_106_lrhalf"/*_done.flag >/dev/null 2>&1; then
    echo "[$(date -Is)] p_theta_106_lrhalf done; launching p_theta_106_lr2x solo on GPU0" >> "${WLOG}"
    launch_solo_and_wait p_theta_106_lr2x 0 2e-3 "${CACHE_GOLD}" "${LABELS_GOLD}"
    launched_106=1
  fi
  if [ ${launched_269} -eq 0 ] && ls "${PROJ}/outputs/p_theta_269_lrhalf"/*_done.flag >/dev/null 2>&1; then
    echo "[$(date -Is)] p_theta_269_lrhalf done; launching p_theta_269_lr2x solo on GPU1" >> "${WLOG}"
    launch_solo_and_wait p_theta_269_lr2x 1 2e-3 "${CACHE_EXP}" "${LABELS_EXP}"
    launched_269=1
  fi
  if [ ${launched_106} -eq 1 ] && [ ${launched_269} -eq 1 ]; then
    break
  fi
  sleep 30
done

echo "[$(date -Is)] sequential watcher done (launched_106=${launched_106} launched_269=${launched_269})" >> "${WLOG}"
'''

c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, PORT, USER, PASS, timeout=15, allow_agent=False, look_for_keys=False)
sftp = c.open_sftp()
with sftp.file("/tmp/_seq_relaunch_watcher.sh", "w") as f:
    f.write(KILL_AND_DEPLOY)
sftp.close()

cmd = ("chmod +x /tmp/_seq_relaunch_watcher.sh && "
       "setsid bash -c 'bash /tmp/_seq_relaunch_watcher.sh > /tmp/_seq_relaunch_nohup.out 2>&1' "
       "< /dev/null > /dev/null 2>&1 &")
c.exec_command(cmd, timeout=5)
print("deployed kill + sequential watcher")
time.sleep(4)

# Show resulting state
_, o, _ = c.exec_command(
    "cat /tmp/_seq_relaunch_watcher.log 2>/dev/null; echo ---; "
    "pgrep -af tri_structure_pipeline.py; echo ---; "
    "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader",
    timeout=10)
print(o.read().decode())
c.close()
