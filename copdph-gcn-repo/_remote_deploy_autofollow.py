"""Deploy a server-side watcher that waits for the 6 tri_structure jobs to
finish, then auto-launches an LR-sweep ablation (2 lr values × 2 cohort sizes).

Rationale:
  The pipeline trains per-sample (no DataLoader batching), so VRAM stays ~3GB
  regardless of VRAM budget — a true --batch_size flag would need a model
  rewrite. Honest substitute: LR sweep using the existing --lr flag around the
  current 1e-3 baseline (2e-3 and 5e-4). This gives 4 extra data points and
  reuses the already-converged config (mpap_aux, pool=mean).

Four follow-up jobs:
  GPU0 (n=106, tri_structure/cache_tri):
    p_theta_106_lr2x    lr=2e-3
    p_theta_106_lrhalf  lr=5e-4
  GPU1 (n=269, cache_tri_converted):
    p_theta_269_lr2x    lr=2e-3
    p_theta_269_lrhalf  lr=5e-4
"""
import paramiko
HOST, PORT, USER, PASS = "10.60.147.117", 22, "imss", "imsslab"

WATCHER = r'''#!/usr/bin/env bash
set -uo pipefail
PROJ="/home/imss/cw/GCN copdnoph copdph"
WATCH_LOG="/tmp/_autofollow_watcher.log"
echo "===== watcher start $(date -Is) =====" > "${WATCH_LOG}"

JOBS=(p_zeta_tri_282 p_eta_sig p_eta_pool_attn p_eta_pool_add p_zeta_attn p_zeta_sig)

all_done() {
  for j in "${JOBS[@]}"; do
    if ! ls "${PROJ}/outputs/${j}"/*_done.flag >/dev/null 2>&1; then
      return 1
    fi
  done
  return 0
}

# Poll every 60s up to 6 hours
for i in $(seq 1 360); do
  if all_done; then
    echo "[$(date -Is)] all 6 flags present after ${i} polls" >> "${WATCH_LOG}"
    break
  fi
  if [ $((i % 10)) -eq 0 ]; then
    echo "[$(date -Is)] poll ${i}: still waiting" >> "${WATCH_LOG}"
    # report which are still running
    for j in "${JOBS[@]}"; do
      if ! ls "${PROJ}/outputs/${j}"/*_done.flag >/dev/null 2>&1; then
        echo "    pending: ${j}" >> "${WATCH_LOG}"
      fi
    done
  fi
  sleep 60
done

if ! all_done; then
  echo "[$(date -Is)] ABORT: timeout before all flags present" >> "${WATCH_LOG}"
  exit 1
fi

echo "[$(date -Is)] launching 4 LR-sweep follow-ups" >> "${WATCH_LOG}"

launch_job() {
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
echo "===== ${name} start \$(date -Is) (GPU ${gpu}, lr=${lr}) =====" > "${log}"
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
  disown || true
  echo "[$(date -Is)] launched ${name} on GPU${gpu} lr=${lr}" >> "${WATCH_LOG}"
  sleep 2
}

CACHE_GOLD="${PROJ}/tri_structure/cache_tri"
LABELS_GOLD="${PROJ}/data/labels_gold.csv"
CACHE_EXP="${PROJ}/cache_tri_converted"
LABELS_EXP="${PROJ}/data/labels_expanded_282.csv"

launch_job p_theta_106_lr2x    0 2e-3 "${CACHE_GOLD}" "${LABELS_GOLD}"
launch_job p_theta_106_lrhalf  0 5e-4 "${CACHE_GOLD}" "${LABELS_GOLD}"
launch_job p_theta_269_lr2x    1 2e-3 "${CACHE_EXP}"  "${LABELS_EXP}"
launch_job p_theta_269_lrhalf  1 5e-4 "${CACHE_EXP}"  "${LABELS_EXP}"

sleep 6
echo "[$(date -Is)] running tri_structure_pipeline PIDs:" >> "${WATCH_LOG}"
pgrep -af tri_structure_pipeline.py >> "${WATCH_LOG}" 2>/dev/null || true
echo "[$(date -Is)] watcher done" >> "${WATCH_LOG}"
'''

c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, PORT, USER, PASS, timeout=15, allow_agent=False, look_for_keys=False)
sftp = c.open_sftp()
with sftp.file("/tmp/_autofollow_watcher.sh", "w") as f:
    f.write(WATCHER)
sftp.close()

# Launch fully detached
cmd = ("chmod +x /tmp/_autofollow_watcher.sh && "
       "setsid bash -c 'bash /tmp/_autofollow_watcher.sh > /tmp/_autofollow_nohup.out 2>&1' "
       "< /dev/null > /dev/null 2>&1 &")
c.exec_command(cmd, timeout=5)
print("deployed watcher script to /tmp/_autofollow_watcher.sh and launched it")

import time
time.sleep(3)
_, o, _ = c.exec_command("ps -ef | grep autofollow_watcher | grep -v grep", timeout=10)
print("\n=== watcher proc ===")
print(o.read().decode() or "(not yet visible — check log in 30s)")
_, o, _ = c.exec_command("cat /tmp/_autofollow_watcher.log 2>/dev/null || echo '(no log yet)'", timeout=5)
print("\n=== initial watcher log ===")
print(o.read().decode())
c.close()
