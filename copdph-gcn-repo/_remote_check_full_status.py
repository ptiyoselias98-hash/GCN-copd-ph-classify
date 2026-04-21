"""Check 6 original jobs + 4 LR-sweep jobs + watcher log."""
import paramiko
HOST, PORT, USER, PASS = "10.60.147.117", 22, "imss", "imsslab"

CMD = r'''
PROJ="/home/imss/cw/GCN copdnoph copdph"

echo "=== watcher log (last 40 lines) ==="
tail -n 40 /tmp/_autofollow_watcher.log 2>/dev/null

echo
echo "=== flag files ==="
for j in p_zeta_tri_282 p_eta_sig p_eta_pool_attn p_eta_pool_add p_zeta_attn p_zeta_sig \
         p_theta_106_lr2x p_theta_106_lrhalf p_theta_269_lr2x p_theta_269_lrhalf; do
  f=$(ls "${PROJ}/outputs/${j}"/*_done.flag 2>/dev/null | head -1)
  if [ -n "$f" ]; then
    echo "DONE  ${j}  rc=$(cat "$f")  $(date -r "$f" -Is)"
  else
    if [ -d "${PROJ}/outputs/${j}" ]; then
      echo "RUN   ${j}"
    else
      echo "NONE  ${j}"
    fi
  fi
done

echo
echo "=== procs ==="
pgrep -af tri_structure_pipeline.py | awk '{print $1, $NF}'

echo
echo "=== GPU ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader

echo
echo "=== last log line per job ==="
for j in p_zeta_attn p_zeta_sig p_theta_106_lr2x p_theta_106_lrhalf p_theta_269_lr2x p_theta_269_lrhalf; do
  L=$(ls "${PROJ}/outputs/${j}"/*.log 2>/dev/null | head -1)
  if [ -n "$L" ]; then
    last=$(tail -n 1 "$L" 2>/dev/null | head -c 160)
    echo "${j}: ${last}"
  fi
done
'''

c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, PORT, USER, PASS, timeout=15, allow_agent=False, look_for_keys=False)
_, o, e = c.exec_command(CMD, timeout=20)
print(o.read().decode())
err = e.read().decode()
if err.strip(): print("ERR:", err)
c.close()
