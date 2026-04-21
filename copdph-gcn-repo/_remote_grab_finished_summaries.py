"""Grab tail summaries (CV aggregate metrics) of the 4 completed jobs."""
import paramiko
HOST, PORT, USER, PASS = "10.60.147.117", 22, "imss", "imsslab"

CMD = r'''
PROJ="/home/imss/cw/GCN copdnoph copdph"
for j in p_zeta_tri_282 p_eta_sig p_eta_pool_attn p_eta_pool_add; do
  echo "================ ${j} ================"
  L="${PROJ}/outputs/${j}/${j}.log"
  if [ ! -f "$L" ]; then L="${PROJ}/outputs/${j}/p_zeta.log"; fi  # p_zeta_tri_282 uses p_zeta.log
  if [ ! -f "$L" ]; then L=$(ls "${PROJ}/outputs/${j}"/*.log 2>/dev/null | head -1); fi
  echo "log: $L"
  # Summary block: between "TRI-STRUCTURE GCN" and "SHARED EMBEDDING"
  awk '/TRI-STRUCTURE GCN/,/SHARED EMBEDDING/' "$L" | head -n 20
  echo
  echo "--- attention profile ---"
  awk '/ATTENTION PROFILES/,/^======/' "$L" | head -n 8
  echo
done

echo
echo "=== summary.json files ==="
find "${PROJ}/outputs" -name "summary.json" -newer /tmp/_autofollow_watcher.sh -type f 2>/dev/null | head -10
'''

c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, PORT, USER, PASS, timeout=15, allow_agent=False, look_for_keys=False)
_, o, e = c.exec_command(CMD, timeout=20)
print(o.read().decode())
err = e.read().decode()
if err.strip(): print("ERR:", err)
c.close()
