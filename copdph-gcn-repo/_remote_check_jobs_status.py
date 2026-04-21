"""Check status of the 6 running tri_structure jobs + flag files."""
import paramiko
HOST, PORT, USER, PASS = "10.60.147.117", 22, "imss", "imsslab"

CMD = r'''
PROJ="/home/imss/cw/GCN copdnoph copdph"
echo "=== flags ==="
for j in p_zeta_tri_282 p_eta_sig p_eta_pool_attn p_eta_pool_add p_zeta_attn p_zeta_sig; do
  f=$(ls "${PROJ}/outputs/${j}"/*_done.flag 2>/dev/null | head -1)
  if [ -n "$f" ]; then
    echo "DONE  ${j}  rc=$(cat "$f")  $(date -r "$f" -Is)"
  else
    echo "RUN   ${j}"
  fi
done
echo
echo "=== procs ==="
pgrep -af tri_structure_pipeline.py | awk '{print $1, $NF}'
echo
echo "=== GPU ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
echo
echo "=== tail last log line of each job ==="
for j in p_zeta_tri_282 p_eta_sig p_eta_pool_attn p_eta_pool_add p_zeta_attn p_zeta_sig; do
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
