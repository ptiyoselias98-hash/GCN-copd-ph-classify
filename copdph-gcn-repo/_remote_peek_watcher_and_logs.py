"""Peek watcher log + tail of long-running job logs to check for hangs."""
import paramiko
HOST, PORT, USER, PASS = "10.60.147.117", 22, "imss", "imsslab"

CMD = r'''
echo "=== watcher log ==="
tail -n 30 /tmp/_autofollow_watcher.log 2>/dev/null

echo
echo "=== p_eta_pool_attn tail (last 25) ==="
tail -n 25 "/home/imss/cw/GCN copdnoph copdph/outputs/p_eta_pool_attn"/p_eta_pool_attn.log 2>/dev/null

echo
echo "=== p_zeta_attn tail (last 20) ==="
tail -n 20 "/home/imss/cw/GCN copdnoph copdph/outputs/p_zeta_attn"/p_zeta_attn.log 2>/dev/null

echo
echo "=== process state for the 4 still running ==="
for pid in 11794 14448 14481 14498; do
  ps -o pid,etime,pcpu,pmem,cmd -p $pid 2>/dev/null
done
'''

c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, PORT, USER, PASS, timeout=15, allow_agent=False, look_for_keys=False)
_, o, e = c.exec_command(CMD, timeout=15)
print(o.read().decode())
err = e.read().decode()
if err.strip(): print("ERR:", err)
c.close()
