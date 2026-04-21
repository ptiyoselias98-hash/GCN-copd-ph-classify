"""Fetch p_theta_269_lr2x summary.json — missing from local _cv_results_lrsweep.json."""
import json, paramiko
HOST, PORT, USER, PASS = "10.60.147.117", 22, "imss", "imsslab"

c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, PORT, USER, PASS, timeout=15, allow_agent=False, look_for_keys=False)
sftp = c.open_sftp()

remote = "/home/imss/cw/GCN copdnoph copdph/outputs/p_theta_269_lr2x/cv_results.json"
try:
    with sftp.file(remote, "r") as f:
        data = json.loads(f.read().decode())
    print(json.dumps(data, indent=2)[:2000])
except IOError as e:
    print(f"not found: {remote} ({e})")
    _, o, _ = c.exec_command(f'ls -la "/home/imss/cw/GCN copdnoph copdph/outputs/p_theta_269_lr2x/" 2>&1', timeout=10)
    print(o.read().decode())
sftp.close()
c.close()
