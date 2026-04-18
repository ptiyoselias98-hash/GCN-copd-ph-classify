"""Poll viz status."""
import os
import paramiko

HOST, PORT, USER = "10.60.147.117", 22, "imss"
PASS = os.environ.get("IMSS_SSH_PASSWORD")
if not PASS:
    raise RuntimeError("Set IMSS_SSH_PASSWORD before running this script.")
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(HOST, PORT, USER, PASS, timeout=15,
            allow_agent=False, look_for_keys=False)
LOG = "/home/imss/cw/GCN copdnoph copdph/outputs/viz_top10.log"
OUT = "/home/imss/cw/GCN copdnoph copdph/outputs/viz_top10"
_, o, _ = cli.exec_command(
    f"pgrep -af visualize.py | grep -v grep || echo NOT_RUNNING; "
    f"echo ---TAIL---; tail -n 30 '{LOG}'; echo ---FILES---; ls -la '{OUT}'"
)
print(o.read().decode("utf-8", "replace"))
cli.close()
