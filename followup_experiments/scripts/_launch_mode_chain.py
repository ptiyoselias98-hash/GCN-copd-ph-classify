#!/usr/bin/env python3
"""Launch 3-mode comparison chain on the server."""
from __future__ import annotations

import os
import posixpath
import sys

import paramiko

HOST = "10.60.147.117"
USER = "imss"
PASSWORD = os.environ.get("IMSS_PASSWORD", "imsslab")
PROJ = "/home/imss/cw/GCN copdnoph copdph"
SCRIPT = posixpath.join(PROJ, "_run_followup_pipeline.sh")


def main() -> int:
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=22, username=USER, password=PASSWORD,
                look_for_keys=False, allow_agent=False, timeout=30)
    try:
        chain = (
            f"bash '{SCRIPT}' mode_gcn 200 5 > /tmp/followup_mode_gcn.log 2>&1 ; "
            f"bash '{SCRIPT}' mode_hybrid 200 5 > /tmp/followup_mode_hybrid.log 2>&1 ; "
            f"bash '{SCRIPT}' mode_radiomics 200 5 > /tmp/followup_mode_radiomics.log 2>&1"
        )
        # Write the chain to a remote helper script to keep escaping sane.
        helper = "/tmp/run_mode_chain.sh"
        sftp = cli.open_sftp()
        with sftp.open(helper, "w") as fh:
            fh.write("#!/usr/bin/env bash\nset -u\n" + chain + "\n")
        sftp.chmod(helper, 0o755)
        sftp.close()

        nohup = (f"nohup bash {helper} < /dev/null "
                 f"> /tmp/mode_chain.log 2>&1 & disown ; echo launched pid=$!")

        t = cli.get_transport()
        ch = t.open_session()
        ch.exec_command(f'bash -lc "{nohup}"')
        out = b""
        while True:
            if ch.recv_ready():
                out += ch.recv(4096)
            if ch.exit_status_ready() and not ch.recv_ready():
                break
        print(out.decode("utf-8", "replace").strip())
        print("[mode chain] gcn -> hybrid -> radiomics launched")
        return 0
    finally:
        cli.close()


if __name__ == "__main__":
    sys.exit(main())
