"""One-shot: SFTP-push 3 changed files + probe remote project + show plan."""
from __future__ import annotations

import sys
from pathlib import Path

import os
import paramiko

HOST = "10.60.147.117"
PORT = 22
USER = "imss"
PASS = os.environ.get("IMSS_SSH_PASSWORD")
if not PASS:
    raise RuntimeError("Set IMSS_SSH_PASSWORD before running this script.")
REMOTE_BASE = "/home/imss/cw/GCN copdnoph copdph"

LOCAL_REPO = Path(r"C:\Users\cheng\Desktop\图卷积-肺小血管演化规律探索\copdph-gcn-repo")
FILES = ["enhance_features.py", "hybrid_gcn.py", "run_sprint2.py"]


def main() -> int:
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, PORT, USER, PASS, timeout=15, allow_agent=False,
                look_for_keys=False)
    print(f"[ssh] connected to {HOST}")

    def run(cmd: str) -> tuple[int, str, str]:
        stdin, stdout, stderr = cli.exec_command(cmd, timeout=120)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        rc = stdout.channel.recv_exit_status()
        return rc, out, err

    # 1) probe remote layout
    rc, out, _ = run(f"ls -la '{REMOTE_BASE}' | head -50")
    print("[probe] ls remote project:")
    print(out)

    # locate the actual repo dir on remote (it might be repo subdir)
    candidates = [REMOTE_BASE, f"{REMOTE_BASE}/copdph-gcn-repo", f"{REMOTE_BASE}/copdph-gcn"]
    remote_repo = None
    for c in candidates:
        rc, out, _ = run(f"test -f '{c}/run_sprint2.py' && echo OK || echo MISS")
        if "OK" in out:
            remote_repo = c
            break
    if not remote_repo:
        # search
        rc, out, _ = run(
            f"find '{REMOTE_BASE}' -maxdepth 4 -name run_sprint2.py 2>/dev/null"
        )
        print("[probe] search run_sprint2.py:\n" + out)
        if out.strip():
            remote_repo = str(Path(out.splitlines()[0]).parent).replace("\\", "/")
    if not remote_repo:
        print("ERROR: cannot find run_sprint2.py on remote", file=sys.stderr)
        return 1
    print(f"[probe] remote_repo = {remote_repo}")

    # 2) backup and upload
    sftp = cli.open_sftp()
    for fn in FILES:
        local = LOCAL_REPO / fn
        remote = f"{remote_repo}/{fn}"
        # backup
        run(f"cp '{remote}' '{remote}.bak.$(date +%Y%m%d_%H%M%S)' 2>/dev/null || true")
        sftp.put(str(local), remote)
        print(f"[upload] {fn}  ({local.stat().st_size} bytes)")
    sftp.close()

    # 3) check radiomics CSV has required columns
    rc, out, _ = run(
        "find '/home/imss/cw' -maxdepth 6 -name 'copd_ph_radiomics.csv' 2>/dev/null"
    )
    csv_path = out.strip().splitlines()[0] if out.strip() else None
    print(f"[probe] radiomics CSV: {csv_path}")
    if csv_path:
        rc, out, _ = run(f"head -1 '{csv_path}' | tr ',' '\\n' | grep -E 'bv5_ratio|artery_vein_vol_ratio|静脉BV5|肺血管BV5|静脉血管分支数量|左右肺密度标准差|肺血管弯曲度' | head -20")
        print("[probe] required columns present:")
        print(out)

    # 4) python compile check on remote
    rc, out, err = run(
        f"cd '{remote_repo}' && python -m py_compile {' '.join(FILES)} && echo COMPILE_OK"
    )
    print("[remote-compile]", out.strip(), err.strip())

    print(f"\nREMOTE_REPO={remote_repo}")
    print(f"CSV_PATH={csv_path}")
    cli.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
