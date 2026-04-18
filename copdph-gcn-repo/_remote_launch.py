"""Launch run_sprint2.py on the remote under nohup; return pid."""
from __future__ import annotations
import sys
import os
import paramiko

HOST, PORT, USER = "10.60.147.117", 22, "imss"
PASS = "imsslab"
REMOTE_REPO = "/home/imss/cw/GCN copdnoph copdph"
ENV_ACT = "source /home/imss/miniconda3/etc/profile.d/conda.sh && conda activate pulmonary_bv5_py39"
OUT_SUBDIR = "outputs/sprint2_enhanced_v2"

# Canonical paths used by Sprint-1 (documented in README).
LABELS_REMOTE = "/home/imss/cw/COPDnonPH COPD-PH /data/tables/labels.csv"
SPLITS_REMOTE = "/home/imss/cw/COPDnonPH COPD-PH /data/splits/folds"

CMD_TEMPLATE = (
    "cd '{repo}' && "
    "mkdir -p '{out}' && "
    "nohup python -u run_sprint2.py "
    "  --cache_dir ./cache "
    "  --radiomics ./data/copd_ph_radiomics.csv "
    "  --labels '{labels}' "
    "  --splits '{splits}' "
    "  --output_dir './{out}' "
    "  --epochs 300 --batch_size 8 --lr 1e-3 "
    "  < /dev/null > '{out}/run.log' 2>&1 & PID=$!; disown; echo $PID"
)


def main() -> int:
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, PORT, USER, PASS, timeout=15, allow_agent=False,
                look_for_keys=False)

    def run(cmd: str, timeout: int = 60) -> tuple[int, str, str]:
        stdin, stdout, stderr = cli.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        rc = stdout.channel.recv_exit_status()
        return rc, out, err

    # 1) verify canonical labels/splits exist
    labels, splits = LABELS_REMOTE, SPLITS_REMOTE
    rc, out, _ = run(
        f"bash -lc \"test -f '{labels}' && echo OK_LABELS; "
        f"test -d '{splits}' && echo OK_SPLITS\""
    )
    print(out)
    if "OK_LABELS" not in out or "OK_SPLITS" not in out:
        print("ERROR: canonical labels/splits missing", file=sys.stderr)
        return 1
    print(f"[using] labels={labels}\n[using] splits={splits}")

    # 2) refuse if a run is already in flight
    rc, out, _ = run("pgrep -fa 'python.*run_sprint2\\.py' || true")
    if out.strip():
        print("[warn] sprint2 already running:\n" + out)
        return 2

    # 3) launch
    launch = CMD_TEMPLATE.format(
        repo=REMOTE_REPO, out=OUT_SUBDIR, labels=labels, splits=splits,
    )
    rc, out, err = run(f"bash -lc \"{ENV_ACT} && {launch}\"", timeout=60)
    pid = out.strip().splitlines()[-1] if out.strip() else ""
    print(f"[launch] rc={rc} pid={pid}")
    if err.strip():
        print("STDERR:", err[-1000:])

    # 4) quick sanity: pid alive, log growing
    rc, out, _ = run(
        f"bash -lc \"ps -o pid,etime,cmd -p {pid} 2>/dev/null; "
        f"echo '---'; tail -n 20 '{REMOTE_REPO}/{OUT_SUBDIR}/run.log' 2>/dev/null\""
    )
    print(out)

    cli.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
