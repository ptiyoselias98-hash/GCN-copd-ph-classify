"""One-shot orchestrator for sprint3 P0 experiments.

Pushes run_sprint3.py then launches 3 arms SERIALLY on remote:
  A: focal + local4   (primary P0 combination)
  B: focal + all      (ablate globals pruning)
  C: weighted_ce + local4  (ablate focal)

Each arm writes to outputs/sprint3_<arm>/sprint3_results.json. When all 3
done, a sentinel file outputs/sprint3_done.flag is created so the polling
script can detect completion without pgrep races.

Usage:
    python _remote_sprint3.py            # push + launch (idempotent)
    python _remote_sprint3.py status     # tail logs + check flag
    python _remote_sprint3.py fetch      # sftp results back locally
"""
from __future__ import annotations
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
import os
import paramiko

HOST, PORT, USER = "10.60.147.117", 22, "imss"
PASS = "imsslab"
REMOTE = "/home/imss/cw/GCN copdnoph copdph"
LOCAL_REPO = Path(r"C:\Users\cheng\Desktop\图卷积-肺小血管演化规律探索\copdph-gcn-repo")
LOCAL_OUT = Path(r"C:\Users\cheng\Desktop\图卷积-肺小血管演化规律探索\outputs")
ENV = "source /home/imss/miniconda3/etc/profile.d/conda.sh && conda activate pulmonary_bv5_py39"
LABELS = "/home/imss/cw/COPDnonPH COPD-PH /data/tables/labels.csv"
SPLITS = "/home/imss/cw/COPDnonPH COPD-PH /data/splits/folds"

ARMS = [
    ("focal_local4", "focal", "local4"),
    ("focal_all",    "focal", "all"),
    ("wce_local4",   "weighted_ce", "local4"),
]
SENTINEL = "outputs/sprint3_done.flag"


def _cli():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, PORT, USER, PASS, timeout=15,
                allow_agent=False, look_for_keys=False)
    return cli


def _run(cli, cmd, t=30):
    _, o, e = cli.exec_command(cmd, timeout=t)
    return o.read().decode("utf-8", "replace"), e.read().decode("utf-8", "replace")


def push(cli):
    sftp = cli.open_sftp()
    for f in ("run_sprint3.py", "run_sprint2.py", "hybrid_gcn.py", "enhance_features.py"):
        sftp.put(str(LOCAL_REPO / f), f"{REMOTE}/{f}")
        print(f"[push] {f}")
    sftp.close()


def launch(cli):
    # Build one big bash loop. Each arm runs after the previous finishes.
    body_lines = [f"cd '{REMOTE}'", "rm -f outputs/sprint3_done.flag"]
    for name, loss, keep in ARMS:
        out = f"outputs/sprint3_{name}"
        body_lines += [
            f"mkdir -p '{out}'",
            f"echo '=== arm={name} loss={loss} keep={keep} ===' > '{out}/run.log'",
            f"python -u run_sprint3.py "
            f"--cache_dir ./cache --radiomics ./data/copd_ph_radiomics.csv "
            f"--labels '{LABELS}' --splits '{SPLITS}' --output_dir './{out}' "
            f"--epochs 300 --batch_size 8 --lr 1e-3 "
            f"--loss {loss} --globals_keep {keep} "
            f">> '{out}/run.log' 2>&1",
        ]
    body_lines.append(f"touch '{SENTINEL}'")
    body = " && ".join(body_lines)
    # Outer nohup wrapper
    launch_cmd = (
        f"nohup bash -lc \"{ENV} && {body}\" "
        f"< /dev/null > '{REMOTE}/outputs/sprint3_launcher.log' 2>&1 & disown; echo launched"
    )
    try:
        cli.exec_command(launch_cmd, timeout=5)
    except Exception:
        pass
    time.sleep(4)
    o, _ = _run(cli, "pgrep -fa 'python.*run_sprint3' || true")
    print("[pgrep]", o.strip() or "(none yet; may still be starting)")
    o, _ = _run(cli, f"tail -n 5 '{REMOTE}/outputs/sprint3_launcher.log' 2>/dev/null || true")
    print("[launcher.log]\n" + o)


def status(cli):
    print("--- sentinel ---")
    o, _ = _run(cli, f"ls -la '{REMOTE}/{SENTINEL}' 2>&1")
    print(o.strip())
    print("\n--- pgrep ---")
    o, _ = _run(cli, "pgrep -fa 'python.*run_sprint3' || true")
    print(o.strip() or "(none)")
    print("\n--- nvidia-smi ---")
    o, _ = _run(cli, "nvidia-smi --query-gpu=index,utilization.gpu,memory.used "
                     "--format=csv,noheader 2>/dev/null || true")
    print(o.strip())
    for name, _, _ in ARMS:
        out = f"outputs/sprint3_{name}"
        print(f"\n--- {name} tail ---")
        o, _ = _run(cli, f"tail -n 8 '{REMOTE}/{out}/run.log' 2>&1 | tail -n 8")
        print(o)
        o, _ = _run(cli, f"ls -la '{REMOTE}/{out}/sprint3_results.json' 2>&1")
        print(o.strip())


def fetch(cli):
    sftp = cli.open_sftp()
    for name, _, _ in ARMS:
        local = LOCAL_OUT / f"sprint3_{name}"
        local.mkdir(parents=True, exist_ok=True)
        for fn in ("sprint3_results.json", "run.log"):
            rp = f"{REMOTE}/outputs/sprint3_{name}/{fn}"
            try:
                sftp.get(rp, str(local / fn))
                print(f"OK {name}/{fn}  ({(local/fn).stat().st_size} B)")
            except Exception as e:
                print(f"MISS {name}/{fn}: {e}")
    try:
        sftp.get(f"{REMOTE}/outputs/sprint3_launcher.log",
                 str(LOCAL_OUT / "sprint3_launcher.log"))
    except Exception:
        pass
    sftp.close()


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else "launch"
    cli = _cli()
    try:
        if action == "launch":
            push(cli); launch(cli)
        elif action == "status":
            status(cli)
        elif action == "fetch":
            fetch(cli)
        elif action == "push":
            push(cli)
        else:
            print(f"unknown action: {action}")
    finally:
        cli.close()


if __name__ == "__main__":
    main()
