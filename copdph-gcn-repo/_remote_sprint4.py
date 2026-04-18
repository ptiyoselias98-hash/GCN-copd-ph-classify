"""Sprint 4 orchestrator — two arms in parallel, one per GPU.

Arm A  (GPU 0)  sprint4a_gated   run_sprint3.py --fusion gated
                                focal / local4 / enhanced-only (fast)

Arm B  (GPU 1)  sprint4b_av      stage 1: _build_av_lookup.py (one-shot)
                                stage 2: run_sprint3.py --av_lookup ...

Both arms launch from a single nohup supervisor.  When both finish the
supervisor touches outputs/sprint4_done.flag.  Outputs live under:

    /home/imss/cw/GCN copdnoph copdph/outputs/sprint4a_gated/
    /home/imss/cw/GCN copdnoph copdph/outputs/sprint4b_av/

Sprint 3 directories are NOT touched.

Usage:
    python _remote_sprint4.py            # push + launch (idempotent-ish)
    python _remote_sprint4.py status
    python _remote_sprint4.py fetch
    python _remote_sprint4.py push       # just refresh scripts, no launch
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import os
import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HOST, PORT, USER = "10.60.147.117", 22, "imss"
PASS = "imsslab"
REMOTE = "/home/imss/cw/GCN copdnoph copdph"
NII_ROOT = "/home/imss/cw/COPDnonPH COPD-PH /data/nii"
LABELS = "/home/imss/cw/COPDnonPH COPD-PH /data/tables/labels.csv"
SPLITS = "/home/imss/cw/COPDnonPH COPD-PH /data/splits/folds"
ENV = ("source /home/imss/miniconda3/etc/profile.d/conda.sh && "
       "conda activate pulmonary_bv5_py39")

LOCAL_REPO = Path(r"C:\Users\cheng\Desktop\图卷积-肺小血管演化规律探索\copdph-gcn-repo")
LOCAL_OUT = Path(r"C:\Users\cheng\Desktop\图卷积-肺小血管演化规律探索\outputs")

PUSH_FILES = [
    "run_sprint3.py", "run_sprint2.py", "hybrid_gcn.py",
    "enhance_features.py", "_build_av_lookup.py",
]

SENTINEL = "outputs/sprint4_done.flag"
OUT_A = "outputs/sprint4a_gated"
OUT_B = "outputs/sprint4b_av"
AV_PT = f"{OUT_B}/av_lookup.pt"


def _cli() -> paramiko.SSHClient:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, USER, PASS, timeout=15,
              allow_agent=False, look_for_keys=False)
    return c


def _run(c: paramiko.SSHClient, cmd: str, t: int = 30) -> tuple[str, str]:
    _, o, e = c.exec_command(cmd, timeout=t)
    return (o.read().decode("utf-8", "replace"),
            e.read().decode("utf-8", "replace"))


def push(c: paramiko.SSHClient) -> None:
    sftp = c.open_sftp()
    for f in PUSH_FILES:
        sftp.put(str(LOCAL_REPO / f), f"{REMOTE}/{f}")
        print(f"[push] {f}")
    sftp.close()


def _arm_a_cmd() -> str:
    """Sprint 4a: run_sprint3 with --fusion gated on GPU 0."""
    return (
        f"mkdir -p '{OUT_A}' && "
        f"echo '=== arm=4a_gated (GPU 0) ===' > '{OUT_A}/run.log' && "
        f"CUDA_VISIBLE_DEVICES=0 python -u run_sprint3.py "
        f"--cache_dir ./cache --radiomics ./data/copd_ph_radiomics.csv "
        f"--labels '{LABELS}' --splits '{SPLITS}' --output_dir './{OUT_A}' "
        f"--epochs 300 --batch_size 8 --lr 1e-3 "
        f"--loss focal --globals_keep local4 --fusion gated "
        f">> '{OUT_A}/run.log' 2>&1"
    )


def _arm_b_cmd() -> str:
    """Sprint 4b: build AV lookup then train, on GPU 1."""
    return (
        f"mkdir -p '{OUT_B}' && "
        f"echo '=== arm=4b_av (GPU 1) ===' > '{OUT_B}/run.log' && "
        f"echo '-- stage 1: build av lookup --' >> '{OUT_B}/run.log' && "
        f"CUDA_VISIBLE_DEVICES=1 python -u _build_av_lookup.py "
        f"--cache_dir ./cache --nii_root '{NII_ROOT}' --out '{AV_PT}' "
        f">> '{OUT_B}/run.log' 2>&1 && "
        f"echo '-- stage 2: train --' >> '{OUT_B}/run.log' && "
        f"CUDA_VISIBLE_DEVICES=1 python -u run_sprint3.py "
        f"--cache_dir ./cache --radiomics ./data/copd_ph_radiomics.csv "
        f"--labels '{LABELS}' --splits '{SPLITS}' --output_dir './{OUT_B}' "
        f"--epochs 300 --batch_size 8 --lr 1e-3 "
        f"--loss focal --globals_keep local4 --fusion concat "
        f"--av_lookup '{AV_PT}' "
        f">> '{OUT_B}/run.log' 2>&1"
    )


def launch(c: paramiko.SSHClient) -> None:
    # Supervisor: runs both arms concurrently, waits, then touches sentinel.
    a = _arm_a_cmd()
    b = _arm_b_cmd()
    supervisor = (
        f"cd '{REMOTE}' && rm -f '{SENTINEL}' && "
        f"( ( {a} ) & PIDA=$!; "
        f"  ( {b} ) & PIDB=$!; "
        f"  wait $PIDA; RA=$?; wait $PIDB; RB=$?; "
        f"  echo armA_exit=$RA armB_exit=$RB >> outputs/sprint4_launcher.log; "
        f"  touch '{SENTINEL}' )"
    )
    launch_cmd = (
        f"nohup bash -lc \"{ENV} && {supervisor}\" "
        f"< /dev/null > '{REMOTE}/outputs/sprint4_launcher.log' 2>&1 "
        f"& disown; echo launched"
    )
    try:
        c.exec_command(launch_cmd, timeout=5)
    except Exception:
        pass
    time.sleep(4)
    o, _ = _run(c, "pgrep -fa 'python.*run_sprint3\\|_build_av_lookup' || true")
    print("[pgrep]", o.strip() or "(none yet; may still be starting)")
    o, _ = _run(c, f"tail -n 5 '{REMOTE}/outputs/sprint4_launcher.log' "
                   f"2>/dev/null || true")
    print("[launcher.log]\n" + o)


def status(c: paramiko.SSHClient) -> None:
    print("--- sentinel ---")
    print(_run(c, f"ls -la '{REMOTE}/{SENTINEL}' 2>&1")[0].strip())
    print("\n--- pgrep ---")
    print(_run(c, "pgrep -fa 'python.*run_sprint3\\|_build_av_lookup' "
                  "|| true")[0].strip() or "(none)")
    print("\n--- nvidia-smi ---")
    print(_run(c, "nvidia-smi --query-gpu=index,utilization.gpu,memory.used "
                  "--format=csv,noheader 2>/dev/null || true")[0].strip())
    for name, d in [("4a_gated", OUT_A), ("4b_av", OUT_B)]:
        print(f"\n--- {name} tail ---")
        print(_run(c, f"tail -n 10 '{REMOTE}/{d}/run.log' 2>&1")[0])
        print(_run(c, f"ls -la '{REMOTE}/{d}/sprint3_results.json' "
                      "2>&1")[0].strip())


def fetch(c: paramiko.SSHClient) -> None:
    sftp = c.open_sftp()
    for name, rd in [("sprint4a_gated", OUT_A), ("sprint4b_av", OUT_B)]:
        local = LOCAL_OUT / name
        local.mkdir(parents=True, exist_ok=True)
        for fn in ("sprint3_results.json", "run.log"):
            rp = f"{REMOTE}/{rd}/{fn}"
            try:
                sftp.get(rp, str(local / fn))
                print(f"OK {name}/{fn} ({(local/fn).stat().st_size} B)")
            except Exception as e:
                print(f"MISS {name}/{fn}: {e}")
    try:
        sftp.get(f"{REMOTE}/outputs/sprint4_launcher.log",
                 str(LOCAL_OUT / "sprint4_launcher.log"))
    except Exception:
        pass
    sftp.close()


def main() -> None:
    action = sys.argv[1] if len(sys.argv) > 1 else "launch"
    c = _cli()
    try:
        if action == "launch":
            push(c); launch(c)
        elif action == "push":
            push(c)
        elif action == "status":
            status(c)
        elif action == "fetch":
            fetch(c)
        else:
            print(f"unknown action: {action}")
    finally:
        c.close()


if __name__ == "__main__":
    main()
