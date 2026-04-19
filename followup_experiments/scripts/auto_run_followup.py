#!/usr/bin/env python3
"""
auto_run_followup.py — paramiko orchestrator for the short/medium
follow-up experiments on top of Sprint 5 Task 5.

Both variants run on the SAME ~105 Excel-matched gold subset (mPAP only
lives in the Excel sheet, so any run that uses the mPAP aux head has to
stay on those 105 cases).

Subcommands
-----------
  gold       Regenerate gold subset artifacts (calls gen_gold_labels.py).
  push       SFTP-push train_plus.py / utils/training_plus.py / shell script
             plus the gold CSV/JSONs into the standard data dir.
  launch     Nohup-launch `_run_followup_pipeline.sh <variant>`.
  status     Tail log + pgrep + flag for a given variant.
  fetch      SFTP-pull outputs/followup_<variant>/ back to local.
  all        gold → push → launch (+ print poll command).

Server: imss@10.60.147.117   (password auth only).  Set IMSS_PASSWORD env
var to override the default "imsslab".

Typical session
---------------
    # One-off: build gold labels/splits/mpap-lookup from Excel.
    python auto_run_followup.py gold

    # Short-term — gold subset, vanilla classifier.
    # Tests the 92-extra-cases label-leakage hypothesis.
    python auto_run_followup.py all --variant short --epochs 200

    # Medium-term — gold subset with Sprint 5 v2 triple
    # (focal + node-drop + mPAP regression aux head).
    python auto_run_followup.py all --variant medium --epochs 300

    python auto_run_followup.py status --variant short
    python auto_run_followup.py fetch  --variant short
"""
from __future__ import annotations

import argparse
import os
import posixpath
import subprocess
import sys
import time
from pathlib import Path
from typing import List

try:
    import paramiko
except ImportError:
    print("[error] pip install paramiko", file=sys.stderr)
    sys.exit(1)

# ──────────────────────────── constants ────────────────────────────
HOST = "10.60.147.117"
PORT = 22
USER = "imss"
PASSWORD = os.environ.get("IMSS_PASSWORD", "imsslab")

REMOTE_PROJ = "/home/imss/cw/GCN copdnoph copdph"
REMOTE_DATA = posixpath.join(REMOTE_PROJ, "data")
REMOTE_UTILS = posixpath.join(REMOTE_PROJ, "utils")
REMOTE_ENV = ("source /home/imss/miniconda3/etc/profile.d/conda.sh "
              "&& conda activate pulmonary_bv5_py39")

HERE = Path(__file__).parent.resolve()
GOLD_DIR = HERE / "gold_data"
PATCH_DIR = HERE / "patches"

# Files to push before every launch.
CORE_PUSH = [
    (PATCH_DIR / "train_plus.py",                  posixpath.join(REMOTE_PROJ, "train_plus.py")),
    (PATCH_DIR / "training_plus.py",               posixpath.join(REMOTE_UTILS, "training_plus.py")),
    (PATCH_DIR / "cluster_vessel_topology.py",     posixpath.join(REMOTE_PROJ, "cluster_vessel_topology.py")),
    (HERE / "_run_followup_pipeline.sh",           posixpath.join(REMOTE_PROJ, "_run_followup_pipeline.sh")),
    (HERE / "_run_cluster_pipeline.sh",            posixpath.join(REMOTE_PROJ, "_run_cluster_pipeline.sh")),
]

# Variants supported by both `launch` and `_run_followup_pipeline.sh`.
VARIANT_CHOICES = [
    "short", "medium", "medium_youden",
    "medium_youden_rep",
    "mode_gcn", "mode_hybrid", "mode_radiomics",
]

# Extra files pushed when available (gold subset artifacts).
OPTIONAL_PUSH = [
    (GOLD_DIR / "labels_gold.csv",          posixpath.join(REMOTE_DATA, "labels_gold.csv")),
    (GOLD_DIR / "splits_gold.json",         posixpath.join(REMOTE_DATA, "splits_gold.json")),
    (GOLD_DIR / "mpap_lookup_gold.json",    posixpath.join(REMOTE_DATA, "mpap_lookup_gold.json")),
]


def remote_log(variant: str) -> str:
    return f"/tmp/followup_{variant}.log"


def remote_output(variant: str) -> str:
    return posixpath.join(REMOTE_PROJ, f"outputs/followup_{variant}")


def remote_flag(variant: str) -> str:
    return posixpath.join(remote_output(variant), f"followup_{variant}_done.flag")


# ──────────────────────────── ssh helpers ────────────────────────────
def connect() -> paramiko.SSHClient:
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PASSWORD,
                look_for_keys=False, allow_agent=False, timeout=30)
    return cli


def run(cli: paramiko.SSHClient, cmd: str, *,
        check: bool = True, print_stream: bool = True,
        timeout: int = 7200):
    full = f'bash -lc "{cmd}"'
    tr = cli.get_transport()
    ch = tr.open_session()
    ch.settimeout(timeout)
    ch.exec_command(full)
    out_b, err_b = [], []
    while True:
        if ch.recv_ready():
            d = ch.recv(4096).decode("utf-8", "replace")
            if print_stream:
                sys.stdout.write(d); sys.stdout.flush()
            out_b.append(d)
        if ch.recv_stderr_ready():
            d = ch.recv_stderr(4096).decode("utf-8", "replace")
            if print_stream:
                sys.stderr.write(d); sys.stderr.flush()
            err_b.append(d)
        if ch.exit_status_ready() and not ch.recv_ready() and not ch.recv_stderr_ready():
            break
        time.sleep(0.05)
    rc = ch.recv_exit_status()
    if check and rc != 0:
        raise RuntimeError(f"remote rc={rc}: {cmd}\nSTDERR:\n{''.join(err_b)}")
    return rc, "".join(out_b), "".join(err_b)


def sftp_put(cli: paramiko.SSHClient, local: Path, remote: str):
    sftp = cli.open_sftp()
    try:
        parts = remote.split("/")
        cur = ""
        for p in parts[:-1]:
            if not p:
                cur += "/"
                continue
            cur = posixpath.join(cur, p) if cur else p
            try:
                sftp.stat(cur)
            except IOError:
                try:
                    sftp.mkdir(cur)
                except IOError:
                    pass
        sftp.put(str(local), remote)
    finally:
        sftp.close()


def sftp_get(cli: paramiko.SSHClient, remote: str, local: Path):
    sftp = cli.open_sftp()
    try:
        local.parent.mkdir(parents=True, exist_ok=True)
        sftp.get(remote, str(local))
    finally:
        sftp.close()


def sftp_get_tree(cli: paramiko.SSHClient, rd: str, ld: Path):
    sftp = cli.open_sftp()
    try:
        def _walk(r, l):
            l.mkdir(parents=True, exist_ok=True)
            for entry in sftp.listdir_attr(r):
                rp = posixpath.join(r, entry.filename)
                lp = l / entry.filename
                if (entry.st_mode or 0) & 0o040000:
                    _walk(rp, lp)
                else:
                    sftp.get(rp, str(lp))
        _walk(rd, ld)
    finally:
        sftp.close()


# ──────────────────────────── subcommands ────────────────────────────
def cmd_gold(args):
    script = HERE / "gen_gold_labels.py"
    print(f"[gold] running {script}")
    rc = subprocess.call([sys.executable, str(script)])
    if rc != 0:
        sys.exit(rc)


def cmd_push(args):
    cli = connect()
    try:
        for lp, rp in CORE_PUSH:
            if not lp.exists():
                print(f"[push] MISSING core file: {lp}")
                sys.exit(2)
            sftp_put(cli, lp, rp)
            print(f"  pushed {lp.name} -> {rp}")
        for lp, rp in OPTIONAL_PUSH:
            if not lp.exists():
                print(f"  skip (missing): {lp}")
                continue
            sftp_put(cli, lp, rp)
            print(f"  pushed {lp.name} -> {rp}")
        run(cli, f"chmod +x '{posixpath.join(REMOTE_PROJ, '_run_followup_pipeline.sh')}'",
            check=False, print_stream=False)
        print("[push] done")
    finally:
        cli.close()


def cmd_launch(args):
    cli = connect()
    try:
        out = remote_output(args.variant)
        log = remote_log(args.variant)
        script = posixpath.join(REMOTE_PROJ, "_run_followup_pipeline.sh")
        run(cli, f"mkdir -p '{out}'", check=False, print_stream=False)
        cmd = (f"nohup bash '{script}' '{args.variant}' "
               f"{args.epochs} {args.n_folds} "
               f"< /dev/null > '{log}' 2>&1 & disown ; "
               f"echo launched pid=$!")
        _, stdout, _ = run(cli, cmd)
        print(stdout.strip())
        print(f"[launch] variant={args.variant}  log={log}  flag={remote_flag(args.variant)}")
    finally:
        cli.close()


def cmd_status(args):
    cli = connect()
    try:
        log = remote_log(args.variant)
        flag = remote_flag(args.variant)
        out = remote_output(args.variant)
        q = (
            f"echo '===== tail {log} =====' && "
            f"(tail -n 80 '{log}' 2>/dev/null || echo '(no log yet)') && "
            f"echo && echo '===== pgrep =====' && "
            f"(pgrep -fa 'python.*train_plus\\.py' || echo '(no python process)') && "
            f"echo && echo '===== flag =====' && "
            f"(ls -l '{flag}' 2>/dev/null || echo 'not yet: {flag}') && "
            f"echo && echo '===== output dir =====' && "
            f"(ls -l '{out}' 2>/dev/null || echo '(no output dir yet)') && "
            f"echo && echo '===== nvidia-smi =====' && "
            f"(nvidia-smi | head -n 20 2>/dev/null || echo 'no gpu info')"
        )
        run(cli, q, check=False)
    finally:
        cli.close()


def cmd_fetch(args):
    cli = connect()
    try:
        local = Path(args.local_dir or f"./outputs/followup_{args.variant}")
        out = remote_output(args.variant)
        log = remote_log(args.variant)
        print(f"[fetch] {out} -> {local}")
        sftp_get_tree(cli, out, local)
        try:
            sftp_get(cli, log, local / f"followup_{args.variant}.log")
            print(f"[fetch] got followup_{args.variant}.log")
        except Exception as e:
            print(f"[fetch] log skipped: {e}")
        print("[fetch] done")
    finally:
        cli.close()


def cmd_all(args):
    # Both short and medium need the gold CSV/JSONs on the server.
    if not args.skip_gold:
        cmd_gold(args)
    cmd_push(args)
    cmd_launch(args)
    print("\n[all] launched; poll with:")
    print(f"    python {Path(__file__).name} status --variant {args.variant}")
    print(f"    python {Path(__file__).name} fetch  --variant {args.variant}   # after flag appears")


# ──────────────────────────── cluster subcommands ────────────────────────────
CLUSTER_REMOTE_OUT = posixpath.join(REMOTE_PROJ, "outputs/cluster_topology")
CLUSTER_LOG = "/tmp/cluster_topology.log"
CLUSTER_FLAG = posixpath.join(CLUSTER_REMOTE_OUT, "cluster_pipeline_done.flag")


def cmd_cluster_launch(args):
    cli = connect()
    try:
        script = posixpath.join(REMOTE_PROJ, "_run_cluster_pipeline.sh")
        run(cli, f"chmod +x '{script}' ; mkdir -p '{CLUSTER_REMOTE_OUT}'",
            check=False, print_stream=False)
        cmd = (f"nohup bash '{script}' "
               f"< /dev/null > '{CLUSTER_LOG}' 2>&1 & disown ; "
               f"echo launched pid=$!")
        _, stdout, _ = run(cli, cmd)
        print(stdout.strip())
        print(f"[cluster launch] log={CLUSTER_LOG}  flag={CLUSTER_FLAG}")
    finally:
        cli.close()


def cmd_cluster_status(args):
    cli = connect()
    try:
        q = (
            f"echo '===== tail {CLUSTER_LOG} =====' && "
            f"(tail -n 80 '{CLUSTER_LOG}' 2>/dev/null || echo '(no log yet)') && "
            f"echo && echo '===== flag =====' && "
            f"(ls -l '{CLUSTER_FLAG}' 2>/dev/null || echo 'not yet: {CLUSTER_FLAG}') && "
            f"echo && echo '===== output =====' && "
            f"(ls -l '{CLUSTER_REMOTE_OUT}' 2>/dev/null || echo '(no output dir yet)')"
        )
        run(cli, q, check=False)
    finally:
        cli.close()


def cmd_cluster_fetch(args):
    cli = connect()
    try:
        local = Path(args.local_dir or "./outputs/cluster_topology")
        print(f"[cluster fetch] {CLUSTER_REMOTE_OUT} -> {local}")
        sftp_get_tree(cli, CLUSTER_REMOTE_OUT, local)
        try:
            sftp_get(cli, CLUSTER_LOG, local / "cluster_topology.log")
            print("[cluster fetch] got cluster_topology.log")
        except Exception as e:
            print(f"[cluster fetch] log skipped: {e}")
        print("[cluster fetch] done")
    finally:
        cli.close()


def cmd_cluster_all(args):
    cmd_push(args)
    cmd_cluster_launch(args)
    print("\n[cluster all] launched; poll with:")
    print(f"    python {Path(__file__).name} cluster status")
    print(f"    python {Path(__file__).name} cluster fetch    # after flag")


def cmd_cluster(args):
    handlers = {
        "launch": cmd_cluster_launch,
        "status": cmd_cluster_status,
        "fetch":  cmd_cluster_fetch,
        "all":    cmd_cluster_all,
    }
    handlers[args.cluster_sub](args)


# ──────────────────────────── argparse ────────────────────────────
def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="sub", required=True)

    sub.add_parser("gold", help="Regenerate gold subset artifacts locally")
    sub.add_parser("push", help="SFTP-push core + optional files to server")

    for name in ("launch", "status", "fetch"):
        s = sub.add_parser(name)
        s.add_argument("--variant", choices=VARIANT_CHOICES, required=True)
        if name == "launch":
            s.add_argument("--epochs", type=int, default=200)
            s.add_argument("--n_folds", type=int, default=5)
        if name == "fetch":
            s.add_argument("--local_dir", default=None)

    a = sub.add_parser("all", help="gold (short/both) -> push -> launch")
    a.add_argument("--variant", choices=VARIANT_CHOICES, required=True)
    a.add_argument("--epochs", type=int, default=200)
    a.add_argument("--n_folds", type=int, default=5)
    a.add_argument("--skip_gold", action="store_true",
                   help="Skip rebuilding gold artifacts before push")

    # cluster subcommand: launch / status / fetch / all
    c = sub.add_parser("cluster", help="Unsupervised vessel topology clustering")
    cs = c.add_subparsers(dest="cluster_sub", required=True)
    cs.add_parser("launch")
    cs.add_parser("status")
    cf = cs.add_parser("fetch")
    cf.add_argument("--local_dir", default=None)
    cs.add_parser("all", help="push (core+cluster script) -> launch")

    return p


def main():
    args = build_parser().parse_args()
    handlers = {
        "gold": cmd_gold, "push": cmd_push, "launch": cmd_launch,
        "status": cmd_status, "fetch": cmd_fetch, "all": cmd_all,
        "cluster": cmd_cluster,
    }
    handlers[args.sub](args)


if __name__ == "__main__":
    main()
