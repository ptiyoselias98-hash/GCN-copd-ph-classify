#!/usr/bin/env python3
"""auto_run_sprint7.py -- paramiko orchestrator for Sprint 7.

Workspace isolation: everything lives under `/home/imss/cw/GCN copdnoph copdph/sprint7/`
so it never collides with Phase 2 / v2 artefacts.

Subcommands (scoped to Tasks 1 + 2 -- sweep/launch added after QA lands):
  push           SFTP Python sources + shell scripts into remote sprint7/.
  rebuild        Launch Task 1 (rebuild_cache_tri.py).
  qa             Launch Task 2 (qa_cache_tri.py) -- hard gate.
  status --what {rebuild,qa}
  watch  --what {rebuild,qa} [--interval 600]   (poll until done flag)
  fetch  --what {rebuild_log,qa}
"""
from __future__ import annotations

import argparse
import os
import posixpath
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import paramiko
except ImportError:
    print("[error] pip install paramiko", file=sys.stderr)
    sys.exit(1)

HOST = "10.60.147.117"
PORT = 22
USER = "imss"
PASSWORD = os.environ.get("IMSS_PASSWORD", "imsslab")

REMOTE_PROJ = "/home/imss/cw/GCN copdnoph copdph"
REMOTE_S7 = posixpath.join(REMOTE_PROJ, "sprint7")
REMOTE_RAW = "/home/imss/cw/COPDnonPH COPD-PH /data/nii"
REMOTE_ENV = ("source /home/imss/miniconda3/etc/profile.d/conda.sh "
              "&& conda activate pulmonary_bv5_py39")

HERE = Path(__file__).parent.resolve()
SRC_DIR = HERE / "src"

# Files pushed into remote sprint7/.
PUSH_LIST = [
    (SRC_DIR / "rebuild_cache_tri.py",      posixpath.join(REMOTE_S7, "rebuild_cache_tri.py")),
    (SRC_DIR / "qa_cache_tri.py",           posixpath.join(REMOTE_S7, "qa_cache_tri.py")),
    # Model + pipeline are needed for Task 3 / Task 5; push now so the
    # workspace is self-contained even before pipeline patches land.
    (SRC_DIR / "models.py",                 posixpath.join(REMOTE_S7, "models.py")),
    (SRC_DIR / "graph_partitioner.py",      posixpath.join(REMOTE_S7, "graph_partitioner.py")),
    (SRC_DIR / "tri_structure_pipeline.py", posixpath.join(REMOTE_S7, "tri_structure_pipeline.py")),
    (HERE / "_run_rebuild_cache_tri.sh",    posixpath.join(REMOTE_S7, "_run_rebuild_cache_tri.sh")),
    (HERE / "_run_qa_cache_tri.sh",         posixpath.join(REMOTE_S7, "_run_qa_cache_tri.sh")),
    (HERE / "_run_sweep_edrop.sh",          posixpath.join(REMOTE_S7, "_run_sweep_edrop.sh")),
    (HERE / "_run_phase2_train.sh",         posixpath.join(REMOTE_S7, "_run_phase2_train.sh")),
]

SHELL_SCRIPTS = [
    "_run_rebuild_cache_tri.sh",
    "_run_qa_cache_tri.sh",
    "_run_sweep_edrop.sh",
    "_run_phase2_train.sh",
]


def _targets(what: str) -> dict:
    if what == "rebuild":
        return {
            "log":  "/tmp/sprint7_rebuild.log",
            "out":  posixpath.join(REMOTE_S7, "cache_tri"),
            "flag": posixpath.join(REMOTE_S7, "cache_tri/rebuild_cache_tri_done.flag"),
            "proc_pattern": r"python.*rebuild_cache_tri\.py",
        }
    if what == "qa":
        return {
            "log":  "/tmp/sprint7_qa.log",
            "out":  posixpath.join(REMOTE_PROJ, "outputs/sprint7_qa"),
            "flag": posixpath.join(REMOTE_PROJ, "outputs/sprint7_qa/qa_done.flag"),
            "proc_pattern": r"python.*qa_cache_tri\.py",
        }
    if what == "sweep":
        return {
            "log":  "/tmp/sprint7_sweep_edrop.log",
            "out":  posixpath.join(REMOTE_S7, "outputs"),
            "flag": posixpath.join(REMOTE_S7, "outputs/sweep_edrop_done.flag"),
            "proc_pattern": r"(_run_sweep_edrop\.sh|tri_structure_pipeline\.py.*sweep_edrop)",
        }
    if what == "phase2":
        return {
            "log":  "/tmp/sprint7_phase2.log",
            "out":  posixpath.join(REMOTE_S7, "outputs/tri_phase2"),
            "flag": posixpath.join(REMOTE_S7, "outputs/tri_phase2/phase2_done.flag"),
            "proc_pattern": r"(_run_phase2_train\.sh|tri_structure_pipeline\.py.*tri_phase2)",
        }
    raise ValueError(f"unknown what={what!r}")


def connect() -> paramiko.SSHClient:
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PASSWORD,
                look_for_keys=False, allow_agent=False, timeout=30)
    return cli


def run(cli, cmd, *, check=True, print_stream=True, timeout=7200):
    # Send the command directly -- paramiko invokes the user's login shell
    # (bash). Wrapping in `bash -lc "..."` causes the outer shell to expand
    # $! / $(...) before the inner bash sees them, which mangles pid capture
    # and command substitutions on Windows-originated paramiko calls.
    tr = cli.get_transport()
    ch = tr.open_session()
    ch.settimeout(timeout)
    ch.exec_command(cmd)
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


def sftp_put(cli, local: Path, remote: str):
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


def sftp_get(cli, remote: str, local: Path):
    sftp = cli.open_sftp()
    try:
        local.parent.mkdir(parents=True, exist_ok=True)
        sftp.get(remote, str(local))
    finally:
        sftp.close()


def sftp_get_tree(cli, rd: str, ld: Path):
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


def cmd_push(args):
    cli = connect()
    try:
        for lp, rp in PUSH_LIST:
            if not lp.exists():
                print(f"[push] MISSING: {lp}")
                sys.exit(2)
            sftp_put(cli, lp, rp)
            print(f"  pushed {lp.name} -> {rp}")
        chmod_list = " ".join(f"'{posixpath.join(REMOTE_S7, s)}'" for s in SHELL_SCRIPTS)
        run(cli, f"chmod +x {chmod_list}", check=False, print_stream=False)
        print("[push] done")
    finally:
        cli.close()


def cmd_rebuild(args):
    tgt = _targets("rebuild")
    cli = connect()
    try:
        script = posixpath.join(REMOTE_S7, "_run_rebuild_cache_tri.sh")
        run(cli, f"mkdir -p '{tgt['out']}'", check=False, print_stream=False)
        raw = args.raw_dir or REMOTE_RAW
        overwrite = "1" if args.overwrite else "0"
        cmd = (f"nohup bash '{script}' '{raw}' {args.workers} {overwrite} "
               f"< /dev/null > '{tgt['log']}' 2>&1 & disown ; "
               f"echo launched pid=$!")
        _, stdout, _ = run(cli, cmd)
        print(stdout.strip())
        print(f"[rebuild] log={tgt['log']} flag={tgt['flag']}")
        print("[rebuild] slow step -- 3 skeletonise per patient.")
        print("          poll: python auto_run_sprint7.py status --what rebuild")
    finally:
        cli.close()


def cmd_qa(args):
    tgt = _targets("qa")
    cli = connect()
    try:
        script = posixpath.join(REMOTE_S7, "_run_qa_cache_tri.sh")
        run(cli, f"mkdir -p '{tgt['out']}'", check=False, print_stream=False)
        cmd = (f"nohup bash '{script}' "
               f"< /dev/null > '{tgt['log']}' 2>&1 & disown ; "
               f"echo launched pid=$!")
        _, stdout, _ = run(cli, cmd)
        print(stdout.strip())
        print(f"[qa] log={tgt['log']} flag={tgt['flag']}")
    finally:
        cli.close()


def cmd_sweep(args):
    tgt = _targets("sweep")
    cli = connect()
    try:
        script = posixpath.join(REMOTE_S7, "_run_sweep_edrop.sh")
        run(cli, f"mkdir -p '{tgt['out']}'", check=False, print_stream=False)
        cmd = (f"nohup bash '{script}' "
               f"< /dev/null > '{tgt['log']}' 2>&1 & disown ; "
               f"echo launched pid=$!")
        _, stdout, _ = run(cli, cmd)
        print(stdout.strip())
        print(f"[sweep] log={tgt['log']} flag={tgt['flag']}")
    finally:
        cli.close()


def cmd_phase2(args):
    tgt = _targets("phase2")
    cli = connect()
    try:
        script = posixpath.join(REMOTE_S7, "_run_phase2_train.sh")
        run(cli, f"mkdir -p '{tgt['out']}'", check=False, print_stream=False)
        cmd = (f"nohup bash '{script}' {args.best_p} "
               f"< /dev/null > '{tgt['log']}' 2>&1 & disown ; "
               f"echo launched pid=$!")
        _, stdout, _ = run(cli, cmd)
        print(stdout.strip())
        print(f"[phase2] log={tgt['log']} flag={tgt['flag']}  best_p={args.best_p}")
    finally:
        cli.close()


def cmd_status(args):
    tgt = _targets(args.what)
    cli = connect()
    try:
        q = (
            f"echo '===== tail {tgt['log']} =====' ; "
            f"(tail -n 60 '{tgt['log']}' 2>/dev/null || echo '(no log yet)') ; "
            f"echo ; echo '===== pgrep =====' ; "
            f"(pgrep -fa '{tgt['proc_pattern']}' || echo '(no matching process)') ; "
            f"echo ; echo '===== flag =====' ; "
            f"(ls -l '{tgt['flag']}' 2>/dev/null || echo 'not yet: {tgt['flag']}') ; "
            f"echo ; echo '===== output dir =====' ; "
            f"(echo \"pkl_count: $(ls -1 '{tgt['out']}' 2>/dev/null | grep -c '_tri.pkl$')\" ; "
            f"ls -l '{tgt['out']}' 2>/dev/null | tail -n 10 || echo '(no output dir yet)') ; "
            f"echo ; echo '===== nvidia-smi =====' ; "
            f"(nvidia-smi | head -n 15 2>/dev/null || echo 'no gpu info')"
        )
        run(cli, q, check=False)
    finally:
        cli.close()


def cmd_watch(args):
    """Poll status every --interval seconds until the done flag appears."""
    tgt = _targets(args.what)
    interval = max(60, int(args.interval))
    deadline = time.time() + args.max_hours * 3600
    tick = 0
    while True:
        tick += 1
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n========== watch tick #{tick} @ {ts}  what={args.what} ==========")
        cli = connect()
        try:
            # Compact status: log tail, proc count, flag, pkl count.
            q = (
                f"echo '---tail---' ; "
                f"(tail -n 20 '{tgt['log']}' 2>/dev/null || echo '(no log)') ; "
                f"echo ; echo '---procs---' ; "
                f"(pgrep -fa '{tgt['proc_pattern']}' | wc -l | xargs -I{{}} echo 'matching procs: {{}}') ; "
                f"echo ; echo '---flag---' ; "
                f"(ls -l '{tgt['flag']}' 2>/dev/null || echo 'flag not yet') ; "
                f"echo ; echo '---output---' ; "
                f"(echo \"pkl_count: $(ls -1 '{tgt['out']}' 2>/dev/null | grep -c '_tri.pkl$')\")"
            )
            run(cli, q, check=False)
            _, flag_out, _ = run(cli, f"test -f '{tgt['flag']}' && echo DONE || echo PENDING",
                                 check=False, print_stream=False)
        finally:
            cli.close()
        if "DONE" in flag_out:
            print(f"\n[watch] done flag detected after {tick} ticks.")
            return
        if time.time() > deadline:
            print(f"\n[watch] timeout after {args.max_hours}h -- flag still absent.")
            sys.exit(3)
        print(f"[watch] sleeping {interval}s ...")
        time.sleep(interval)


def cmd_fetch(args):
    cli = connect()
    try:
        if args.what == "rebuild_log":
            tgt = _targets("rebuild")
            local = (Path(args.local_dir or HERE / "outputs" / "sprint7_rebuild")
                     .expanduser().resolve())
            local.mkdir(parents=True, exist_ok=True)
            sftp_get(cli, tgt["log"], local / "sprint7_rebuild.log")
            print(f"[fetch] wrote {local}/sprint7_rebuild.log")
        elif args.what == "qa":
            tgt = _targets("qa")
            local = (Path(args.local_dir or HERE / "outputs" / "sprint7_qa")
                     .expanduser().resolve())
            sftp_get_tree(cli, tgt["out"], local)
            # also grab the stdout log
            try:
                sftp_get(cli, tgt["log"], local / "sprint7_qa.log")
            except Exception:
                pass
            print(f"[fetch] {tgt['out']} -> {local}")
        elif args.what == "sweep":
            tgt = _targets("sweep")
            local = (Path(args.local_dir or HERE / "outputs" / "sprint7_sweep_edrop")
                     .expanduser().resolve())
            local.mkdir(parents=True, exist_ok=True)
            # Fetch each per-p output dir (sweep_edrop_p<p>).
            for p in ("0.0", "0.05", "0.10", "0.15"):
                remote_dir = posixpath.join(REMOTE_S7, f"outputs/sweep_edrop_p{p}")
                try:
                    sftp_get_tree(cli, remote_dir, local / f"sweep_edrop_p{p}")
                except Exception as e:
                    print(f"  skip p={p}: {e}")
            try:
                sftp_get(cli, tgt["log"], local / "sprint7_sweep_edrop.log")
            except Exception:
                pass
            print(f"[fetch] sweep outputs -> {local}")
        elif args.what == "phase2":
            tgt = _targets("phase2")
            local = (Path(args.local_dir or HERE / "outputs" / "sprint7_phase2")
                     .expanduser().resolve())
            sftp_get_tree(cli, tgt["out"], local)
            try:
                sftp_get(cli, tgt["log"], local / "sprint7_phase2.log")
            except Exception:
                pass
            print(f"[fetch] {tgt['out']} -> {local}")
        else:
            raise ValueError(f"unknown --what {args.what!r}")
    finally:
        cli.close()


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="sub", required=True)

    sub.add_parser("push", help="SFTP src + shell scripts to remote sprint7/.")

    reb = sub.add_parser("rebuild", help="Launch Task 1 (rebuild_cache_tri.py).")
    reb.add_argument("--raw_dir", default=None,
                     help=f"Raw masks dir (default: {REMOTE_RAW})")
    reb.add_argument("--workers", type=int, default=8)
    reb.add_argument("--no-overwrite", dest="overwrite", action="store_false",
                     help="Skip cases that already have a pkl (default: overwrite).")
    reb.set_defaults(overwrite=True)

    sub.add_parser("qa", help="Launch Task 2 (QA gate).")

    sub.add_parser("sweep", help="Launch Task 3 edge_drop_p sweep.")

    ph = sub.add_parser("phase2", help="Launch Task 5 Phase 2 full training.")
    ph.add_argument("--best_p", type=float, required=True,
                    help="Edge dropout winner from Task 3 sweep.")

    st = sub.add_parser("status", help="Tail log + pgrep + flag + output dir.")
    st.add_argument("--what", choices=["rebuild", "qa", "sweep", "phase2"], required=True)

    wt = sub.add_parser("watch", help="Poll status every --interval s until done flag.")
    wt.add_argument("--what", choices=["rebuild", "qa", "sweep", "phase2"], required=True)
    wt.add_argument("--interval", type=int, default=600, help="Seconds between polls (default 600).")
    wt.add_argument("--max_hours", type=float, default=6.0, help="Abort after this many hours.")

    ft = sub.add_parser("fetch", help="SFTP pull log / qa output.")
    ft.add_argument("--what", choices=["rebuild_log", "qa", "sweep", "phase2"], required=True)
    ft.add_argument("--local_dir", default=None)

    return p


def main():
    args = build_parser().parse_args()
    handlers = {
        "push":    cmd_push,
        "rebuild": cmd_rebuild,
        "qa":      cmd_qa,
        "sweep":   cmd_sweep,
        "phase2":  cmd_phase2,
        "status":  cmd_status,
        "watch":   cmd_watch,
        "fetch":   cmd_fetch,
    }
    handlers[args.sub](args)


if __name__ == "__main__":
    main()
