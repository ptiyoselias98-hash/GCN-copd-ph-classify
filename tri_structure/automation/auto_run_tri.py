#!/usr/bin/env python3
"""
auto_run_tri.py — paramiko orchestrator for the tri-structure GCN pipeline
(artery + vein + airway encoders + cross-structure attention).

Mirrors the followup_automation pattern: SFTP push, nohup launch with
stdin redirect from /dev/null, done-flag polling, SFTP fetch.

Subcommands
-----------
  push             SFTP-push the 4 Python sources + 2 shell scripts into
                   the remote tri_structure/ dir.
  launch --phase   Nohup-launch `_run_tri_pipeline.sh <phase>`.
                     phase=1 → existing cache + heuristic partition
                     phase=2 → cache_tri/ (requires rebuild first)
  rebuild          Nohup-launch `_run_tri_rebuild_cache.sh` (Phase 2 prep).
  status --what    Tail log + pgrep + flag + output dir listing.
                     what ∈ {phase1, phase2, rebuild}
  fetch --what     SFTP-pull the corresponding remote output dir.
  all --phase      push → launch phase N.  For phase 2 you must have
                   run `rebuild` (and confirmed its flag) separately first.

Server: imss@10.60.147.117 (password auth only).  Set IMSS_PASSWORD env
var to override the default "imsslab".

Typical session
---------------
    # Phase 1 (existing cache, 200 epochs, 3 repeats, mPAP aux):
    python auto_run_tri.py all --phase 1

    python auto_run_tri.py status --what phase1
    python auto_run_tri.py fetch  --what phase1

    # Phase 2 (after Phase 1 lands, rebuild per-structure cache from raw masks):
    python auto_run_tri.py push
    python auto_run_tri.py rebuild
    python auto_run_tri.py status --what rebuild
    # ... (rebuild is slow; wait for flag)
    python auto_run_tri.py launch --phase 2
"""
from __future__ import annotations

import argparse
import os
import posixpath
import sys
import time
from pathlib import Path

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
REMOTE_TRI = posixpath.join(REMOTE_PROJ, "tri_structure")
REMOTE_ENV = ("source /home/imss/miniconda3/etc/profile.d/conda.sh "
              "&& conda activate pulmonary_bv5_py39")

HERE = Path(__file__).parent.resolve()
SRC_DIR = HERE / "src"

# Python sources go into tri_structure/ (the pipeline's working dir).
CORE_PUSH = [
    (SRC_DIR / "models.py",                  posixpath.join(REMOTE_TRI, "models.py")),
    (SRC_DIR / "graph_partitioner.py",       posixpath.join(REMOTE_TRI, "graph_partitioner.py")),
    (SRC_DIR / "tri_structure_pipeline.py",  posixpath.join(REMOTE_TRI, "tri_structure_pipeline.py")),
    (SRC_DIR / "rebuild_cache_separate.py",  posixpath.join(REMOTE_TRI, "rebuild_cache_separate.py")),
    (HERE / "_run_tri_pipeline.sh",          posixpath.join(REMOTE_TRI, "_run_tri_pipeline.sh")),
    (HERE / "_run_tri_pipeline_v2.sh",       posixpath.join(REMOTE_TRI, "_run_tri_pipeline_v2.sh")),
    (HERE / "_run_tri_rebuild_cache.sh",     posixpath.join(REMOTE_TRI, "_run_tri_rebuild_cache.sh")),
]


# Remote artifacts keyed by "what" name.
def _targets(what: str) -> dict:
    if what == "phase1":
        return {
            "log":  "/tmp/tri_phase1.log",
            "out":  posixpath.join(REMOTE_PROJ, "outputs/tri_phase1"),
            "flag": posixpath.join(REMOTE_PROJ, "outputs/tri_phase1/tri_phase1_done.flag"),
            "proc_pattern": r"python.*tri_structure_pipeline\.py",
        }
    if what == "phase2":
        return {
            "log":  "/tmp/tri_phase2.log",
            "out":  posixpath.join(REMOTE_PROJ, "outputs/tri_phase2"),
            "flag": posixpath.join(REMOTE_PROJ, "outputs/tri_phase2/tri_phase2_done.flag"),
            "proc_pattern": r"python.*tri_structure_pipeline\.py",
        }
    if what == "phase2_v2":
        return {
            "log":  "/tmp/tri_phase2_v2.log",
            "out":  posixpath.join(REMOTE_PROJ, "outputs/tri_phase2_v2"),
            "flag": posixpath.join(REMOTE_PROJ, "outputs/tri_phase2_v2/tri_phase2_v2_done.flag"),
            "proc_pattern": r"python.*tri_structure_pipeline\.py.*pool_mode",
        }
    if what == "rebuild":
        return {
            "log":  "/tmp/tri_rebuild_cache.log",
            "out":  posixpath.join(REMOTE_TRI, "cache_tri"),
            "flag": posixpath.join(REMOTE_TRI, "cache_tri/rebuild_cache_done.flag"),
            "proc_pattern": r"python.*rebuild_cache_separate\.py",
        }
    raise ValueError(f"unknown what={what!r}")


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
def cmd_push(args):
    cli = connect()
    try:
        for lp, rp in CORE_PUSH:
            if not lp.exists():
                print(f"[push] MISSING: {lp}")
                sys.exit(2)
            sftp_put(cli, lp, rp)
            print(f"  pushed {lp.name} -> {rp}")
        # make all shell scripts executable
        run(cli,
            f"chmod +x '{posixpath.join(REMOTE_TRI, '_run_tri_pipeline.sh')}' "
            f"'{posixpath.join(REMOTE_TRI, '_run_tri_pipeline_v2.sh')}' "
            f"'{posixpath.join(REMOTE_TRI, '_run_tri_rebuild_cache.sh')}'",
            check=False, print_stream=False)
        print("[push] done")
    finally:
        cli.close()


def cmd_launch(args):
    what = f"phase{args.phase}"
    tgt = _targets(what)
    cli = connect()
    try:
        script = posixpath.join(REMOTE_TRI, "_run_tri_pipeline.sh")
        run(cli, f"mkdir -p '{tgt['out']}'", check=False, print_stream=False)
        cmd = (f"nohup bash '{script}' {args.phase} "
               f"{args.epochs} {args.repeats} {args.n_folds} "
               f"< /dev/null > '{tgt['log']}' 2>&1 & disown ; "
               f"echo launched pid=$!")
        _, stdout, _ = run(cli, cmd)
        print(stdout.strip())
        print(f"[launch] phase={args.phase} log={tgt['log']} flag={tgt['flag']}")
    finally:
        cli.close()


def cmd_launch_v2(args):
    """Launch the v2 pipeline: attention pooling + graph signatures.

    Writes to outputs/tri_phase2_v2/ and uses a separate /tmp log so it never
    collides with the v1 phase2 launch.
    """
    tgt = _targets("phase2_v2")
    cli = connect()
    try:
        script = posixpath.join(REMOTE_TRI, "_run_tri_pipeline_v2.sh")
        run(cli, f"mkdir -p '{tgt['out']}'", check=False, print_stream=False)
        cmd = (f"nohup bash '{script}' "
               f"{args.epochs} {args.repeats} {args.n_folds} "
               f"< /dev/null > '{tgt['log']}' 2>&1 & disown ; "
               f"echo launched pid=$!")
        _, stdout, _ = run(cli, cmd)
        print(stdout.strip())
        print(f"[launch_v2] log={tgt['log']} flag={tgt['flag']}")
    finally:
        cli.close()


def cmd_rebuild(args):
    tgt = _targets("rebuild")
    cli = connect()
    try:
        script = posixpath.join(REMOTE_TRI, "_run_tri_rebuild_cache.sh")
        run(cli, f"mkdir -p '{tgt['out']}'", check=False, print_stream=False)
        # Positional args: raw_dir, workers. The shell script defaults both
        # when omitted, but if we want workers we must also pass raw_dir.
        raw_arg = f"'{args.raw_dir}'" if args.raw_dir else "''"
        cmd = (f"nohup bash '{script}' {raw_arg} {args.workers} "
               f"< /dev/null > '{tgt['log']}' 2>&1 & disown ; "
               f"echo launched pid=$!")
        _, stdout, _ = run(cli, cmd)
        print(stdout.strip())
        print(f"[rebuild] log={tgt['log']} flag={tgt['flag']}")
        print("[rebuild] this is slow (re-skeletonization of 3 masks per patient).")
        print("          poll with: python auto_run_tri.py status --what rebuild")
    finally:
        cli.close()


def cmd_status(args):
    tgt = _targets(args.what)
    cli = connect()
    try:
        q = (
            f"echo '===== tail {tgt['log']} =====' && "
            f"(tail -n 80 '{tgt['log']}' 2>/dev/null || echo '(no log yet)') && "
            f"echo && echo '===== pgrep =====' && "
            f"(pgrep -fa '{tgt['proc_pattern']}' || echo '(no matching python process)') && "
            f"echo && echo '===== flag =====' && "
            f"(ls -l '{tgt['flag']}' 2>/dev/null || echo 'not yet: {tgt['flag']}') && "
            f"echo && echo '===== output dir =====' && "
            f"(echo \"pkl_count: $(ls -1 '{tgt['out']}' 2>/dev/null | grep -c '_tri.pkl$')\"; ls -l '{tgt['out']}' 2>/dev/null | tail -n 10 || echo '(no output dir yet)') && "
            f"echo && echo '===== nvidia-smi =====' && "
            f"(nvidia-smi | head -n 20 2>/dev/null || echo 'no gpu info')"
        )
        run(cli, q, check=False)
    finally:
        cli.close()


def cmd_fetch(args):
    tgt = _targets(args.what)
    cli = connect()
    try:
        default_local = {
            "phase1":    "./outputs/tri_phase1",
            "phase2":    "./outputs/tri_phase2",
            "phase2_v2": "./outputs/tri_phase2_v2",
            "rebuild":   "./outputs/tri_rebuild_cache",
        }[args.what]
        local = Path(args.local_dir or default_local)

        if args.what == "rebuild":
            # cache_tri/ is huge — by default only pull the log + flag,
            # not the per-patient .pkl files.  Use --include_cache to pull all.
            local.mkdir(parents=True, exist_ok=True)
            try:
                sftp_get(cli, tgt["log"], local / "tri_rebuild_cache.log")
                print(f"[fetch] got tri_rebuild_cache.log -> {local}")
            except Exception as e:
                print(f"[fetch] log skipped: {e}")
            try:
                sftp_get(cli, tgt["flag"], local / "rebuild_cache_done.flag")
                print("[fetch] got rebuild_cache_done.flag")
            except Exception as e:
                print(f"[fetch] flag skipped: {e}")
            if args.include_cache:
                print(f"[fetch] pulling full cache_tri/ -> {local} (this can be large)")
                sftp_get_tree(cli, tgt["out"], local / "cache_tri")
            else:
                print("[fetch] skipping full cache_tri/ (rerun with --include_cache to pull)")
            return

        print(f"[fetch] {tgt['out']} -> {local}")
        sftp_get_tree(cli, tgt["out"], local)
        try:
            sftp_get(cli, tgt["log"], local / f"tri_{args.what}.log")
            print(f"[fetch] got tri_{args.what}.log")
        except Exception as e:
            print(f"[fetch] log skipped: {e}")
        print("[fetch] done")
    finally:
        cli.close()


def cmd_all(args):
    cmd_push(args)
    if args.phase == 2:
        # Sanity: warn if the cache_tri dir doesn't exist yet.
        cli = connect()
        try:
            rc, _, _ = run(cli,
                           f"test -d '{posixpath.join(REMOTE_TRI, 'cache_tri')}'",
                           check=False, print_stream=False)
            if rc != 0:
                print("[all] WARNING: cache_tri/ not found on server.")
                print("      Phase 2 pipeline will exit with 'cache dir missing'.")
                print("      Run `python auto_run_tri.py rebuild` first.")
        finally:
            cli.close()
    cmd_launch(args)
    here = Path(__file__).name
    print("\n[all] launched; poll with:")
    print(f"    python {here} status --what phase{args.phase}")
    print(f"    python {here} fetch  --what phase{args.phase}   # after flag appears")


# ──────────────────────────── argparse ────────────────────────────
def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="sub", required=True)

    sub.add_parser("push", help="SFTP-push sources + shell scripts to tri_structure/")

    launch = sub.add_parser("launch", help="Launch tri pipeline for a phase")
    launch.add_argument("--phase", type=int, choices=[1, 2], required=True)
    launch.add_argument("--epochs",  type=int, default=200)
    launch.add_argument("--repeats", type=int, default=3)
    launch.add_argument("--n_folds", type=int, default=5)

    rebuild = sub.add_parser("rebuild", help="Phase 2 prep: re-skeletonize masks")
    rebuild.add_argument("--raw_dir", default=None,
                         help="Override raw NIfTI root (e.g. if it has a trailing space)")
    rebuild.add_argument("--workers", type=int, default=8,
                         help="Pool size for parallel per-patient processing (default 8)")

    launch_v2 = sub.add_parser("launch_v2",
                               help="Launch v2 pipeline (attention pooling + signatures)")
    launch_v2.add_argument("--epochs",  type=int, default=200)
    launch_v2.add_argument("--repeats", type=int, default=3)
    launch_v2.add_argument("--n_folds", type=int, default=5)

    status = sub.add_parser("status", help="Tail log + process + flag")
    status.add_argument("--what", choices=["phase1", "phase2", "phase2_v2", "rebuild"], required=True)

    fetch = sub.add_parser("fetch", help="Pull remote output dir")
    fetch.add_argument("--what", choices=["phase1", "phase2", "phase2_v2", "rebuild"], required=True)
    fetch.add_argument("--local_dir", default=None)
    fetch.add_argument("--include_cache", action="store_true",
                       help="For rebuild: also pull the full cache_tri/ tree")

    a = sub.add_parser("all", help="push -> launch <phase>")
    a.add_argument("--phase", type=int, choices=[1, 2], required=True)
    a.add_argument("--epochs",  type=int, default=200)
    a.add_argument("--repeats", type=int, default=3)
    a.add_argument("--n_folds", type=int, default=5)

    return p


def main():
    args = build_parser().parse_args()
    handlers = {
        "push":      cmd_push,
        "launch":    cmd_launch,
        "launch_v2": cmd_launch_v2,
        "rebuild":   cmd_rebuild,
        "status":    cmd_status,
        "fetch":     cmd_fetch,
        "all":       cmd_all,
    }
    handlers[args.sub](args)


if __name__ == "__main__":
    main()
