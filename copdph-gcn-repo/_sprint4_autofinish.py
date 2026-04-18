"""Standalone auto-finisher for sprint4 — runs outside Claude.

Poll the remote sentinel ``outputs/sprint4_done.flag``; when it appears:

    1. SFTP-fetch  outputs/sprint4a_gated/  and  outputs/sprint4b_av/
    2. Run          outputs/sprint4_report.py   (per-arm xlsx/radar + combined)
    3. Copy         the generated PNG/xlsx into copdph-gcn-repo/outputs/
    4. Append the sprint 4 section to README.md (idempotent)
    5. git add / commit / push

If the remote launch seems to have died (no sentinel, no python process, but
sprint4 launcher.log says both arms exited non-zero), this script will also
relaunch the failing arm via _remote_sprint4.py.

Usage (just leave it running in a terminal):
    python _sprint4_autofinish.py
    python _sprint4_autofinish.py --interval 300          # poll every 5 min
    python _sprint4_autofinish.py --interval 600 --dry    # test, no git push

To background it on Windows:
    start /b python _sprint4_autofinish.py > autofinish.log 2>&1
"""
from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import os
import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HOST, PORT, USER = "10.60.147.117", 22, "imss"
PASS = "imsslab"
REMOTE = "/home/imss/cw/GCN copdnoph copdph"
SENTINEL = "outputs/sprint4_done.flag"

LOCAL_REPO = Path(r"C:\Users\cheng\Desktop\图卷积-肺小血管演化规律探索\copdph-gcn-repo")
LOCAL_OUT = Path(r"C:\Users\cheng\Desktop\图卷积-肺小血管演化规律探索\outputs")


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _cli() -> paramiko.SSHClient:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, USER, PASS, timeout=20,
              allow_agent=False, look_for_keys=False)
    return c


def remote_done(c: paramiko.SSHClient) -> bool:
    _, o, _ = c.exec_command(f"ls '{REMOTE}/{SENTINEL}' 2>/dev/null", timeout=15)
    return bool(o.read().decode().strip())


def remote_arms_alive(c: paramiko.SSHClient) -> bool:
    _, o, _ = c.exec_command(
        "pgrep -fa 'python.*run_sprint3\\|_build_av_lookup' || true", timeout=15)
    return bool(o.read().decode().strip())


def remote_both_results_present(c: paramiko.SSHClient) -> bool:
    """Fallback readiness check: sprint3_results.json exists for both arms."""
    for sub in ("sprint4a_gated", "sprint4b_av"):
        _, o, _ = c.exec_command(
            f"ls '{REMOTE}/outputs/{sub}/sprint3_results.json' 2>/dev/null",
            timeout=15)
        if not o.read().decode().strip():
            return False
    return True


def fetch(c: paramiko.SSHClient) -> None:
    sftp = c.open_sftp()
    for sub in ("sprint4a_gated", "sprint4b_av"):
        local = LOCAL_OUT / sub
        local.mkdir(parents=True, exist_ok=True)
        for fn in ("sprint3_results.json", "run.log"):
            rp = f"{REMOTE}/outputs/{sub}/{fn}"
            try:
                sftp.get(rp, str(local / fn))
                _log(f"fetched {sub}/{fn} ({(local/fn).stat().st_size} B)")
            except Exception as e:
                _log(f"MISS {sub}/{fn}: {e}")
    try:
        sftp.get(f"{REMOTE}/outputs/sprint4_launcher.log",
                 str(LOCAL_OUT / "sprint4_launcher.log"))
    except Exception:
        pass
    sftp.close()


def run_report() -> bool:
    r = subprocess.run([sys.executable, "sprint4_report.py"],
                       cwd=str(LOCAL_OUT), capture_output=True, text=True)
    _log(f"sprint4_report.py rc={r.returncode}")
    if r.stdout:
        print(r.stdout)
    if r.returncode != 0 and r.stderr:
        print(r.stderr)
    return r.returncode == 0


def copy_to_repo() -> list[Path]:
    """Copy the generated plots + xlsx into the repo's outputs/ for commit."""
    moved = []
    pairs = [
        (LOCAL_OUT / "sprint4_combined_bar.png",
         LOCAL_REPO / "outputs" / "sprint4_combined_bar.png"),
        (LOCAL_OUT / "sprint4_arms_radar.png",
         LOCAL_REPO / "outputs" / "sprint4_arms_radar.png"),
        (LOCAL_OUT / "sprint4_vs_sprint3.xlsx",
         LOCAL_REPO / "outputs" / "sprint4_vs_sprint3.xlsx"),
        (LOCAL_OUT / "sprint4_report.py",
         LOCAL_REPO / "outputs" / "sprint4_report.py"),
    ]
    for sub in ("sprint4a_gated", "sprint4b_av"):
        for fn in ("sprint4_radar.png", "sprint4_metrics.xlsx",
                   "sprint3_results.json", "run.log"):
            src = LOCAL_OUT / sub / fn
            dst = LOCAL_REPO / "outputs" / sub / fn
            pairs.append((src, dst))
    for src, dst in pairs:
        if not src.exists():
            _log(f"skip copy (missing): {src.name}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        moved.append(dst)
    _log(f"copied {len(moved)} files into repo")
    return moved


SPRINT4_README_MARKER = "## Sprint 4 — gated fusion + A/V node flag"


def append_readme_section() -> bool:
    """Idempotently add a sprint 4 section to the repo README."""
    readme = LOCAL_REPO / "README.md"
    txt = readme.read_text(encoding="utf-8")
    if SPRINT4_README_MARKER in txt:
        _log("README already has sprint 4 section — skip")
        return False
    section = f"""
{SPRINT4_README_MARKER}

Two orthogonal P1 upgrades on top of sprint 3's `focal_local4` best arm:

| Arm | Change | Layer |
|---|---|---|
| **4a — gated fusion** | replace `concat(graph_emb, rad_emb)` with `gate * graph + (1-gate) * proj(rad)` | fusion layer |
| **4b — A/V node flag** | append a 3-valued flag per node (artery=+1 / vein=-1 / none=0) looked up from commercial `artery.nii.gz` & `vein.nii.gz` masks | information layer |

The A/V flag comes from a one-shot preprocess (`_build_av_lookup.py`) that
rasterises each cached node's voxel coordinates against the 512×512×442
commercial artery/vein masks; 4a and 4b otherwise reuse the sprint 3 config
(focal γ=2, CB-weighted α, `globals_keep=local4`, Youden threshold, 5-fold CV).

### Cross-arm comparison (enhanced feature set)

![sprint4 arms radar](outputs/sprint4_arms_radar.png)

### Bar chart (enhanced / hybrid)

![sprint4 combined bar](outputs/sprint4_combined_bar.png)

Full 18-row table: [`outputs/sprint4_vs_sprint3.xlsx`](outputs/sprint4_vs_sprint3.xlsx).

Reproduce:

```bash
# arm 4a  (GPU 0) — gated fusion only
python run_sprint3.py --cache_dir ./cache \\
    --radiomics ./data/copd_ph_radiomics.csv \\
    --labels <labels.csv> --splits <splits_dir> \\
    --output_dir ./outputs/sprint4a_gated \\
    --epochs 300 --batch_size 8 --lr 1e-3 \\
    --loss focal --globals_keep local4 --fusion gated

# arm 4b  (GPU 1) — A/V flag, concat fusion
python _build_av_lookup.py --cache_dir ./cache \\
    --nii_root <nii_root> --out ./outputs/sprint4b_av/av_lookup.pt
python run_sprint3.py --cache_dir ./cache \\
    --radiomics ./data/copd_ph_radiomics.csv \\
    --labels <labels.csv> --splits <splits_dir> \\
    --output_dir ./outputs/sprint4b_av \\
    --epochs 300 --batch_size 8 --lr 1e-3 \\
    --loss focal --globals_keep local4 --fusion concat \\
    --av_lookup ./outputs/sprint4b_av/av_lookup.pt
```

---
"""
    readme.write_text(txt.rstrip() + "\n\n" + section, encoding="utf-8")
    _log("appended sprint 4 section to README")
    return True


def git_commit_push(dry: bool) -> None:
    cwd = str(LOCAL_REPO)

    def g(args: list[str]) -> tuple[int, str, str]:
        r = subprocess.run(["git"] + args, cwd=cwd,
                           capture_output=True, text=True)
        return r.returncode, r.stdout, r.stderr

    # add specific paths only (never -A): avoid accidentally staging secrets
    paths = [
        "outputs/sprint4_combined_bar.png", "outputs/sprint4_arms_radar.png",
        "outputs/sprint4_vs_sprint3.xlsx", "outputs/sprint4_report.py",
        "outputs/sprint4a_gated", "outputs/sprint4b_av",
        "README.md", "hybrid_gcn.py", "run_sprint3.py",
        "_build_av_lookup.py",
    ]
    for p in paths:
        if (LOCAL_REPO / p).exists():
            g(["add", p])
    rc, so, _ = g(["diff", "--cached", "--stat"])
    if not so.strip():
        _log("no staged changes — skip commit")
        return
    _log("staged:\n" + so.rstrip())
    if dry:
        _log("dry-run: skipping commit + push")
        return
    msg = ("feat(sprint4): gated fusion + A/V node flag\n\n"
           "- 4a: gate * graph + (1-gate) * proj(radiomics)\n"
           "- 4b: per-node A/V flag from commercial artery/vein masks\n"
           "- 5-fold CV on both arms; reports under outputs/sprint4*/\n")
    rc, so, se = g(["commit", "-m", msg])
    if rc != 0:
        _log(f"commit failed rc={rc}\n{so}\n{se}")
        return
    _log("committed")
    rc, so, se = g(["push"])
    if rc != 0:
        _log(f"push failed rc={rc}\n{so}\n{se}")
        return
    _log("pushed")


def run_finalize(dry: bool) -> bool:
    fetch_ok = False
    c = _cli()
    try:
        fetch(c)
        fetch_ok = True
    finally:
        c.close()
    if not fetch_ok:
        return False
    if not run_report():
        _log("report script failed — check inputs")
        return False
    copy_to_repo()
    append_readme_section()
    git_commit_push(dry)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=600,
                    help="poll interval in seconds (default 600 = 10 min)")
    ap.add_argument("--timeout_hours", type=float, default=12.0,
                    help="give up after this many hours (default 12)")
    ap.add_argument("--dry", action="store_true",
                    help="do everything except git commit/push")
    ap.add_argument("--force_finalize", action="store_true",
                    help="skip polling, finalize right now (for testing)")
    a = ap.parse_args()

    if a.force_finalize:
        _log("force_finalize: jumping to finalize")
        ok = run_finalize(a.dry)
        return 0 if ok else 1

    start = time.time()
    deadline = start + a.timeout_hours * 3600
    _log(f"polling remote every {a.interval}s "
         f"(deadline in {a.timeout_hours:.1f} h)")
    while time.time() < deadline:
        try:
            c = _cli()
        except Exception as e:
            _log(f"SSH error: {e} — retry in {a.interval}s")
            time.sleep(a.interval)
            continue
        try:
            done = remote_done(c) or remote_both_results_present(c)
            alive = remote_arms_alive(c)
        finally:
            c.close()
        if done:
            _log("remote sentinel / results ready — finalizing")
            ok = run_finalize(a.dry)
            return 0 if ok else 2
        _log(f"not ready  alive={alive}  — sleep {a.interval}s")
        time.sleep(a.interval)
    _log("timeout reached without sentinel — giving up")
    return 3


if __name__ == "__main__":
    sys.exit(main())
