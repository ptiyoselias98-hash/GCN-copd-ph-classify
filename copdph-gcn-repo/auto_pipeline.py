"""Autonomous end-to-end experiment pipeline for the COPD-PH GCN project.

Designed to run unattended even if the user's Claude balance runs out:
  * Each step is idempotent and resumable (state file).
  * Long-running remote jobs are polled with sparse sleeps (5 min) so the
    script can tolerate restart without wasting compute.
  * Failures of non-critical steps (local analyses) do not abort the pipeline.

Steps:
  1. attribution_analysis  (local, fast, ~1 min)
  2. improvement_experiments  (local, fast, ~2 min)
  3. generate mPAP-stratified splits  (local)
  4. generate mPAP lookup JSON  (local)
  5. push + launch sprint5 remote (3 arms, serial; node-drop + mpap aux)
  6. poll until sprint5 done (sparse 5-min sleeps)
  7. fetch sprint5 results
  8. final_comparison_report  (local, emits xlsx + png)

Usage (one-click):
  python auto_pipeline.py

Skip/repeat a step:
  python auto_pipeline.py --from-step 5
  python auto_pipeline.py --only-step 1
  python auto_pipeline.py --reset      # wipe state and restart

The state file at outputs/auto_pipeline_state.json tracks completed steps;
re-running picks up from the first incomplete step.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s pipeline: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("auto")

REPO = Path(__file__).resolve().parent
PROJECT = REPO.parent
OUTPUTS = PROJECT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

XLSX = PROJECT / "copd-ph患者113例0331.xlsx"
SPLITS_JSON = REPO / "data" / "splits_mpap_stratified.json"
MPAP_LOOKUP = REPO / "data" / "mpap_lookup.json"
STATE_FILE = OUTPUTS / "auto_pipeline_state.json"
RADIOMICS = PROJECT / "data" / "copd_ph_radiomics.csv"

PY = sys.executable


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"completed": []}


def save_state(st: dict) -> None:
    STATE_FILE.write_text(json.dumps(st, indent=2, ensure_ascii=False),
                          encoding="utf-8")


def run(cmd: list[str], cwd: Path | None = None,
        allow_fail: bool = False) -> int:
    logger.info("$ %s", " ".join(str(x) for x in cmd))
    try:
        r = subprocess.run(cmd, cwd=str(cwd or REPO))
        if r.returncode != 0:
            logger.error("step returned %d", r.returncode)
            if not allow_fail:
                raise SystemExit(r.returncode)
        return r.returncode
    except FileNotFoundError as e:
        logger.error("command not found: %s", e)
        if not allow_fail:
            raise
        return -1


# ---------------- Step definitions ----------------
def step1_attribution(st):
    """Local: attribution analysis. Non-critical; continue on failure."""
    outdir = OUTPUTS / "attribution"
    outdir.mkdir(exist_ok=True)
    run([PY, "attribution_analysis.py",
         "--xlsx", str(XLSX),
         "--output_dir", str(outdir)],
        allow_fail=True)


def step2_improvements(st):
    """Local: improvement experiments. Non-critical."""
    outdir = OUTPUTS / "improvements"
    outdir.mkdir(exist_ok=True)
    run([PY, "improvement_experiments.py",
         "--xlsx", str(XLSX),
         "--output_dir", str(outdir),
         "--radiomics", str(RADIOMICS)],
        allow_fail=True)


def step3_gen_splits(st):
    """Local: generate mPAP-stratified splits. Critical for step 5."""
    SPLITS_JSON.parent.mkdir(parents=True, exist_ok=True)
    # Labels CSV: we use the radiomics CSV's patient_id column as the valid
    # cohort (same pinyin convention). This is best-effort — the generator
    # degrades gracefully if intersection misses some ids.
    run([PY, "gen_mpap_stratified_splits.py",
         "--xlsx", str(XLSX),
         "--labels_csv", str(RADIOMICS),
         "--out", str(SPLITS_JSON)])


def step4_gen_mpap_lookup(st):
    """Local: generate {pid: mPAP} JSON for aux regression."""
    MPAP_LOOKUP.parent.mkdir(parents=True, exist_ok=True)
    run([PY, "gen_mpap_lookup.py",
         "--xlsx", str(XLSX),
         "--out", str(MPAP_LOOKUP)])


def step5_launch_sprint5(st):
    """Remote: push + nohup launch sprint5 (3 arms serial)."""
    if not SPLITS_JSON.exists():
        raise SystemExit("missing splits JSON; run step 3 first")
    if not MPAP_LOOKUP.exists():
        raise SystemExit("missing mpap lookup; run step 4 first")
    run([PY, "_remote_sprint5.py", "launch"])


def step6_poll_sprint5(st):
    """Remote: poll is_done every 5 minutes until sentinel appears.

    Tolerates up to 12 hours (144 polls). The remote launcher is fully
    autonomous; even if this local poller dies, the job continues. On
    restart, step 6 simply resumes polling.
    """
    max_polls = 288  # 24 hours @ 5 min
    for i in range(max_polls):
        try:
            r = subprocess.run([PY, "_remote_sprint5.py", "is_done"],
                               cwd=str(REPO), capture_output=True, text=True,
                               timeout=60)
            out = (r.stdout or "").strip()
            logger.info("poll %d/%d: %s", i + 1, max_polls, out or "??")
            if "DONE" in out:
                return
        except Exception as e:
            logger.warning("poll error: %s (will retry)", e)
        # periodic tail for context
        if i % 6 == 0:
            try:
                subprocess.run([PY, "_remote_sprint5.py", "status"],
                               cwd=str(REPO), timeout=60)
            except Exception:
                pass
        time.sleep(300)
    raise SystemExit("sprint5 polling exceeded 24h without DONE")


def step7_fetch_sprint5(st):
    run([PY, "_remote_sprint5.py", "fetch"])


def step8_final_report(st):
    run([PY, "final_comparison_report.py"], allow_fail=True)


STEPS = [
    ("step1_attribution",     step1_attribution),
    ("step2_improvements",    step2_improvements),
    ("step3_gen_splits",      step3_gen_splits),
    ("step4_gen_mpap_lookup", step4_gen_mpap_lookup),
    ("step5_launch_sprint5",  step5_launch_sprint5),
    ("step6_poll_sprint5",    step6_poll_sprint5),
    ("step7_fetch_sprint5",   step7_fetch_sprint5),
    ("step8_final_report",    step8_final_report),
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--from-step", type=int, default=0,
                   help="1-indexed step to start from; 0=resume from state")
    p.add_argument("--only-step", type=int, default=0,
                   help="run only this 1-indexed step and exit")
    p.add_argument("--reset", action="store_true",
                   help="wipe state and restart from step 1")
    args = p.parse_args()

    if args.reset and STATE_FILE.exists():
        STATE_FILE.unlink()
        logger.info("state reset")

    st = load_state()
    logger.info("state: completed=%s", st.get("completed", []))

    if args.only_step:
        idx = args.only_step - 1
        name, fn = STEPS[idx]
        logger.info(">>> only: %s", name)
        fn(st)
        st.setdefault("completed", [])
        if name not in st["completed"]:
            st["completed"].append(name)
        save_state(st)
        return 0

    start = max(0, args.from_step - 1) if args.from_step else 0
    for i, (name, fn) in enumerate(STEPS):
        if i < start:
            continue
        if name in st.get("completed", []) and i != start:
            logger.info("SKIP %s (already completed)", name)
            continue
        logger.info(">>> %s", name)
        try:
            fn(st)
        except SystemExit:
            raise
        except Exception as e:
            logger.exception("step %s failed: %s", name, e)
            # Step 1/2/8 are non-critical (allow_fail within); others are hard
            if name in ("step1_attribution", "step2_improvements",
                        "step8_final_report"):
                logger.warning("continuing despite %s failure", name)
            else:
                return 1
        st.setdefault("completed", [])
        if name not in st["completed"]:
            st["completed"].append(name)
        save_state(st)
    logger.info("pipeline complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
