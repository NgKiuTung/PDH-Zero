#!/usr/bin/env python3
"""Launch, inspect, resume, or stop the detached Gate B long campaign."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse long-job lifecycle commands."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    start = subparsers.add_parser("start")
    start.add_argument("--training-hours", type=float, default=7.5)
    start.add_argument("--chunk-updates", type=int, default=20)
    start.add_argument("--methods", default="plain,privileged")
    start.add_argument("--seeds", default="20260929,3407,9102")
    start.add_argument("--evaluation-wall-seeds", type=int, default=5000)
    start.add_argument("--device", default="cuda", choices=("auto", "cpu", "cuda", "rocm"))
    start.add_argument("--job-id")
    start.add_argument("--resume-job-dir")
    status = subparsers.add_parser("status")
    status.add_argument("--job-dir")
    stop = subparsers.add_parser("stop")
    stop.add_argument("--job-dir")
    return parser.parse_args()


def results_root() -> Path:
    """Return the configured Gate B long-run root."""
    base = os.environ.get("PDH_RESULTS_ROOT")
    if not base:
        raise RuntimeError("PDH_RESULTS_ROOT is unset; source the project env.sh first")
    root = Path(base) / "gate-b" / "long-runs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def latest_job(root: Path) -> Path:
    """Return the most recently modified long-job directory."""
    jobs = sorted((item for item in root.iterdir() if item.is_dir()), key=lambda item: item.stat().st_mtime)
    if not jobs:
        raise RuntimeError(f"No long jobs found under {root}")
    return jobs[-1]


def read_pid(job_dir: Path) -> int | None:
    """Read the detached campaign PID when available."""
    path = job_dir / "launcher.json"
    if not path.is_file():
        return None
    return int(json.loads(path.read_text(encoding="utf-8"))["pid"])


def pid_alive(pid: int | None) -> bool:
    """Return whether the process is live and not a zombie."""
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (OSError, FileNotFoundError):
        return False
    fields = stat.split()
    return len(fields) > 2 and fields[2] != "Z"


def start_job(args: argparse.Namespace) -> int:
    """Start a detached campaign that survives Terminal closure."""
    root = results_root()
    if args.resume_job_dir:
        job_dir = Path(args.resume_job_dir).resolve()
        state_path = job_dir / "state.json"
        if not state_path.is_file():
            raise RuntimeError(f"Resume state is missing: {job_dir}")
        stored = json.loads(state_path.read_text(encoding="utf-8"))["settings"]
        args.training_hours = float(stored["training_seconds"]) / 3600.0
        args.chunk_updates = int(stored["chunk_updates"])
        args.evaluation_wall_seeds = int(stored["evaluation_wall_seeds"])
        args.device = str(stored["device"])
        participant_values = json.loads(state_path.read_text(encoding="utf-8"))["participants"].values()
        args.methods = ",".join(dict.fromkeys(str(item["method"]) for item in participant_values))
        args.seeds = ",".join(str(item["seed"]) for item in participant_values if item["method"] == args.methods.split(",")[0])
        resume = True
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        job_id = args.job_id or f"B06_balanced_{args.training_hours:g}h_{timestamp}"
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", job_id):
            raise ValueError("job-id may contain only letters, digits, dot, underscore, and dash")
        job_dir = root / job_id
        job_dir.mkdir(parents=False, exist_ok=False)
        resume = False
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "13_long_gate_b_campaign.py"),
        "--job-dir",
        str(job_dir),
        "--training-hours",
        str(args.training_hours),
        "--chunk-updates",
        str(args.chunk_updates),
        "--methods",
        args.methods,
        "--seeds",
        args.seeds,
        "--evaluation-wall-seeds",
        str(args.evaluation_wall_seeds),
        "--device",
        args.device,
    ]
    if resume:
        command.append("--resume")
    stdout = (job_dir / "stdout.log").open("ab")
    stderr = (job_dir / "stderr.log").open("ab")
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
    )
    stdout.close()
    stderr.close()
    launcher = {
        "pid": process.pid,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "project_root": str(PROJECT_ROOT),
        "job_dir": str(job_dir),
        "resume": resume,
    }
    (job_dir / "launcher.json").write_text(json.dumps(launcher, indent=2) + "\n", encoding="utf-8")
    print(f"JOB_DIR={job_dir}")
    print(f"PID={process.pid}")
    print("Long campaign started in a detached process.")
    return 0


def show_status(args: argparse.Namespace) -> int:
    """Print the latest durable campaign state and process status."""
    root = results_root()
    job_dir = Path(args.job_dir).resolve() if args.job_dir else latest_job(root)
    pid = read_pid(job_dir)
    print(f"JOB_DIR={job_dir}")
    print(f"PID={pid}")
    print(f"PROCESS_ALIVE={pid_alive(pid)}")
    state = job_dir / "state.json"
    if state.is_file():
        print(state.read_text(encoding="utf-8"))
    else:
        print("state.json is not available yet")
    return 0


def stop_job(args: argparse.Namespace) -> int:
    """Request a checkpoint-safe stop from one detached campaign."""
    root = results_root()
    job_dir = Path(args.job_dir).resolve() if args.job_dir else latest_job(root)
    pid = read_pid(job_dir)
    if not pid_alive(pid):
        print(f"No live campaign process for {job_dir}")
        return 0
    assert pid is not None
    os.kill(pid, signal.SIGTERM)
    print(f"SIGTERM sent to PID {pid}; the current chunk will finish before exit.")
    return 0


def main() -> int:
    """Dispatch lifecycle commands."""
    args = parse_args()
    if args.command == "start":
        return start_job(args)
    if args.command == "status":
        return show_status(args)
    if args.command == "stop":
        return stop_job(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
