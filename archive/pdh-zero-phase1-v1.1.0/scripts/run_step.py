#!/usr/bin/env python3
"""Execute one bounded experiment command in an immutable evidence directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pdh_phase1 import __version__ as package_version  # noqa: E402
from pdh_phase1.io_utils import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse wrapper arguments and the command after ``--``."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--gate", default="LOCAL", choices=("A", "B", "C", "LOCAL"))
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.experiment_id):
        parser.error("experiment id may contain only letters, digits, dot, underscore, and dash")
    if args.timeout_seconds <= 0:
        parser.error("timeout must be positive")
    return args


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for one regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(run_dir: Path) -> None:
    """Write deterministic checksums for every evidence file except the checksum file itself."""
    lines = []
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        if path.name == "checksums.sha256":
            continue
        lines.append(f"{sha256(path)}  {path.relative_to(run_dir).as_posix()}")
    (run_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def backend_name() -> str:
    """Return the active Torch backend without triggering a fallback claim."""
    if not torch.cuda.is_available():
        return "cpu"
    return "rocm" if torch.version.hip is not None else "cuda"


def git_commit() -> str | None:
    """Return the current commit when this package is inside a Git worktree."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> int:
    """Run the command with a timeout and preserve fresh stdout/stderr evidence."""
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_root = (PROJECT_ROOT / args.results_root).resolve()
    run_dir = results_root / f"{args.experiment_id}_{timestamp}"
    if run_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing run directory: {run_dir}")
    (run_dir / "artifacts").mkdir(parents=True)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    started = datetime.now(timezone.utc)
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "experiment_id": args.experiment_id,
        "gate": args.gate,
        "run_directory": str(run_dir),
        "command": args.command,
        "cwd": str(PROJECT_ROOT),
        "start_time_utc": started.isoformat(),
        "end_time_utc": None,
        "timeout_seconds": args.timeout_seconds,
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_version": package_version,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_hip": torch.version.hip,
        "backend": backend_name(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "git_commit": git_commit(),
        "exit_code": None,
        "timed_out": False,
        "status": "running",
    }
    write_json(run_dir / "manifest.json", manifest)
    environment = os.environ.copy()
    environment["PDH_RUN_DIR"] = str(run_dir)
    timed_out = False
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            args.command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
        )
        try:
            exit_code = process.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            exit_code = 124
    finished = datetime.now(timezone.utc)
    status = "timeout" if timed_out else ("success" if exit_code == 0 else "failed")
    manifest.update(
        {
            "end_time_utc": finished.isoformat(),
            "elapsed_seconds": (finished - started).total_seconds(),
            "exit_code": exit_code,
            "timed_out": timed_out,
            "status": status,
        }
    )
    write_json(run_dir / "manifest.json", manifest)
    write_checksums(run_dir)
    print(run_dir)
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
