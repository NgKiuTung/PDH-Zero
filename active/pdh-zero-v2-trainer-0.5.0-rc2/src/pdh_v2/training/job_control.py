"""长训练任务的查重、后台启动、状态、停止与恢复。"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

from pdh_v2.training.long_config import LongRunConfig
from pdh_v2.training.long_runtime import atomic_json, read_json, utc_now
from pdh_v2.training.promotion import validate_smoke


def _tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def pid_alive(pid: int) -> bool:
    """检查 PID 是否存在；不扫描或终止其他 Python 进程。"""
    if pid < 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def job_process_alive(pid: int, job_dir: Path) -> bool:
    """验证 PID 仍是绑定到该目录的长任务，防止 PID 复用误伤其他进程。"""
    if not pid_alive(pid):
        return False
    command = Path(f"/proc/{pid}/cmdline")
    if not command.exists():
        return True
    try:
        arguments = command.read_bytes().split(b"\0")
    except OSError:
        return False
    expected = str(job_dir.resolve()).encode()
    return b"_worker" in arguments and expected in arguments


def _lock_path(pdh_home: Path) -> Path:
    return pdh_home / "state" / "v2-long-job.lock.json"


def _archive_stale_lock(path: Path) -> None:
    if not path.exists():
        return
    target = path.parent / "stale" / f"{path.stem}-{_tag()}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(path, target)


def _active_lock(pdh_home: Path) -> dict[str, Any] | None:
    """返回当前活动任务；失效锁会归档而不会阻断新任务。"""
    path = _lock_path(pdh_home)
    if not path.exists():
        return None
    lock = read_json(path)
    pid = int(lock.get("pid", 0))
    job_dir = Path(str(lock.get("job_dir", "")))
    reserved = lock.get("status") == "reserved"
    alive = pid_alive(pid) if reserved else job_process_alive(pid, job_dir)
    if alive:
        return {
            "status": "already_starting" if reserved else "already_running",
            "job_dir": str(job_dir),
            "pid": pid,
        }
    _archive_stale_lock(path)
    return None


def _reserve_lock(pdh_home: Path, job_dir: Path) -> Path:
    path = _lock_path(pdh_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        lock = read_json(path)
        pid = int(lock.get("pid", 0))
        locked_job = Path(str(lock.get("job_dir", "")))
        reserved = lock.get("status") == "reserved"
        alive = pid_alive(pid) if reserved else job_process_alive(pid, locked_job)
        if alive:
            raise RuntimeError(f"active long job already exists: {locked_job}")
        _archive_stale_lock(path)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(
            {
                "schema_version": 1,
                "status": "reserved",
                "job_dir": str(job_dir),
                "pid": os.getpid(),
                "created_at_utc": utc_now(),
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )
        stream.write("\n")
    return path


def _spawn(script: Path, job_dir: Path, resume: bool) -> subprocess.Popen[str]:
    root = script.resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root)))
    arguments = [sys.executable, str(script), "_worker", "--job-dir", str(job_dir)]
    if resume:
        arguments.append("--resume")
    with (job_dir / "stdout.log").open("a", encoding="utf-8") as stdout, (
        job_dir / "stderr.log"
    ).open("a", encoding="utf-8") as stderr:
        return subprocess.Popen(
            arguments,
            cwd=root,
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )


def launch_new(
    script: Path,
    pdh_home: Path,
    smoke_dir: Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """验证 Smoke 后创建并后台启动一个新的长任务。"""
    active = _active_lock(pdh_home)
    if active is not None:
        return active
    smoke = validate_smoke(smoke_dir)
    output_root = pdh_home / "evidence" / "v2-long-runs"
    output_root.mkdir(parents=True, exist_ok=True)
    job_dir = output_root / f"V2_BASELINE_48H_{_tag()}"
    lock_path = _reserve_lock(pdh_home, job_dir)
    try:
        job_dir.mkdir(parents=True, exist_ok=False)
        values: dict[str, Any] = {
            "output": job_dir,
            "smoke_dir": smoke.smoke_dir,
            "hidden_size": smoke.hidden_size,
        }
        values.update(overrides or {})
        config = LongRunConfig(**values)
        atomic_json(job_dir / "config.json", config.to_dict())
        process = _spawn(script, job_dir, resume=False)
        time.sleep(1.0)
        if process.poll() is not None:
            error = (job_dir / "stderr.log").read_text(encoding="utf-8")[-4000:]
            raise RuntimeError(f"long worker exited during startup: {error}")
        atomic_json(
            job_dir / "launcher.json",
            {
                "schema_version": 1,
                "created_at_utc": utc_now(),
                "pid": process.pid,
                "job_dir": str(job_dir),
                "smoke_dir": str(smoke.smoke_dir),
                "config": config.to_dict(),
            },
        )
        atomic_json(
            lock_path,
            {
                "schema_version": 1,
                "status": "running",
                "job_dir": str(job_dir),
                "pid": process.pid,
                "created_at_utc": utc_now(),
            },
        )
        current = pdh_home / "state" / "current_v2_long_job.txt"
        current.write_text(str(job_dir) + "\n", encoding="utf-8")
    except BaseException:
        lock_path.unlink(missing_ok=True)
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    return {"status": "started", "job_dir": str(job_dir), "pid": process.pid}


def job_status(job_dir: Path) -> dict[str, Any]:
    """读取任务状态并以根 PID 是否存活作交叉验证。"""
    launcher = read_json(job_dir / "launcher.json")
    pid = int(launcher["pid"])
    state_path = job_dir / "state.json"
    state = read_json(state_path) if state_path.exists() else {"status": "starting"}
    return {
        **state,
        "job_dir": str(job_dir),
        "pid": pid,
        "process_alive": job_process_alive(pid, job_dir),
    }


def stop_job(job_dir: Path) -> dict[str, Any]:
    """向该任务根 PID 发送 SIGTERM；不使用全局 pkill。"""
    launcher = read_json(job_dir / "launcher.json")
    pid = int(launcher["pid"])
    if not job_process_alive(pid, job_dir):
        return {"status": "not_running", "job_dir": str(job_dir), "pid": pid}
    os.kill(pid, signal.SIGTERM)
    return {"status": "stop_requested", "job_dir": str(job_dir), "pid": pid}


def resume_job(script: Path, pdh_home: Path, job_dir: Path) -> dict[str, Any]:
    """从本任务最新完整更新边界检查点继续。"""
    status = job_status(job_dir)
    if status["process_alive"]:
        raise RuntimeError("job is already running")
    if status.get("status") == "completed":
        raise RuntimeError("completed job cannot be resumed")
    checkpoint = job_dir / "checkpoints" / "checkpoint_latest.pt"
    if not checkpoint.is_file():
        raise RuntimeError("long checkpoint missing")
    lock_path = _reserve_lock(pdh_home, job_dir)
    try:
        process = _spawn(script, job_dir, resume=True)
        time.sleep(1.0)
        if process.poll() is not None:
            error = (job_dir / "stderr.log").read_text(encoding="utf-8")[-4000:]
            raise RuntimeError(f"long worker exited during resume: {error}")
        launcher = read_json(job_dir / "launcher.json")
        launcher.update({"pid": process.pid, "resumed_at_utc": utc_now()})
        atomic_json(job_dir / "launcher.json", launcher)
        atomic_json(
            lock_path,
            {
                "schema_version": 1,
                "status": "running",
                "job_dir": str(job_dir),
                "pid": process.pid,
                "created_at_utc": utc_now(),
            },
        )
    except BaseException:
        lock_path.unlink(missing_ok=True)
        raise
    return {"status": "resumed", "job_dir": str(job_dir), "pid": process.pid}
