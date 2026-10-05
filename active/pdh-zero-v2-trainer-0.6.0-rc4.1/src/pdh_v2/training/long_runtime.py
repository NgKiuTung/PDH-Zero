"""长任务的原子状态、资源检查、信号和 GPU 遥测。"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
from typing import Any, IO


def utc_now() -> str:
    """返回带时区的 UTC 时间。"""
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """原子替换 JSON 状态文件，避免中断留下半文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    """读取对象型 JSON。"""
    return dict(json.loads(path.read_text(encoding="utf-8")))


def gpu_processes() -> tuple[int, ...]:
    """查询 GPU 计算 PID；查询失败时抛错而不是假定设备空闲。"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("unable to query GPU processes") from exc
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    return tuple(int(line.strip()) for line in result.stdout.splitlines() if line.strip())


def available_bytes(path: Path) -> int:
    """返回任务所在文件系统可用字节数。"""
    return int(shutil.disk_usage(path).free)


class StopFlag:
    """SIGTERM/SIGINT 只请求在当前 PPO 更新边界安全停止。"""

    def __init__(self) -> None:
        self.requested = False
        self.signal_number: int | None = None

    def install(self) -> None:
        """安装当前进程的安全停止处理器。"""
        signal.signal(signal.SIGTERM, self._handle)
        signal.signal(signal.SIGINT, self._handle)

    def _handle(self, number: int, _frame: object) -> None:
        self.requested = True
        self.signal_number = number


class Telemetry:
    """仅管理本任务启动的 nvidia-smi 子进程。"""

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.process: subprocess.Popen[str] | None = None
        self.stdout: IO[str] | None = None
        self.stderr: IO[str] | None = None

    def start(self) -> None:
        """每 10 秒记录 GPU 温度、功耗、利用率和显存。"""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.stdout = self.csv_path.open("a", encoding="utf-8")
        self.stderr = self.csv_path.with_suffix(".stderr").open("a", encoding="utf-8")
        query = (
            "timestamp,name,temperature.gpu,power.draw,utilization.gpu,"
            "utilization.memory,memory.used,memory.total,clocks.sm,clocks.mem"
        )
        self.process = subprocess.Popen(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv",
                "-l",
                "10",
            ],
            stdout=self.stdout,
            stderr=self.stderr,
            text=True,
        )

    def stop(self) -> None:
        """停止且只停止本实例创建的遥测子进程。"""
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if self.stdout is not None:
            self.stdout.close()
        if self.stderr is not None:
            self.stderr.close()
