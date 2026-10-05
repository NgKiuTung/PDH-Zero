"""项目内部CPU环境并行；spawn隔离CUDA，只传合法决策投影和私有终局牌谱。"""
from __future__ import annotations

from dataclasses import replace
import multiprocessing as mp
from multiprocessing.connection import Connection
import os
import time
from typing import Callable

from pdh_v2.rollout.vector_env import BatchedSessions, EnvDecision, TerminalHand


def _worker(connection: Connection, seeds: tuple[int, ...]) -> None:
    """每个进程独占一组环境；不创建GPU上下文或修改模型。"""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        env = BatchedSessions(len(seeds), 0, seeds)
        connection.send((True, os.getpid()))
        while True:
            command, data = connection.recv()
            if command == "close":
                break
            if command == "requests":
                result = env.requests()
            elif command == "reset":
                env.reset(data)
                result = None
            elif command == "reactivate":
                env.reactivate_all()
                result = None
            elif command == "heuristic":
                requests = {r.env_index: r for r in env.requests()}
                result = env.heuristic_actions(tuple(requests[i] for i in data))
            elif command == "step":
                choices, drain = data
                result = (env.step(choices, drain), env.all_inactive())
            elif command == "snapshot":
                from pdh_v2.rollout.snapshot import snapshot_batched
                result = snapshot_batched(env)
            else:
                raise ValueError(f"unknown worker command: {command}")
            connection.send((True, result))
    except EOFError:
        pass
    except BaseException as exc:
        try:
            connection.send((False, (type(exc).__name__, str(exc))))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class ParallelSessions(BatchedSessions):
    """同一任务内部的一组CPU子进程；由调用方close，不能用全局pkill。"""

    def __init__(
        self, count: int, seed: int, workers: int = 4,
        seeds: tuple[int, ...] | None = None, timeout: float = 120.0,
        check_stop: Callable[[], None] | None = None,
    ) -> None:
        if count < 1 or workers < 1 or timeout <= 0:
            raise ValueError("count, workers and timeout must be positive")
        keys = seeds or tuple(seed + i * 1_000_003 for i in range(count))
        if len(keys) != count:
            raise ValueError("seed count mismatch")
        self._count = count
        self._timeout = timeout
        self._check_stop = check_stop
        self._pipes: list[Connection] = []
        self._processes: list[mp.Process] = []
        self._offsets: list[int] = []
        self._sizes: list[int] = []
        self._inactive = [False] * min(workers, count)
        self._closed = False
        context = mp.get_context("spawn")
        worker_count = min(workers, count)
        try:
            for i in range(worker_count):
                lo, hi = count * i // worker_count, count * (i + 1) // worker_count
                parent, child = context.Pipe()
                process = context.Process(target=_worker, args=(child, keys[lo:hi]))
                process.start()
                child.close()
                self._pipes.append(parent)
                self._processes.append(process)
                self._offsets.append(lo)
                self._sizes.append(hi - lo)
            for i in range(worker_count):
                self._receive(i)
        except BaseException:
            self.close()
            raise

    @property
    def count(self) -> int:
        """返回总牌桌数，不是进程数。"""
        return self._count

    @property
    def pids(self) -> tuple[int, ...]:
        """提供精确子进程身份用于运行清单。"""
        return tuple(p.pid for p in self._processes if p.pid is not None)

    def _receive(self, index: int):
        deadline = time.monotonic() + self._timeout
        pipe = self._pipes[index]
        while not pipe.poll(0.2):
            if self._check_stop is not None:
                self._check_stop()
            if not self._processes[index].is_alive():
                raise RuntimeError(f"environment worker exited: {self._processes[index].pid}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"environment worker timeout: {self._processes[index].pid}")
        ok, data = pipe.recv()
        if not ok:
            raise RuntimeError(f"environment worker failure: {data}")
        return data

    def _call(self, command: str, data: list):
        if self._closed:
            raise RuntimeError("environment backend is closed")
        for pipe, item in zip(self._pipes, data):
            pipe.send((command, item))
        return [self._receive(i) for i in range(len(self._pipes))]

    def reactivate_all(self) -> None:
        """只在上批排空之后继续同一会话的已发下一手。"""
        self._call("reactivate", [None] * len(self._pipes))
        self._inactive = [False] * len(self._pipes)

    def requests(self) -> tuple[EnvDecision, ...]:
        """返回与串行相同顺序的合法Actor投影，不传GameSession真值给Actor。"""
        results = self._call("requests", [None] * len(self._pipes))
        return tuple(replace(r, env_index=r.env_index + offset)
                     for offset, batch in zip(self._offsets, results) for r in batch)

    def heuristic_actions(self, requests: tuple[EnvDecision, ...]) -> dict[int, int]:
        """启发式在各环境所属CPU进程计算。"""
        selected = {r.env_index for r in requests}
        arguments = [tuple(i - lo for i in sorted(selected) if lo <= i < lo + n)
                     for lo, n in zip(self._offsets, self._sizes)]
        batches = self._call("heuristic", arguments)
        return {i + offset: choice for offset, batch in zip(self._offsets, batches)
                for i, choice in batch.items()}

    def step(self, choices: dict[int, int], drain: bool = False) -> tuple[TerminalHand, ...]:
        """批量推进；全局桌号稳定，不由进程完成顺序决定随机流。"""
        if any(type(i) is not int or not 0 <= i < self.count for i in choices):
            raise ValueError("invalid environment index")
        arguments = [({i - lo: a for i, a in choices.items() if lo <= i < lo + n}, drain)
                     for lo, n in zip(self._offsets, self._sizes)]
        batches = self._call("step", arguments)
        self._inactive = [inactive for _, inactive in batches]
        return tuple(replace(t, env_index=t.env_index + offset)
                     for offset, (batch, _) in zip(self._offsets, batches) for t in batch)

    def all_inactive(self) -> bool:
        """所有进行中手局均已排空后才允许变更模型。"""
        return all(self._inactive)

    def snapshot(self) -> tuple[dict, ...]:
        """只在完整更新边界保存会话；不伪造进行中恢复。"""
        return tuple(s for batch in self._call("snapshot", [None] * len(self._pipes))
                     for s in batch)

    def close(self) -> None:
        """回收本对象创建的进程；不触碰其他Python或GPU任务。"""
        if self._closed:
            return
        self._closed = True
        for pipe in self._pipes:
            try:
                pipe.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self._processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
            if process.is_alive():
                process.kill()
                process.join(timeout=3)
        for pipe in self._pipes:
            pipe.close()

    def reset(self, seeds: tuple[int, ...]) -> None:
        """同一批CPU进程复用新评估种子，不每64手重新启动解释器。"""
        if len(seeds) != self.count or not self.all_inactive():
            raise ValueError("reset requires inactive environments and matching seeds")
        self._call("reset", [seeds[lo:lo+n] for lo, n in zip(self._offsets, self._sizes)])
        self._inactive = [False] * len(self._pipes)
