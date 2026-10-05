"""长训练更新循环；每次提交均位于完整更新边界。"""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import torch

from pdh_v2.policy import CandidateActorCritic
from pdh_v2.rollout import BatchedSessions
from pdh_v2.training.collector import RolloutCollector
from pdh_v2.training.events import JsonlWriter
from pdh_v2.training.long_artifacts import (
    _atomic_copy,
    _ensure_finite,
    _evaluate,
    _metric_payload,
    _save,
    _state,
    _write_update_metric,
)
from pdh_v2.training.long_config import LongRunConfig
from pdh_v2.training.long_runtime import StopFlag, atomic_json, available_bytes, utc_now
from pdh_v2.training.ppo import PPOConfig, update_ppo
from pdh_v2.training.replay_updates import write_replay_update


class _LongTrainingLoop:
    """管理预算门禁、更新提交、评估、快照和失败现场。"""

    def __init__(
        self,
        config: LongRunConfig,
        model: CandidateActorCritic,
        optimizer: torch.optim.Optimizer,
        environments: BatchedSessions,
        update: int,
        counters: dict[str, Any],
        device: torch.device,
        collector: RolloutCollector,
        ppo_config: PPOConfig,
        events: JsonlWriter,
        live_metrics: JsonlWriter,
        stop: StopFlag,
    ) -> None:
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.environments = environments
        self.update = update
        self.counters = counters
        self.device = device
        self.collector = collector
        self.ppo_config = ppo_config
        self.events = events
        self.live_metrics = live_metrics
        self.stop = stop
        self.attempt_started = time.monotonic()

    def _commit_elapsed(self) -> None:
        now = time.monotonic()
        elapsed = now - self.attempt_started
        self.counters["consumed_training_seconds"] = (
            float(self.counters["consumed_training_seconds"]) + elapsed
        )
        self.attempt_started = now

    def initialize_evaluation(self) -> None:
        """在首个更新前建立固定基线和第一个完整恢复点。"""
        if int(self.counters["next_evaluation_decisions"]) != 0:
            return
        evaluation = _evaluate(self.config, self.model, self.device, 0)
        self.counters["best_evaluation_mean_score"] = evaluation["mean_score"]
        self.counters["next_evaluation_decisions"] = (
            self.config.evaluation_interval_decisions
        )
        latest = self._save()
        _atomic_copy(
            latest,
            self.config.output / "checkpoints" / "checkpoint_best.pt",
        )

    def _save(self) -> Path:
        return _save(
            self.config,
            self.model,
            self.optimizer,
            self.environments,
            self.update,
            self.counters,
        )

    def _budget_stop_reason(self) -> tuple[str | None, int]:
        active_elapsed = (
            float(self.counters["consumed_training_seconds"])
            + time.monotonic()
            - self.attempt_started
        )
        free = available_bytes(self.config.output)
        if int(self.counters["long_decisions"]) >= self.config.target_decisions:
            return "target_decisions_reached", free
        if active_elapsed >= self.config.max_wall_seconds:
            return "max_wall_seconds_reached", free
        if self.stop.requested:
            return f"signal_{self.stop.signal_number}_requested", free
        if free < self.config.min_free_bytes:
            return "disk_free_space_gate", free
        return None, free

    def _train_update(self) -> tuple[Any, Any, float, float]:
        collection_started = time.monotonic()
        collected = self.collector.collect()
        collection_seconds = time.monotonic() - collection_started
        update_started = time.monotonic()
        trained = update_ppo(
            self.model,
            self.optimizer,
            collected.episodes,
            self.device,
            self.ppo_config,
            self.config.seed + self.update,
        )
        update_seconds = time.monotonic() - update_started
        return collected, trained, collection_seconds, update_seconds

    def _commit_update(
        self,
        collected: Any,
        trained: Any,
        collection_seconds: float,
        update_seconds: float,
        free: int,
    ) -> Path:
        self.update += 1
        self.counters["long_updates"] = int(self.counters["long_updates"]) + 1
        self.counters["long_hands"] = (
            int(self.counters["long_hands"]) + collected.hands
        )
        self.counters["long_decisions"] = (
            int(self.counters["long_decisions"]) + collected.decisions
        )
        self._commit_elapsed()
        raw_total = sum(collected.raw_score_sum)
        if raw_total != 0:
            raise RuntimeError(f"non-zero-sum collection: {raw_total}")
        replay = write_replay_update(
            self.config.output / "replays",
            collected.terminals,
            self.update,
        )
        record = _metric_payload(
            self.update,
            self.counters,
            collected,
            trained,
            collection_seconds,
            update_seconds,
            free,
            replay,
        )
        if self.device.type == "cuda":
            record["cuda_peak_memory_bytes"] = torch.cuda.max_memory_allocated(
                self.device
            )
            torch.cuda.reset_peak_memory_stats(self.device)
        _ensure_finite(record)
        _write_update_metric(self.config.output, self.update, record)
        latest = self._save()
        self.live_metrics.write(record)
        self.events.write({"event": "update_committed", **record})
        return latest

    def _snapshot_if_due(self, latest: Path) -> Path:
        decisions = int(self.counters["long_decisions"])
        next_snapshot = int(self.counters["next_snapshot_decisions"])
        if decisions < next_snapshot:
            return latest
        snapshot = (
            self.config.output
            / "checkpoints"
            / "snapshots"
            / f"checkpoint-d{decisions:012d}.pt"
        )
        _atomic_copy(latest, snapshot)
        while decisions >= int(self.counters["next_snapshot_decisions"]):
            self.counters["next_snapshot_decisions"] = (
                int(self.counters["next_snapshot_decisions"])
                + self.config.snapshot_interval_decisions
            )
        return self._save()

    def _evaluation_if_due(self, latest: Path) -> Path:
        decisions = int(self.counters["long_decisions"])
        next_evaluation = int(self.counters["next_evaluation_decisions"])
        if decisions < next_evaluation:
            return latest
        evaluation = _evaluate(self.config, self.model, self.device, decisions)
        best = self.counters.get("best_evaluation_mean_score")
        improved = best is None or float(evaluation["mean_score"]) > float(best)
        if improved:
            self.counters["best_evaluation_mean_score"] = evaluation["mean_score"]
        while decisions >= int(self.counters["next_evaluation_decisions"]):
            self.counters["next_evaluation_decisions"] = (
                int(self.counters["next_evaluation_decisions"])
                + self.config.evaluation_interval_decisions
            )
        latest = self._save()
        if improved:
            _atomic_copy(
                latest,
                self.config.output / "checkpoints" / "checkpoint_best.pt",
            )
        self.events.write({"event": "evaluation_completed", **evaluation})
        return latest

    def run(self) -> str:
        """运行到预算、人工停止或磁盘门禁，并返回停止原因。"""
        self.initialize_evaluation()
        while True:
            stop_reason, free = self._budget_stop_reason()
            if stop_reason is not None:
                return stop_reason
            collected, trained, collection_seconds, update_seconds = (
                self._train_update()
            )
            latest = self._commit_update(
                collected,
                trained,
                collection_seconds,
                update_seconds,
                free,
            )
            latest = self._snapshot_if_due(latest)
            self._evaluation_if_due(latest)
            atomic_json(
                self.config.output / "state.json",
                _state(
                    self.config,
                    "running",
                    self.update,
                    self.counters,
                    None,
                ),
            )

    def finish(self, stop_reason: str) -> int:
        """在完整更新边界提交最终检查点、状态和清单。"""
        self._commit_elapsed()
        completed_reasons = {
            "target_decisions_reached",
            "max_wall_seconds_reached",
        }
        status = "completed" if stop_reason in completed_reasons else "stopped"
        latest = self._save()
        _atomic_copy(
            latest,
            self.config.output / "checkpoints" / "checkpoint_final.pt",
        )
        final_state = _state(
            self.config,
            status,
            self.update,
            self.counters,
            stop_reason,
        )
        atomic_json(self.config.output / "state.json", final_state)
        manifest = dict(
            json.loads((self.config.output / "manifest.json").read_text())
        )
        manifest.update(
            {
                "status": status,
                "completed_at_utc": utc_now(),
                "stop_reason": stop_reason,
                "updates": self.update,
                "long_hands": self.counters["long_hands"],
                "long_decisions": self.counters["long_decisions"],
                "consumed_training_seconds": self.counters[
                    "consumed_training_seconds"
                ],
            }
        )
        atomic_json(self.config.output / "manifest.json", manifest)
        self.events.write(
            {
                "timestamp": utc_now(),
                "event": "long_training_finished",
                **final_state,
            }
        )
        return 0

    def fail(self, exc: BaseException) -> None:
        """保存失败现场，不伪造completed，也不吞掉原异常。"""
        self._commit_elapsed()
        failure = _state(
            self.config,
            "failed",
            self.update,
            self.counters,
            type(exc).__name__,
        )
        failure["error_type"] = type(exc).__name__
        failure["error"] = str(exc)
        atomic_json(self.config.output / "state.json", failure)
        self.events.write(
            {
                "timestamp": utc_now(),
                "event": "long_training_failed",
                **failure,
            }
        )
