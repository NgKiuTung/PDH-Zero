"""课程学习短试运行器；用于晋级正式长训练前的学习效果门禁。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import subprocess
import time
from typing import Any

import torch

from pdh_v2.config import SOURCE_SHA256
from pdh_v2.game.actions import ActionKind
from pdh_v2.policy import CandidateActorCritic, ModelConfig
from pdh_v2.rollout import BatchedSessions
from pdh_v2.training.checkpoint import save_checkpoint
from pdh_v2.training.collector import CollectionConfig, RolloutCollector
from pdh_v2.training.curriculum import CurriculumSchedule
from pdh_v2.training.evaluation import EvaluationConfig, evaluate_vs_heuristic
from pdh_v2.training.events import JsonlWriter, ReplayWriter
from pdh_v2.training.imitation import (
    behavior_clone,
    collect_teacher_episodes,
    evaluate_imitation,
    split_teacher_episodes,
)
from pdh_v2.training.ppo import PPOConfig, update_ppo


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    """短时训练任务的完整可审计配置。"""

    output: Path
    duration_seconds: int = 600
    environments: int = 64
    hands_per_update: int = 128
    hidden_size: int = 384
    learning_rate: float = 2e-4
    reward_scale: float = 32.0
    ppo_epochs: int = 3
    minibatch_decisions: int = 8192
    warmup_hands: int = 128
    warmup_epochs: int = 2
    warmup_validation_fraction: float = 0.1
    teacher_review_hands: int = 0
    teacher_review_epochs: int = 1
    teacher_review_interval_updates: int = 0
    evaluation_hands: int = 0
    evaluation_environments: int = 64
    evaluation_interval_decisions: int = 500_000
    seed: int = 20260930
    device: str = "auto"
    allow_shared_gpu: bool = False

    def __post_init__(self) -> None:
        if self.duration_seconds < 1 or self.environments < 1 or self.hands_per_update < 1:
            raise ValueError("duration, environments and hands-per-update must be positive")
        if self.hidden_size < 1 or self.minibatch_decisions < 1 or self.ppo_epochs < 1:
            raise ValueError("model and PPO sizes must be positive")
        if self.learning_rate <= 0 or self.reward_scale <= 0:
            raise ValueError("learning-rate and reward-scale must be positive")
        if self.warmup_hands < 0 or self.teacher_review_hands < 0:
            raise ValueError("teacher hand budgets must not be negative")
        if self.evaluation_hands < 0:
            raise ValueError("evaluation-hands must not be negative")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    return device


def _gpu_processes() -> list[int]:
    """查询已有 GPU 计算进程；nvidia-smi 不可用时返回空列表。"""
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
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]


def _manifest(config: RunnerConfig, device: torch.device) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "created_at_utc": _utc(),
        "status": "running",
        "scope": "curriculum_training_candidate",
        "rule_source_sha256": SOURCE_SHA256,
        "config": {**asdict(config), "output": str(config.output)},
        "runtime": {
            "python": os.sys.version.split()[0],
            "torch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
        },
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _evaluate(
    model: CandidateActorCritic,
    device: torch.device,
    config: RunnerConfig,
    decisions: int,
    events: JsonlWriter,
) -> dict[str, Any] | None:
    if config.evaluation_hands == 0:
        return None
    result = evaluate_vs_heuristic(
        model,
        device,
        EvaluationConfig(
            hands=config.evaluation_hands,
            environments=min(config.evaluation_environments, config.evaluation_hands),
            seed=config.seed ^ 0xECA12027,
        ),
    )
    payload: dict[str, Any] = {
        "timestamp": _utc(),
        "event": "evaluation_completed",
        "learner_decisions": decisions,
        **result.to_dict(),
    }
    _atomic_json(
        config.output / "evaluations" / f"evaluation-d{decisions:012d}.json",
        payload,
    )
    events.write(payload)
    return payload


def _warmup(
    model: CandidateActorCritic,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: RunnerConfig,
    events: JsonlWriter,
) -> None:
    if config.warmup_hands == 0:
        return
    teacher_env = BatchedSessions(
        min(config.environments, 96),
        config.seed ^ 0x7EA_C4E,
    )
    warmup_started = time.monotonic()
    episodes, teacher_hands = collect_teacher_episodes(teacher_env, config.warmup_hands)
    train_episodes, validation_episodes = split_teacher_episodes(
        episodes,
        config.warmup_validation_fraction,
        config.seed ^ 0x5A11D,
    )
    before = evaluate_imitation(
        model,
        validation_episodes,
        device,
        config.minibatch_decisions,
    )
    trained = behavior_clone(
        model,
        optimizer,
        train_episodes,
        device,
        config.warmup_epochs,
        config.minibatch_decisions,
        config.seed ^ 0xBC,
    )
    after = evaluate_imitation(
        model,
        validation_episodes,
        device,
        config.minibatch_decisions,
    )
    events.write(
        {
            "timestamp": _utc(),
            "event": "heuristic_warmup_completed",
            "hands": teacher_hands,
            "seconds": time.monotonic() - warmup_started,
            "train": asdict(trained),
            "validation_before": asdict(before),
            "validation_after": asdict(after),
        }
    )
    save_checkpoint(
        config.output / "checkpoints" / "checkpoint_warmup.pt",
        model,
        optimizer,
        0,
        {
            "teacher_hands": teacher_hands,
            "validation_accuracy": after.accuracy,
        },
    )


def _teacher_review(
    model: CandidateActorCritic,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: RunnerConfig,
    update: int,
    events: JsonlWriter,
) -> None:
    if (
        config.teacher_review_hands == 0
        or config.teacher_review_interval_updates == 0
        or update % config.teacher_review_interval_updates != 0
    ):
        return
    environments = BatchedSessions(
        min(config.environments, 64),
        config.seed ^ 0x7EAC000 ^ update,
    )
    episodes, hands = collect_teacher_episodes(
        environments,
        config.teacher_review_hands,
    )
    metrics = behavior_clone(
        model,
        optimizer,
        episodes,
        device,
        config.teacher_review_epochs,
        config.minibatch_decisions,
        config.seed ^ update ^ 0xA11CE,
    )
    events.write(
        {
            "timestamp": _utc(),
            "event": "teacher_review_completed",
            "update": update,
            "hands": hands,
            **asdict(metrics),
        }
    )


def run(config: RunnerConfig) -> int:
    """运行课程学习短试并写入牌谱、指标、评估和检查点。"""
    config.output.mkdir(parents=True, exist_ok=False)
    device = _device(config.device)
    if device.type == "cuda" and not config.allow_shared_gpu:
        foreign = _gpu_processes()
        if foreign:
            raise RuntimeError(f"foreign GPU compute processes detected: {foreign}")

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    torch.set_num_threads(max(1, min(6, os.cpu_count() or 1)))

    manifest_path = config.output / "manifest.json"
    manifest = _manifest(config, device)
    _atomic_json(manifest_path, manifest)
    events = JsonlWriter(config.output / "events.jsonl")
    replays = ReplayWriter(config.output / "replays" / "hands.jsonl.gz")
    metrics = JsonlWriter(config.output / "metrics.jsonl")

    model = CandidateActorCritic(ModelConfig(hidden_size=config.hidden_size)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    _warmup(model, optimizer, device, config, events)

    environments = BatchedSessions(config.environments, config.seed ^ 0x51A71E)
    schedule = CurriculumSchedule()
    first_stage = schedule.stage(0)
    collector = RolloutCollector(
        environments,
        model,
        device,
        CollectionConfig(
            minimum_hands=config.hands_per_update,
            reward_scale=config.reward_scale,
            seat_mix=first_stage.seat_mix,
            seat_seed=config.seed ^ 0x5EA7,
        ),
    )
    ppo_config = PPOConfig(
        epochs=config.ppo_epochs,
        minibatch_decisions=config.minibatch_decisions,
    )

    update = 0
    total_hands = 0
    total_decisions = 0
    total_environment_decisions = 0
    next_evaluation = 0
    try:
        if config.evaluation_hands > 0:
            _evaluate(model, device, config, 0, events)
            next_evaluation = config.evaluation_interval_decisions
        started = time.monotonic()
        events.write({"timestamp": _utc(), "event": "training_started"})
        while time.monotonic() - started < config.duration_seconds:
            stage = schedule.stage(total_decisions)
            collector.set_seat_mix(stage.seat_mix)
            collection_started = time.monotonic()
            collected = collector.collect()
            collection_seconds = time.monotonic() - collection_started
            update_started = time.monotonic()
            trained = update_ppo(
                model,
                optimizer,
                collected.episodes,
                device,
                ppo_config,
                config.seed + update,
            )
            update_seconds = time.monotonic() - update_started
            update += 1
            _teacher_review(model, optimizer, device, config, update, events)
            total_hands += collected.hands
            total_decisions += collected.decisions
            total_environment_decisions += collected.environment_decisions
            replays.write_many(collected.terminals, update)
            elapsed = time.monotonic() - started
            raw_total = sum(collected.raw_score_sum)
            if raw_total != 0:
                raise RuntimeError(f"non-zero-sum collection: {raw_total}")
            learner_positions = sum(
                count * hands
                for count, hands in enumerate(collected.learner_seat_histogram)
            )
            action_counts = {
                kind.value: collected.action_counts[index]
                for index, kind in enumerate(ActionKind)
            }
            record = {
                "timestamp": _utc(),
                "update": update,
                "elapsed_seconds": elapsed,
                "curriculum_stage": stage.name,
                "hands": collected.hands,
                "learner_decisions": collected.decisions,
                "environment_decisions": collected.environment_decisions,
                "heuristic_decisions": collected.heuristic_decisions,
                "total_hands": total_hands,
                "total_decisions": total_decisions,
                "total_environment_decisions": total_environment_decisions,
                "draw_rate": collected.draws / collected.hands,
                "win_rate_any": collected.wins / collected.hands,
                "learner_win_rate": collected.learner_wins / collected.hands,
                "point_shoot_end_rate": collected.point_shoots / collected.hands,
                "learner_point_shoot_rate": (
                    collected.learner_point_shoots / collected.hands
                ),
                "learner_mean_score": (
                    collected.learner_score_sum / max(1, learner_positions)
                ),
                "raw_score_sum": list(collected.raw_score_sum),
                "learner_seat_histogram": list(collected.learner_seat_histogram),
                "action_counts": action_counts,
                "win_opportunities": collected.win_opportunities,
                "wins_taken": collected.wins_taken,
                "missed_wins": collected.missed_wins,
                "win_capture_rate": (
                    collected.wins_taken / max(1, collected.win_opportunities)
                ),
                "risk_samples": collected.risk_samples,
                "risk_positives": collected.risk_positives,
                "collection_seconds": collection_seconds,
                "update_seconds": update_seconds,
                "learner_decisions_per_second": (
                    collected.decisions / collection_seconds
                ),
                "environment_decisions_per_second": (
                    collected.environment_decisions / collection_seconds
                ),
                **asdict(trained),
            }
            if device.type == "cuda":
                record["cuda_peak_memory_bytes"] = torch.cuda.max_memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
            metrics.write(record)
            events.write({"timestamp": _utc(), "event": "update_completed", **record})
            save_checkpoint(
                config.output / "checkpoints" / "checkpoint_latest.pt",
                model,
                optimizer,
                update,
                {
                    "elapsed_seconds": elapsed,
                    "total_hands": total_hands,
                    "total_decisions": total_decisions,
                    "total_environment_decisions": total_environment_decisions,
                    "curriculum_stage": stage.name,
                },
            )
            if config.evaluation_hands > 0 and total_decisions >= next_evaluation:
                _evaluate(model, device, config, total_decisions, events)
                while total_decisions >= next_evaluation:
                    next_evaluation += config.evaluation_interval_decisions

        final_evaluation = _evaluate(model, device, config, total_decisions, events)
        manifest.update(
            {
                "status": "completed",
                "completed_at_utc": _utc(),
                "updates": update,
                "total_hands": total_hands,
                "total_decisions": total_decisions,
                "total_environment_decisions": total_environment_decisions,
                "elapsed_seconds": time.monotonic() - started,
                "final_evaluation": final_evaluation,
            }
        )
        _atomic_json(manifest_path, manifest)
        events.write({"timestamp": _utc(), "event": "training_completed", "updates": update})
        return 0
    except BaseException as exc:
        manifest.update(
            {
                "status": "failed",
                "failed_at_utc": _utc(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        _atomic_json(manifest_path, manifest)
        events.write(
            {
                "timestamp": _utc(),
                "event": "training_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        raise
