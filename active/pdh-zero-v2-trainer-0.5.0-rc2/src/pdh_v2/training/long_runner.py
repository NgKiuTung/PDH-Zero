"""Smoke晋级后的可恢复长训练基线入口。"""
from __future__ import annotations

import json
import os
from typing import Any

import torch

from pdh_v2.policy import CandidateActorCritic, ModelConfig
from pdh_v2.rollout import BatchedSessions
from pdh_v2.training.checkpoint import load_checkpoint
from pdh_v2.training.collector import CollectionConfig, RolloutCollector
from pdh_v2.training.events import JsonlWriter
from pdh_v2.training.long_artifacts import (
    _manifest,
    _rebuild_live_metrics,
    _reconcile_metrics,
    _state,
)
from pdh_v2.training.long_checkpoint import load_long_checkpoint
from pdh_v2.training.long_config import LongRunConfig
from pdh_v2.training.long_loop import _LongTrainingLoop
from pdh_v2.training.long_runtime import (
    StopFlag,
    Telemetry,
    atomic_json,
    gpu_processes,
    utc_now,
)
from pdh_v2.training.ppo import PPOConfig
from pdh_v2.training.promotion import SmokeGateResult, validate_smoke
from pdh_v2.training.replay_updates import reconcile_replays


def _new_runtime(
    config: LongRunConfig,
    smoke: SmokeGateResult,
    device: torch.device,
) -> tuple[
    CandidateActorCritic,
    torch.optim.Optimizer,
    BatchedSessions,
    int,
    dict[str, Any],
]:
    if smoke.hidden_size != config.hidden_size:
        raise RuntimeError("smoke and long-run hidden size mismatch")
    model = CandidateActorCritic(ModelConfig(hidden_size=config.hidden_size)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    update, smoke_metadata = load_checkpoint(smoke.checkpoint, model, optimizer, device)
    for group in optimizer.param_groups:
        group["lr"] = config.learning_rate
    environments = BatchedSessions(config.environments, config.seed ^ 0x51A71E)
    environments.active = [False] * environments.count
    counters: dict[str, Any] = {
        "source_smoke_metadata": smoke_metadata,
        "long_updates": 0,
        "long_hands": 0,
        "long_decisions": 0,
        "consumed_training_seconds": 0.0,
        "next_evaluation_decisions": 0,
        "next_snapshot_decisions": config.snapshot_interval_decisions,
        "best_evaluation_mean_score": None,
        "attempts": 1,
    }
    return model, optimizer, environments, update, counters


def _resume_runtime(
    config: LongRunConfig,
    device: torch.device,
) -> tuple[
    CandidateActorCritic,
    torch.optim.Optimizer,
    BatchedSessions,
    int,
    dict[str, Any],
]:
    model = CandidateActorCritic(ModelConfig(hidden_size=config.hidden_size)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    checkpoint = config.output / "checkpoints" / "checkpoint_latest.pt"
    environments, update, counters = load_long_checkpoint(
        checkpoint,
        model,
        optimizer,
        device,
        config.checkpoint_identity(),
    )
    counters["attempts"] = int(counters.get("attempts", 1)) + 1
    reconcile_replays(config.output / "replays", update)
    _reconcile_metrics(config.output, update)
    _rebuild_live_metrics(config.output, update)
    return model, optimizer, environments, update, counters


def _prepare_run(
    config: LongRunConfig,
    resume: bool,
) -> tuple[
    torch.device,
    CandidateActorCritic,
    torch.optim.Optimizer,
    BatchedSessions,
    int,
    dict[str, Any],
]:
    config.output.mkdir(parents=True, exist_ok=True)
    smoke = None if resume else validate_smoke(config.smoke_dir)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    if device.type == "cuda" and not config.allow_shared_gpu:
        foreign = gpu_processes()
        if foreign:
            raise RuntimeError(f"foreign GPU compute processes detected: {foreign}")
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    if resume:
        model, optimizer, environments, update, counters = _resume_runtime(
            config,
            device,
        )
    else:
        if smoke is None:
            raise RuntimeError("validated smoke identity missing")
        model, optimizer, environments, update, counters = _new_runtime(
            config,
            smoke,
            device,
        )
        atomic_json(config.output / "manifest.json", _manifest(config, smoke))
        atomic_json(config.output / "config.json", config.to_dict())
    return device, model, optimizer, environments, update, counters


def _start_attempt(
    config: LongRunConfig,
    resume: bool,
    update: int,
    counters: dict[str, Any],
    events: JsonlWriter,
) -> None:
    events.write(
        {
            "timestamp": utc_now(),
            "event": "long_training_attempt_started",
            "attempt": counters["attempts"],
            "resume": resume,
        }
    )
    atomic_json(
        config.output / "state.json",
        _state(config, "running", update, counters, None),
    )
    if not resume:
        return
    manifest = dict(json.loads((config.output / "manifest.json").read_text()))
    manifest.update({"status": "running", "resumed_at_utc": utc_now()})
    atomic_json(config.output / "manifest.json", manifest)


def run_long(config: LongRunConfig, resume: bool = False) -> int:
    """运行默认48小时或5000万有效决策的可恢复基线。"""
    device, model, optimizer, environments, update, counters = _prepare_run(
        config,
        resume,
    )
    events = JsonlWriter(config.output / "events.jsonl")
    live_metrics = JsonlWriter(config.output / "metrics.jsonl")
    stop = StopFlag()
    stop.install()
    telemetry = (
        Telemetry(
            config.output
            / "telemetry"
            / f"attempt-{int(counters['attempts']):02d}.csv"
        )
        if device.type == "cuda"
        else None
    )
    if telemetry is not None:
        telemetry.start()
    collector = RolloutCollector(
        environments,
        model,
        device,
        CollectionConfig(
            minimum_hands=config.hands_per_update,
            reward_scale=config.reward_scale,
        ),
    )
    ppo_config = PPOConfig(
        epochs=config.ppo_epochs,
        minibatch_decisions=config.minibatch_decisions,
    )
    loop = _LongTrainingLoop(
        config,
        model,
        optimizer,
        environments,
        update,
        counters,
        device,
        collector,
        ppo_config,
        events,
        live_metrics,
        stop,
    )
    _start_attempt(config, resume, update, counters, events)
    try:
        return loop.finish(loop.run())
    except BaseException as exc:
        loop.fail(exc)
        raise
    finally:
        if telemetry is not None:
            telemetry.stop()
