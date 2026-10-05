"""训练 Smoke 运行器；用于 GPU 门禁，不代表正式论文长跑。"""
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
from pdh_v2.policy import CandidateActorCritic, ModelConfig
from pdh_v2.rollout import BatchedSessions
from pdh_v2.training.checkpoint import save_checkpoint
from pdh_v2.training.collector import CollectionConfig, RolloutCollector
from pdh_v2.training.events import JsonlWriter, ReplayWriter
from pdh_v2.training.imitation import behavior_clone, collect_teacher_episodes
from pdh_v2.training.ppo import PPOConfig, update_ppo


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    """短时训练任务的完整可审计配置。"""

    output: Path
    duration_seconds: int = 600
    environments: int = 64
    hands_per_update: int = 128
    hidden_size: int = 256
    learning_rate: float = 3e-4
    reward_scale: float = 32.0
    ppo_epochs: int = 4
    minibatch_decisions: int = 4096
    warmup_hands: int = 128
    warmup_epochs: int = 2
    seed: int = 20260930
    device: str = "auto"
    allow_shared_gpu: bool = False


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
        "schema_version": 1,
        "created_at_utc": _utc(),
        "status": "running",
        "scope": "training_smoke_candidate",
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


def run(config: RunnerConfig) -> int:
    """运行限时训练任务并写入牌谱、指标和检查点。"""
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
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    manifest_path = config.output / "manifest.json"
    manifest = _manifest(config, device)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    events = JsonlWriter(config.output / "events.jsonl")
    replays = ReplayWriter(config.output / "replays" / "hands.jsonl.gz")
    metrics = JsonlWriter(config.output / "metrics.jsonl")

    model = CandidateActorCritic(ModelConfig(hidden_size=config.hidden_size)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    environments = BatchedSessions(config.environments, config.seed ^ 0x51A71E)
    collector = RolloutCollector(
        environments,
        model,
        device,
        CollectionConfig(
            minimum_hands=config.hands_per_update,
            reward_scale=config.reward_scale,
        ),
    )
    if config.warmup_hands > 0:
        warmup_started = time.monotonic()
        teacher_episodes, teacher_hands = collect_teacher_episodes(
            environments, config.warmup_hands
        )
        warmup_metrics = behavior_clone(
            model,
            optimizer,
            teacher_episodes,
            device,
            config.warmup_epochs,
            config.minibatch_decisions,
            config.seed ^ 0xBC,
        )
        events.write(
            {
                "timestamp": _utc(),
                "event": "heuristic_warmup_completed",
                "hands": teacher_hands,
                "seconds": time.monotonic() - warmup_started,
                **asdict(warmup_metrics),
            }
        )
    ppo_config = PPOConfig(
        epochs=config.ppo_epochs,
        minibatch_decisions=config.minibatch_decisions,
    )

    started = time.monotonic()
    update = 0
    total_hands = 0
    total_decisions = 0
    events.write({"timestamp": _utc(), "event": "training_started"})
    try:
        while time.monotonic() - started < config.duration_seconds:
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
            total_hands += collected.hands
            total_decisions += collected.decisions
            replays.write_many(collected.terminals, update)
            elapsed = time.monotonic() - started
            raw_total = sum(collected.raw_score_sum)
            if raw_total != 0:
                raise RuntimeError(f"non-zero-sum collection: {raw_total}")
            record = {
                "timestamp": _utc(),
                "update": update,
                "elapsed_seconds": elapsed,
                "hands": collected.hands,
                "decisions": collected.decisions,
                "total_hands": total_hands,
                "total_decisions": total_decisions,
                "draw_rate": collected.draws / collected.hands,
                "win_rate_any": collected.wins / collected.hands,
                "point_shoot_end_rate": collected.point_shoots / collected.hands,
                "raw_score_sum": list(collected.raw_score_sum),
                "collection_seconds": collection_seconds,
                "update_seconds": update_seconds,
                "decisions_per_second": collected.decisions / collection_seconds,
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
                },
            )

        manifest.update(
            {
                "status": "completed",
                "completed_at_utc": _utc(),
                "updates": update,
                "total_hands": total_hands,
                "total_decisions": total_decisions,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
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
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        events.write(
            {
                "timestamp": _utc(),
                "event": "training_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        raise
