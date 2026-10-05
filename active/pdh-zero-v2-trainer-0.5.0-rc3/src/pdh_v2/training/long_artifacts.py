"""长训练的清单、状态、指标、评估与检查点提交工具。"""
from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any

import torch

from pdh_v2.config import SOURCE_SHA256
from pdh_v2.policy import CandidateActorCritic
from pdh_v2.rollout import BatchedSessions
from pdh_v2.training.evaluation import EvaluationConfig, evaluate_vs_heuristic
from pdh_v2.training.long_checkpoint import save_long_checkpoint
from pdh_v2.training.long_config import LongRunConfig
from pdh_v2.training.long_runtime import atomic_json, utc_now
from pdh_v2.training.promotion import SmokeGateResult


def _ensure_finite(values: dict[str, Any]) -> None:
    for key, value in values.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(f"non-finite training metric: {key}={value}")


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, target)


def _write_update_metric(root: Path, update: int, payload: dict[str, Any]) -> Path:
    path = root / "metrics" / "updates" / f"update-{update:09d}.json"
    if path.exists():
        raise RuntimeError(f"metric update already exists: {path}")
    atomic_json(path, payload)
    return path


def _reconcile_metrics(root: Path, committed_update: int) -> tuple[Path, ...]:
    moved: list[Path] = []
    metrics_root = root / "metrics" / "updates"
    for path in sorted(metrics_root.glob("update-*.json")):
        update = int(path.stem.removeprefix("update-"))
        if update <= committed_update:
            continue
        target = root / "metrics" / "orphaned" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), target)
        moved.append(target)
    return tuple(moved)


def _rebuild_live_metrics(root: Path, committed_update: int) -> int:
    """从已提交的逐更新文件重建便捷JSONL，移除崩溃留下的孤儿行。"""
    records: list[str] = []
    metrics_root = root / "metrics" / "updates"
    for path in sorted(metrics_root.glob("update-*.json")):
        update = int(path.stem.removeprefix("update-"))
        if update > committed_update:
            continue
        payload = dict(json.loads(path.read_text(encoding="utf-8")))
        records.append(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    target = root / "metrics.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    raw = "\n".join(records)
    temporary.write_text(raw + ("\n" if raw else ""), encoding="utf-8")
    os.replace(temporary, target)
    return len(records)


def _manifest(config: LongRunConfig, smoke: SmokeGateResult) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scope": "full_rules_long_baseline_candidate",
        "status": "running",
        "created_at_utc": utc_now(),
        "rule_source_sha256": SOURCE_SHA256,
        "config": config.to_dict(),
        "source_smoke": {
            "path": str(smoke.smoke_dir),
            "checkpoint": str(smoke.checkpoint),
            "update": smoke.update,
            "hands": smoke.total_hands,
            "decisions": smoke.total_decisions,
            "elapsed_seconds": smoke.elapsed_seconds,
        },
    }


def _state(
    config: LongRunConfig,
    status: str,
    update: int,
    counters: dict[str, Any],
    stop_reason: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": status,
        "updated_at_utc": utc_now(),
        "pid": os.getpid(),
        "update": update,
        "long_updates": int(counters["long_updates"]),
        "long_hands": int(counters["long_hands"]),
        "long_decisions": int(counters["long_decisions"]),
        "consumed_training_seconds": float(counters["consumed_training_seconds"]),
        "target_decisions": config.target_decisions,
        "max_wall_seconds": config.max_wall_seconds,
        "next_evaluation_decisions": int(counters["next_evaluation_decisions"]),
        "next_snapshot_decisions": int(counters["next_snapshot_decisions"]),
        "best_evaluation_mean_score": counters.get("best_evaluation_mean_score"),
        "attempts": int(counters["attempts"]),
        "stop_reason": stop_reason,
    }


def _save(
    config: LongRunConfig,
    model: CandidateActorCritic,
    optimizer: torch.optim.Optimizer,
    environments: BatchedSessions,
    update: int,
    counters: dict[str, Any],
) -> Path:
    path = config.output / "checkpoints" / "checkpoint_latest.pt"
    save_long_checkpoint(
        path,
        model,
        optimizer,
        environments,
        update,
        counters,
        config.checkpoint_identity(),
    )
    return path


def _evaluate(
    config: LongRunConfig,
    model: CandidateActorCritic,
    device: torch.device,
    decisions: int,
) -> dict[str, Any]:
    path = config.output / "evaluations" / f"evaluation-{decisions:012d}.json"
    if path.exists():
        return dict(json.loads(path.read_text(encoding="utf-8")))
    result = evaluate_vs_heuristic(
        model,
        device,
        EvaluationConfig(
            hands=config.evaluation_hands,
            environments=config.evaluation_environments,
            seed=config.seed ^ 0xECA12027,
        ),
    )
    payload = {
        "timestamp": utc_now(),
        "long_decisions": decisions,
        **result.to_dict(),
    }
    atomic_json(path, payload)
    return payload


def _metric_payload(
    update: int,
    counters: dict[str, Any],
    collected: Any,
    trained: Any,
    collection_seconds: float,
    update_seconds: float,
    free: int,
    replay: dict[str, Any],
) -> dict[str, Any]:
    return {
        "timestamp": utc_now(),
        "update": update,
        "long_updates": counters["long_updates"],
        "long_hands": counters["long_hands"],
        "long_decisions": counters["long_decisions"],
        "hands": collected.hands,
        "decisions": collected.decisions,
        "draw_rate": collected.draws / collected.hands,
        "win_rate_any": collected.wins / collected.hands,
        "point_shoot_end_rate": collected.point_shoots / collected.hands,
        "raw_score_sum": list(collected.raw_score_sum),
        "collection_seconds": collection_seconds,
        "update_seconds": update_seconds,
        "decisions_per_second": collected.decisions / collection_seconds,
        "disk_free_bytes": free,
        "replay": replay,
        **asdict(trained),
    }
