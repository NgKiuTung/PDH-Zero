"""长训练更新边界检查点。"""
from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import random
from typing import Any

import torch

from pdh_v2.config import SOURCE_SHA256
from pdh_v2.policy import CandidateActorCritic
from pdh_v2.rollout import BatchedSessions
from pdh_v2.rollout.snapshot import restore_batched, snapshot_batched
from pdh_v2.training.rng_state import restore_rng_states

SCHEMA_VERSION = 1


def save_long_checkpoint(
    path: Path,
    model: CandidateActorCritic,
    optimizer: torch.optim.Optimizer,
    environments: BatchedSessions,
    update: int,
    counters: dict[str, Any],
    job_identity: dict[str, Any],
) -> None:
    """原子保存长任务完整更新边界状态。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "rule_source_sha256": SOURCE_SHA256,
        "model_config": asdict(model.config),
        "job_identity": dict(job_identity),
        "update": int(update),
        "counters": dict(counters),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "environments": snapshot_batched(environments),
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_long_checkpoint(
    path: Path,
    model: CandidateActorCritic,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    expected_identity: dict[str, Any],
) -> tuple[BatchedSessions, int, dict[str, Any]]:
    """恢复同一房规、模型结构和任务契约的长任务检查点。"""
    # SAFETY(v0.5.0-rc3): 仅加载本任务目录自产、哈希受控的可信检查点。
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("long checkpoint schema mismatch")
    if payload.get("rule_source_sha256") != SOURCE_SHA256:
        raise RuntimeError("long checkpoint rule source mismatch")
    if payload.get("model_config") != asdict(model.config):
        raise RuntimeError("long checkpoint model config mismatch")
    if payload.get("job_identity") != expected_identity:
        raise RuntimeError("long checkpoint job identity mismatch")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    random.setstate(payload["python_rng"])
    restore_rng_states(payload["torch_rng"], payload.get("cuda_rng"))
    environments = restore_batched(tuple(payload["environments"]))
    return environments, int(payload["update"]), dict(payload["counters"])
