"""训练检查点的原子写入与基础身份校验。"""
from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
import random
from typing import Any

import torch

from pdh_v2.config import SOURCE_SHA256
from pdh_v2.policy import CandidateActorCritic
from pdh_v2.training.rng_state import restore_rng_states

SCHEMA_VERSION = 1


def save_checkpoint(
    path: Path,
    model: CandidateActorCritic,
    optimizer: torch.optim.Optimizer,
    update: int,
    metadata: dict[str, Any],
) -> None:
    """原子保存更新边界检查点；不会覆盖一半写入的目标文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "rule_source_sha256": SOURCE_SHA256,
        "update": update,
        "model_config": asdict(model.config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "metadata": metadata,
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: Path,
    model: CandidateActorCritic,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, dict[str, Any]]:
    """读取相同冻结房规和模型结构的更新边界检查点。"""
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("checkpoint schema mismatch")
    if payload.get("rule_source_sha256") != SOURCE_SHA256:
        raise RuntimeError("checkpoint rule source mismatch")
    if payload.get("model_config") != asdict(model.config):
        raise RuntimeError("checkpoint model config mismatch")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    random.setstate(payload["python_rng"])
    restore_rng_states(payload["torch_rng"], payload.get("cuda_rng"))
    return int(payload["update"]), dict(payload.get("metadata", {}))
