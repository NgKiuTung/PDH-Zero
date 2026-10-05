"""10 分钟 Smoke 到长训练的晋级门禁。"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
import math
from pathlib import Path
from typing import Any

from pdh_v2.config import SOURCE_SHA256


@dataclass(frozen=True, slots=True)
class SmokeGateResult:
    """已通过晋级门禁的 Smoke 身份与训练摘要。"""

    smoke_dir: Path
    checkpoint: Path
    update: int
    total_hands: int
    total_decisions: int
    elapsed_seconds: float
    hidden_size: int
    seed: int


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _last_jsonl(path: Path) -> dict[str, Any]:
    last = ""
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                last = line
    if not last:
        raise RuntimeError(f"empty JSONL file: {path}")
    return dict(json.loads(last))


def _finite_metrics(record: dict[str, Any]) -> None:
    for key, value in record.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(f"non-finite smoke metric: {key}={value}")


def validate_smoke(smoke_dir: Path) -> SmokeGateResult:
    """验证 Smoke 成功、数值有限、零和且具备可加载检查点。"""
    root = smoke_dir.resolve(strict=True)
    manifest = _read_json(root / "manifest.json")
    if manifest.get("status") != "completed":
        raise RuntimeError("smoke manifest is not completed")
    if manifest.get("scope") != "training_smoke_candidate":
        raise RuntimeError("unexpected smoke scope")
    if manifest.get("rule_source_sha256") != SOURCE_SHA256:
        raise RuntimeError("smoke rule source mismatch")
    config = dict(manifest.get("config", {}))
    elapsed = float(manifest.get("elapsed_seconds", 0.0))
    if int(config.get("duration_seconds", 0)) < 600 or elapsed < 540.0:
        raise RuntimeError("smoke duration gate not met")
    update = int(manifest.get("updates", 0))
    hands = int(manifest.get("total_hands", 0))
    decisions = int(manifest.get("total_decisions", 0))
    if update < 1 or hands < 1 or decisions < 1:
        raise RuntimeError("smoke produced no trainable work")

    latest = _last_jsonl(root / "metrics.jsonl")
    _finite_metrics(latest)
    raw_scores = tuple(int(value) for value in latest.get("raw_score_sum", ()))
    if len(raw_scores) != 4 or sum(raw_scores) != 0:
        raise RuntimeError("smoke latest update is not zero-sum")
    if int(latest.get("update", -1)) != update:
        raise RuntimeError("smoke manifest and metrics update mismatch")

    checkpoint = root / "checkpoints" / "checkpoint_latest.pt"
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise RuntimeError("smoke checkpoint missing")
    replay = root / "replays" / "hands.jsonl.gz"
    try:
        with gzip.open(replay, "rt", encoding="utf-8") as stream:
            first = stream.readline()
    except (OSError, EOFError) as exc:
        raise RuntimeError("smoke replay is unreadable") from exc
    if not first.strip():
        raise RuntimeError("smoke replay is empty")
    json.loads(first)

    return SmokeGateResult(
        root,
        checkpoint,
        update,
        hands,
        decisions,
        elapsed,
        int(config["hidden_size"]),
        int(config["seed"]),
    )
