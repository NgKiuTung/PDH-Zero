"""长任务按 PPO 更新原子提交的 gzip 牌谱。"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil

from pdh_v2.rollout import TerminalHand


def _payload(terminal: TerminalHand, update: int) -> dict[str, object]:
    transcript = json.loads(terminal.transcript.to_json())
    return {
        "schema_version": 1,
        "update": update,
        "env_index": terminal.env_index,
        "receipt": list(terminal.receipt),
        "draw": terminal.draw,
        "method": terminal.method,
        "winners": list(terminal.winners),
        "discarder": terminal.discarder,
        "transcript": transcript,
    }


def replay_path(root: Path, update: int) -> Path:
    """返回分层的更新牌谱路径，避免单目录积累大量文件。"""
    bucket = update // 1000
    return root / f"{bucket:06d}" / f"update-{update:09d}.jsonl.gz"


def write_replay_update(
    root: Path,
    terminals: tuple[TerminalHand, ...],
    update: int,
) -> dict[str, object]:
    """原子写入一轮完整牌谱；相同更新已存在时拒绝覆盖。"""
    target = replay_path(root, update)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise RuntimeError(f"replay update already exists: {target}")
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            for terminal in terminals:
                stream.write(
                    json.dumps(
                        _payload(terminal, update),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return {
        "path": str(target.relative_to(root.parent)),
        "hands": len(terminals),
        "sha256": digest,
        "size_bytes": target.stat().st_size,
    }


def reconcile_replays(root: Path, committed_update: int) -> tuple[Path, ...]:
    """隔离检查点之后的孤儿牌谱，避免恢复时重复记账。"""
    orphaned: list[Path] = []
    for path in sorted(root.glob("*/*.jsonl.gz")):
        stem = path.name.removeprefix("update-").removesuffix(".jsonl.gz")
        try:
            update = int(stem)
        except ValueError as exc:
            raise RuntimeError(f"unexpected replay filename: {path}") from exc
        if update <= committed_update:
            continue
        quarantine = root / "orphaned" / path.parent.name / path.name
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), quarantine)
        orphaned.append(quarantine)
    return tuple(orphaned)
