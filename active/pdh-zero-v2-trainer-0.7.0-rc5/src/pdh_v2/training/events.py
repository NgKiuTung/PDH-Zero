"""训练事件与紧凑牌谱的结构化落盘。"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from pdh_v2.rollout import TerminalHand


class JsonlWriter:
    """按行追加 JSON；每次写入后刷新，便于中途审计。"""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def write(self, payload: dict[str, Any]) -> None:
        """写入一条规范 JSON 事件。"""
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()


class ReplayWriter:
    """保存每手可重放牌谱与原始四家账本。"""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def write_many(self, terminals: tuple[TerminalHand, ...], update: int) -> None:
        """将完整手局追加到 gzip JSONL。"""
        with gzip.open(self.path, "at", encoding="utf-8") as stream:
            for terminal in terminals:
                transcript = json.loads(terminal.transcript.to_json())
                payload = {
                    "schema_version": 2,
                    "update": update,
                    "env_index": terminal.env_index,
                    "receipt": list(terminal.receipt),
                    "draw": terminal.draw,
                    "method": terminal.method,
                    "winners": list(terminal.winners),
                    "discarder": terminal.discarder,
                    "transcript": transcript,
                    "audit": terminal.audit,
                }
                stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
