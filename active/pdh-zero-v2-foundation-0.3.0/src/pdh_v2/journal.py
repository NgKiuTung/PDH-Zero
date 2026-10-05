"""结算事件的规范JSON快照与恢复；不执行pickle、外部路径或任意代码。"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from pdh_v2.config import SOURCE_SHA256
from pdh_v2.contracts import RuleError
from pdh_v2.ledger import SettlementBook
from pdh_v2.qualifications import WinMethod
from pdh_v2.responsibility import TwelveGroundedLiability
from pdh_v2.settlement import DrawRecord, MultiDiscardWin, MultiWinner, OrdinaryWin


SNAPSHOT_SCHEMA_VERSION = 2


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def dump_snapshot(book: SettlementBook) -> bytes:
    """导出规范JSON和SHA256；校验用于发现损坏，不是发布者身份认证。"""
    payload = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "rule_source_sha256": SOURCE_SHA256,
        "events": book.events,
    }
    digest = hashlib.sha256(_canonical(payload)).hexdigest()
    return _canonical({"payload": payload, "sha256": digest}) + b"\n"


def _multi_win(data: dict[str, Any]) -> MultiDiscardWin:
    values = dict(data)
    values["winners"] = tuple(MultiWinner(**dict(row)) for row in values["winners"])
    return MultiDiscardWin(**values)


def _apply_event(book: SettlementBook, event: dict[str, Any]) -> None:
    operation = event.get("op")
    if operation == "DRAW" and set(event) == {"op", "draw"}:
        book.record_draw(DrawRecord(**dict(event["draw"])))
    elif operation == "BEGIN" and set(event) == {"op", "win", "old_available"}:
        win_data = dict(event["win"])
        win_data["method"] = WinMethod(win_data["method"])
        book.begin(OrdinaryWin(**win_data), event["old_available"])
    elif operation == "BEGIN_TWELVE_GROUNDED" and set(event) == {
        "op",
        "win",
        "old_available",
        "liability",
    }:
        win_data = dict(event["win"])
        win_data["method"] = WinMethod(win_data["method"])
        book.begin_twelve_grounded(
            OrdinaryWin(**win_data),
            event["old_available"],
            TwelveGroundedLiability(**dict(event["liability"])),
        )
    elif operation == "BEGIN_MULTI" and set(event) == {
        "op",
        "win",
        "old_available",
    }:
        book.begin_multi(_multi_win(dict(event["win"])), event["old_available"])
    elif operation == "BIND" and set(event) == {
        "op",
        "settlement_id",
        "source_id",
        "ordered_source",
    }:
        book.bind(event["settlement_id"], event["source_id"], event["ordered_source"])
    elif operation == "BIND_MULTI" and set(event) == {
        "op",
        "settlement_id",
        "source_id",
        "ordered_source",
    }:
        book.bind_multi(
            event["settlement_id"],
            event["source_id"],
            event["ordered_source"],
        )
    elif operation == "PAY" and set(event) == {"op", "settlement_id"}:
        book.pay(event["settlement_id"])
    elif operation == "PAY_MULTI" and set(event) == {"op", "settlement_id"}:
        book.pay_multi(event["settlement_id"])
    else:
        raise RuleError("snapshot.invalid")


def load_snapshot(raw: bytes) -> SettlementBook:
    """校验来源和事件后重建账本；失败不修改任何既有账本。"""
    try:
        envelope = json.loads(raw)
        if set(envelope) != {"payload", "sha256"}:
            raise RuleError("snapshot.invalid")
        payload = envelope["payload"]
        if set(payload) != {"schema_version", "rule_source_sha256", "events"}:
            raise RuleError("snapshot.invalid")
        if (
            type(payload["schema_version"]) is not int
            or payload["schema_version"] != SNAPSHOT_SCHEMA_VERSION
        ):
            raise RuleError("snapshot.invalid")
        if payload["rule_source_sha256"] != SOURCE_SHA256:
            raise RuleError("snapshot.invalid")
        if hashlib.sha256(_canonical(payload)).hexdigest() != envelope["sha256"]:
            raise RuleError("snapshot.invalid")
        book = SettlementBook()
        for event in payload["events"]:
            _apply_event(book, event)
        if _canonical(book.events) != _canonical(payload["events"]):
            raise RuleError("snapshot.invalid")
        return book
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise RuleError("snapshot.invalid") from exc


def save_snapshot(book: SettlementBook, path: Path) -> None:
    """新建不可覆盖快照；同目录临时文件fsync后再原子提交。"""
    raw = dump_snapshot(book)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=".pdh-",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise RuleError("snapshot.exists") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
