"""基于规则哈希、初始牌序和真实动作的确定性手局牌谱。"""
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

from pdh_v2.config import SOURCE_SHA256
from pdh_v2.contracts import RuleError, integer, text
from pdh_v2.game.actions import Action, ActionKind
from pdh_v2.game.engine import MahjongHand
from pdh_v2.tiles import validate_tiles

SCHEMA_VERSION = 2
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _action_from_data(data: dict[str, Any]) -> Action:
    """严格解析牌谱动作；未知字段会破坏审计契约，必须拒绝。"""
    expected = {"player", "kind", "tiles", "meld_index"}
    if set(data) != expected:
        raise RuleError("snapshot.invalid")
    try:
        return Action(
            player=data["player"],
            kind=ActionKind(data["kind"]),
            tiles=tuple(data["tiles"]),
            meld_index=data["meld_index"],
        )
    except (KeyError, TypeError, ValueError, RuleError) as exc:
        raise RuleError("snapshot.invalid") from exc


@dataclass(frozen=True, slots=True)
class HandTranscript:
    """完整可重放输入；规则来源和最终状态摘要均由SHA256锁定。"""

    hand_id: str
    dealer: int
    wall: tuple[int, ...]
    actions: tuple[dict[str, Any], ...]
    state_sha256: str
    rule_source_sha256: str = SOURCE_SHA256

    def __post_init__(self) -> None:
        text(self.hand_id, "hand_id")
        integer(self.dealer, "dealer", 0, 3)
        validate_tiles(self.wall, 136)
        if not SHA256_RE.fullmatch(self.state_sha256):
            raise RuleError("snapshot.invalid")
        if self.rule_source_sha256 != SOURCE_SHA256:
            raise RuleError("snapshot.invalid")
        if not all(isinstance(action, dict) for action in self.actions):
            raise RuleError("snapshot.invalid")
        for action in self.actions:
            _action_from_data(action)

    @classmethod
    def from_hand(cls, hand: MahjongHand) -> "HandTranscript":
        """从当前手局提取可重放输入和最终摘要哈希。"""
        actions = tuple(event["action"] for event in hand.events if event["event"] == "ACTION")
        digest = hashlib.sha256(_canonical(hand.state_digest())).hexdigest()
        return cls(hand.hand_id, hand.dealer, hand.wall.order, actions, digest)

    def to_json(self) -> bytes:
        """导出规范JSON字节；不包含可执行对象。"""
        payload = {
            "schema_version": SCHEMA_VERSION,
            "rule_source_sha256": self.rule_source_sha256,
            "hand_id": self.hand_id,
            "dealer": self.dealer,
            "wall": list(validate_tiles(self.wall, 136)),
            "actions": list(self.actions),
            "state_sha256": self.state_sha256,
        }
        return _canonical(payload) + b"\n"

    @classmethod
    def from_json(cls, raw: bytes) -> "HandTranscript":
        """从规范JSON读取牌谱，并拒绝未知字段、版本或房规来源。"""
        try:
            data = json.loads(raw)
            expected = {
                "schema_version",
                "rule_source_sha256",
                "hand_id",
                "dealer",
                "wall",
                "actions",
                "state_sha256",
            }
            if set(data) != expected or data["schema_version"] != SCHEMA_VERSION:
                raise RuleError("snapshot.invalid")
            return cls(
                data["hand_id"],
                data["dealer"],
                tuple(data["wall"]),
                tuple(data["actions"]),
                data["state_sha256"],
                data["rule_source_sha256"],
            )
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise RuleError("snapshot.invalid") from exc


def replay_hand(transcript: HandTranscript) -> MahjongHand:
    """重放动作并验证最终摘要；动作非法或摘要不一致时立即失败。"""
    hand = MahjongHand(transcript.hand_id, transcript.dealer, transcript.wall)
    for data in transcript.actions:
        hand.step(_action_from_data(data))
    digest = hashlib.sha256(_canonical(hand.state_digest())).hexdigest()
    if digest != transcript.state_sha256:
        raise RuleError("snapshot.invalid")
    return hand
