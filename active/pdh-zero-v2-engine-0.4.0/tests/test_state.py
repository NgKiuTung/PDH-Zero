"""手局终局契约与牌谱边界测试。"""
import json
import unittest

from pdh_v2.contracts import RuleError
from pdh_v2.game.state import HandOutcome, WinnerResult
from pdh_v2.replay import HandTranscript
from pdh_v2.scoring import ScoreResult


def _winner(player: int = 1) -> WinnerResult:
    """构造不依赖具体牌型的已裁定赢家。"""
    return WinnerResult(player, ScoreResult(3, (), (), "TEST", 0))


class OutcomeContractTests(unittest.TestCase):
    """拒绝无法进入结算层的不一致终局。"""

    def test_draw_requires_draw_method_and_no_winners(self) -> None:
        with self.assertRaises(RuleError):
            HandOutcome("h", "DRAW", (_winner(),), None, True, "FRONT", ())

    def test_self_draw_rejects_discarder_and_multiple_winners(self) -> None:
        with self.assertRaises(RuleError):
            HandOutcome(
                "h",
                "SELF_DRAW",
                (_winner(1), _winner(2)),
                0,
                False,
                "FRONT",
                (),
            )

    def test_discard_win_requires_external_discarder(self) -> None:
        with self.assertRaises(RuleError):
            HandOutcome("h", "DISCARD_WIN", (_winner(),), None, False, "FRONT", ())
        with self.assertRaises(RuleError):
            HandOutcome("h", "DISCARD_WIN", (_winner(),), 1, False, "FRONT", ())

    def test_old_wall_rejects_duplicate_entities(self) -> None:
        with self.assertRaises(RuleError):
            HandOutcome("h", "SELF_DRAW", (_winner(),), None, False, "FRONT", (1, 1))


class TranscriptContractTests(unittest.TestCase):
    """牌谱动作字段必须与规范版本完全一致。"""

    def test_unknown_action_field_is_rejected(self) -> None:
        transcript = HandTranscript(
            "h",
            0,
            tuple(range(136)),
            (),
            "0" * 64,
        )
        payload = json.loads(transcript.to_json())
        payload["actions"] = [
            {
                "player": 0,
                "kind": "PASS",
                "tiles": [],
                "meld_index": None,
                "hidden": "not allowed",
            }
        ]
        with self.assertRaises(RuleError):
            HandTranscript.from_json(json.dumps(payload).encode())


if __name__ == "__main__":
    unittest.main()
