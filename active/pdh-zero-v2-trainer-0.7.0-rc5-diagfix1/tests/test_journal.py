"""普通结算日志恢复、幂等与磁盘提交测试；无云实例或外部支付。"""
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from typing import Any

from pdh_v2.contracts import RuleError
from pdh_v2.journal import dump_snapshot, load_snapshot, save_snapshot
from pdh_v2.ledger import SettlementBook
from pdh_v2.responsibility import TwelveGroundedLiability
from tests.helpers import multi_win, next_hand_cards, ordinary_win


def rehash(envelope: dict[str, Any]) -> bytes:
    """测试用重签：校验只是完整性，业务校验仍必须拒绝错误事件。"""
    raw = json.dumps(envelope["payload"],ensure_ascii=False,sort_keys=True,
                     separators=(",",":"),allow_nan=False).encode()
    envelope["sha256"] = hashlib.sha256(raw).hexdigest()
    return json.dumps(envelope).encode()


class JournalTests(unittest.TestCase):
    """按断点分别重建；所有金额由同一事件流重算，非另存聚合结果。"""

    def setUp(self) -> None:
        self.book = SettlementBook()
        self.win = ordinary_win()
        self.book.begin(self.win,3)

    def test_restore_before_horses_is_still_pending(self) -> None:
        restored = load_snapshot(dump_snapshot(self.book))
        self.assertEqual(restored.get(self.win.settlement_id).status,"WAITING_HORSES")
        self.assertEqual(restored.balances,(0,0,0,0))

    def test_restore_after_bind_does_not_redraw(self) -> None:
        self.book.bind(self.win.settlement_id,"hand-2",next_hand_cards())
        restored = load_snapshot(dump_snapshot(self.book))
        restored.pay(self.win.settlement_id)
        self.assertEqual(restored.balances,(-24,72,-24,-24))
        self.assertEqual(restored.events[1],self.book.events[1])

    def test_restore_after_paid_does_not_pay_twice(self) -> None:
        self.book.bind(self.win.settlement_id,"hand-2",next_hand_cards())
        self.book.pay(self.win.settlement_id)
        restored = load_snapshot(dump_snapshot(self.book))
        restored.pay(self.win.settlement_id)
        self.assertEqual(restored.balances,self.book.balances)
        self.assertEqual(dump_snapshot(restored),dump_snapshot(self.book))

    def test_tampered_payload_rejected(self) -> None:
        envelope = json.loads(dump_snapshot(self.book))
        envelope["payload"]["events"][0]["win"]["multiplier"] = 9
        with self.assertRaises(RuleError):
            load_snapshot(json.dumps(envelope).encode())

    def test_bad_rule_hash_rejected_even_with_valid_digest(self) -> None:
        envelope = json.loads(dump_snapshot(self.book))
        envelope["payload"]["rule_source_sha256"] = "other"
        with self.assertRaises(RuleError):
            load_snapshot(rehash(envelope))

    def test_wrong_schema_rejected(self) -> None:
        envelope = json.loads(dump_snapshot(self.book))
        envelope["payload"]["schema_version"] = True
        with self.assertRaises(RuleError):
            load_snapshot(rehash(envelope))

    def test_pay_without_horses_rejected_during_restore(self) -> None:
        envelope = json.loads(dump_snapshot(self.book))
        envelope["payload"]["events"].append({"op":"PAY","settlement_id":self.win.settlement_id})
        with self.assertRaises(RuleError):
            load_snapshot(rehash(envelope))

    def test_duplicate_event_stream_rejected(self) -> None:
        envelope = json.loads(dump_snapshot(self.book))
        envelope["payload"]["events"].append(envelope["payload"]["events"][0])
        with self.assertRaises(RuleError):
            load_snapshot(rehash(envelope))

    def test_unknown_operation_rejected(self) -> None:
        envelope = json.loads(dump_snapshot(self.book))
        envelope["payload"]["events"].append({"op":"EXEC","source":"not executed"})
        with self.assertRaises(RuleError):
            load_snapshot(rehash(envelope))

    def test_malformed_json_rejected(self) -> None:
        with self.assertRaises(RuleError):
            load_snapshot(b'{"payload":')

    def test_save_and_load_real_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/"snapshot.json"
            save_snapshot(self.book,target)
            restored = load_snapshot(target.read_bytes())
            self.assertEqual(dump_snapshot(restored),dump_snapshot(self.book))
            self.assertEqual(list(Path(directory).iterdir()),[target])

    def test_existing_snapshot_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/"snapshot.json"
            target.write_bytes(b"original evidence")
            with self.assertRaises(RuleError):
                save_snapshot(self.book,target)
            self.assertEqual(target.read_bytes(),b"original evidence")
            self.assertEqual(list(Path(directory).iterdir()),[target])

    def test_over_capacity_horse_cycles_round_trip(self) -> None:
        """N>136 的专门开马可重复实体编号，但每轮仍保持确定顺序。"""
        win = replace(ordinary_win(), winner=0, previous_dealer=0, continuation=66)
        book = SettlementBook()
        book.begin(win, old_available=0)
        cards = tuple(range(136)) + (0, 1)
        book.bind(win.settlement_id, "horse-cycles", cards)
        book.pay(win.settlement_id)

        restored = load_snapshot(dump_snapshot(book))

        self.assertEqual(dump_snapshot(restored), dump_snapshot(book))
        self.assertEqual(restored.get(win.settlement_id).selection.cards, cards)



class DrawJournalTests(unittest.TestCase):
    """流局零分事件仍进入恢复证据。"""

    def test_draw_event_round_trips_without_score(self) -> None:
        from pdh_v2.settlement import DrawRecord
        book = SettlementBook()
        draw = DrawRecord("draw-1", "hand-draw", 2, 3)
        book.record_draw(draw)
        restored = load_snapshot(dump_snapshot(book))
        self.assertEqual(restored.get_draw("draw-1"), draw)
        self.assertEqual(restored.balances, (0, 0, 0, 0))
        self.assertEqual(restored.events, book.events)


class MultiJournalTests(unittest.TestCase):
    """多响旧墙不足时专门开马的断点恢复与幂等付款。"""

    def test_multi_waiting_round_trip(self) -> None:
        book = SettlementBook()
        book.begin_multi(multi_win(), old_available=5)
        restored = load_snapshot(dump_snapshot(book))
        self.assertEqual(restored.get_multi("multi-1").status, "WAITING_HORSES")
        self.assertEqual(restored.balances, (0, 0, 0, 0))

    def test_multi_bound_round_trip_does_not_redraw(self) -> None:
        book = SettlementBook()
        book.begin_multi(multi_win(), old_available=5)
        book.bind_multi("multi-1", "horse-deck-1", range(136))
        restored = load_snapshot(dump_snapshot(book))
        receipt = restored.pay_multi("multi-1")
        self.assertEqual(sum(receipt), 0)
        self.assertEqual(restored.events[1], book.events[1])

    def test_multi_paid_round_trip_does_not_pay_twice(self) -> None:
        book = SettlementBook()
        book.begin_multi(multi_win(), old_available=5)
        book.bind_multi("multi-1", "horse-deck-1", range(136))
        book.pay_multi("multi-1")
        restored = load_snapshot(dump_snapshot(book))
        restored.pay_multi("multi-1")
        self.assertEqual(restored.balances, book.balances)
        self.assertEqual(dump_snapshot(restored), dump_snapshot(book))


class ResponsibilityJournalTests(unittest.TestCase):
    """第四碰责任随规范事件流恢复，不跨手或重复支付。"""

    def test_direct_kong_liability_paid_round_trip(self) -> None:
        book = SettlementBook()
        book.begin_kong_draw(ordinary_win(), 6, responsible=2)
        book.bind("settlement-1", "wall-kong", range(6))
        receipt = book.pay("settlement-1")

        restored = load_snapshot(dump_snapshot(book))

        self.assertEqual(restored.pay("settlement-1"), receipt)
        self.assertEqual(restored.balances, book.balances)
        self.assertEqual(dump_snapshot(restored), dump_snapshot(book))

    def test_grounded_liability_paid_round_trip(self) -> None:
        book = SettlementBook()
        liability = TwelveGroundedLiability("hand-1", 1, 2)
        book.begin_twelve_grounded(ordinary_win(), 6, liability)
        book.bind("settlement-1", "wall-1", range(6))
        book.pay("settlement-1")
        restored = load_snapshot(dump_snapshot(book))
        self.assertEqual(restored.pay("settlement-1"), book.balances)
        self.assertEqual(dump_snapshot(restored), dump_snapshot(book))
