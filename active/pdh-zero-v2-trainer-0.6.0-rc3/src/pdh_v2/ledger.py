"""结算的单写入者内存账本；事件快照是唯一事实源，不执行外部支付。"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from pdh_v2.config import PLAYERS
from pdh_v2.contracts import RuleError
from pdh_v2.horses import (
    HorsePlan,
    HorseSelection,
    HorseSource,
    SharedHorseSelection,
    plan_horses,
    plan_multiwin_horses,
    select_horses,
    select_shared_horses,
)
from pdh_v2.responsibility import KongDrawLiability, TwelveGroundedLiability
from pdh_v2.settlement import (
    DrawRecord,
    MultiDiscardWin,
    OrdinaryWin,
    multi_discard_payments,
    ordinary_payments,
    kong_draw_payments,
    twelve_grounded_payments,
)
from pdh_v2.tiles import tile_type


@dataclass(frozen=True, slots=True)
class SettlementRecord:
    """普通单赢家阶段依次为未绑定、待支付、已支付。"""

    win: OrdinaryWin
    plan: HorsePlan
    selection: HorseSelection | None = None
    receipt: tuple[int, ...] | None = None
    liability: TwelveGroundedLiability | KongDrawLiability | None = None

    @property
    def status(self) -> str:
        """返回普通账单阶段。"""
        if self.receipt is not None:
            return "PAID"
        return "READY" if self.selection is not None else "WAITING_HORSES"


@dataclass(frozen=True, slots=True)
class MultiSettlementRecord:
    """一炮多响共用一批马牌；旧墙不足时只接受专门新牌序。"""

    win: MultiDiscardWin
    plan: HorsePlan
    selection: SharedHorseSelection | None = None
    receipt: tuple[int, ...] | None = None

    @property
    def status(self) -> str:
        """返回多响账单阶段。"""
        if self.receipt is not None:
            return "PAID"
        return "READY" if self.selection is not None else "WAITING_HORSES"


class SettlementBook:
    """同一进程单写入者账本；重复相同操作幂等，不支持并发修改。"""

    def __init__(self) -> None:
        self._records: dict[str, SettlementRecord] = {}
        self._multi_records: dict[str, MultiSettlementRecord] = {}
        self._draws: dict[str, DrawRecord] = {}
        self._events: list[dict[str, Any]] = []

    @property
    def events(self) -> list[dict[str, Any]]:
        """返回独立审计副本，内含私有牌源，禁止直接作为Actor观察。"""
        return deepcopy(self._events)

    @property
    def event_count(self) -> int:
        """返回游标，避免每手为取增量而复制全会话历史。"""
        return len(self._events)

    def events_since(self, offset: int) -> list[dict[str, Any]]:
        """只返回指定游标后的私有审计事件；不得送入Actor。"""
        if type(offset) is not int or not 0 <= offset <= len(self._events):
            raise ValueError("invalid event cursor")
        return deepcopy(self._events[offset:])

    @property
    def balances(self) -> tuple[int, ...]:
        """从全部已支付收据派生四家净分，不维护另一份可能漂移的累计账。"""
        receipts = [
            record.receipt
            for record in (*self._records.values(), *self._multi_records.values())
            if record.receipt is not None
        ]
        return tuple(sum(receipt[seat] for receipt in receipts) for seat in range(PLAYERS))

    def _id_in_use(self, settlement_id: str) -> bool:
        return (
            settlement_id in self._records
            or settlement_id in self._multi_records
            or settlement_id in self._draws
        )

    def record_draw(self, draw: DrawRecord) -> DrawRecord:
        """记录流局事实但不改变余额；重复相同事件幂等，不同内容冲突。"""
        old = self._draws.get(draw.draw_id)
        if old is not None:
            if old != draw:
                raise RuleError("settlement.conflict")
            return old
        if self._id_in_use(draw.draw_id):
            raise RuleError("settlement.conflict")
        self._draws[draw.draw_id] = draw
        self._events.append({"op": "DRAW", "draw": asdict(draw)})
        return draw

    def get_draw(self, draw_id: str) -> DrawRecord:
        """按ID读取流局记录；不存在时拒绝。"""
        try:
            return self._draws[draw_id]
        except KeyError as exc:
            raise RuleError("settlement.missing") from exc

    def get(self, settlement_id: str) -> SettlementRecord:
        """按ID读取单赢家账单；不存在时拒绝。"""
        try:
            return self._records[settlement_id]
        except KeyError as exc:
            raise RuleError("settlement.missing") from exc

    def get_multi(self, settlement_id: str) -> MultiSettlementRecord:
        """按ID读取多响账单；不存在时拒绝。"""
        try:
            return self._multi_records[settlement_id]
        except KeyError as exc:
            raise RuleError("settlement.missing") from exc

    def begin(self, win: OrdinaryWin, old_available: int) -> SettlementRecord:
        """冻结普通单赢家快照并选择牌源；不生成随机牌或更换庄位。"""
        record = SettlementRecord(win, plan_horses(win.continuation, old_available))
        old = self._records.get(win.settlement_id)
        if old is not None:
            if old.win != record.win or old.plan != record.plan:
                raise RuleError("settlement.conflict")
            return old
        if self._id_in_use(win.settlement_id):
            raise RuleError("settlement.conflict")
        self._records[win.settlement_id] = record
        self._events.append(
            {"op": "BEGIN", "win": asdict(win), "old_available": old_available}
        )
        return record

    def begin_twelve_grounded(
        self,
        win: OrdinaryWin,
        old_available: int,
        liability: TwelveGroundedLiability,
    ) -> SettlementRecord:
        """冻结第四碰责任；责任只对同局该玩家自摸生效，吃胡仍走普通付款。"""
        if win.hand_id != liability.hand_id or win.winner != liability.claimant:
            raise RuleError("responsibility.context")
        record = SettlementRecord(
            win,
            plan_horses(win.continuation, old_available),
            liability=liability,
        )
        old = self._records.get(win.settlement_id)
        if old is not None:
            if (
                old.win != record.win
                or old.plan != record.plan
                or old.liability != record.liability
            ):
                raise RuleError("settlement.conflict")
            return old
        if self._id_in_use(win.settlement_id):
            raise RuleError("settlement.conflict")
        self._records[win.settlement_id] = record
        self._events.append(
            {
                "op": "BEGIN_TWELVE_GROUNDED",
                "win": asdict(win),
                "old_available": old_available,
                "liability": asdict(liability),
            }
        )
        return record

    def begin_kong_draw(
        self,
        win: OrdinaryWin,
        old_available: int,
        responsible: int,
    ) -> SettlementRecord:
        """冻结直接点杠补牌自摸责任；该责任优先且不与十二张责任叠付。"""
        liability = KongDrawLiability(win.hand_id, win.winner, responsible)
        record = SettlementRecord(
            win,
            plan_horses(win.continuation, old_available),
            liability=liability,
        )
        old = self._records.get(win.settlement_id)
        if old is not None:
            if old.win != record.win or old.plan != record.plan or old.liability != liability:
                raise RuleError("settlement.conflict")
            return old
        if self._id_in_use(win.settlement_id):
            raise RuleError("settlement.conflict")
        self._records[win.settlement_id] = record
        self._events.append({
            "op": "BEGIN_KONG_DRAW",
            "win": asdict(win),
            "old_available": old_available,
            "liability": asdict(liability),
        })
        return record

    def begin_multi(
        self,
        win: MultiDiscardWin,
        old_available: int,
    ) -> MultiSettlementRecord:
        """冻结多响快照；旧墙不足时直接选择专门新牌序，不借下局手牌。"""
        record = MultiSettlementRecord(
            win,
            plan_multiwin_horses(win.continuation, old_available),
        )
        old = self._multi_records.get(win.settlement_id)
        if old is not None:
            if old.win != record.win or old.plan != record.plan:
                raise RuleError("settlement.conflict")
            return old
        if self._id_in_use(win.settlement_id):
            raise RuleError("settlement.conflict")
        self._multi_records[win.settlement_id] = record
        self._events.append(
            {"op": "BEGIN_MULTI", "win": asdict(win), "old_available": old_available}
        )
        return record

    def bind(
        self,
        settlement_id: str,
        source_id: str,
        ordered_source: Iterable[int],
    ) -> SettlementRecord:
        """绑定普通账单固定来源；重试只能提交完全相同的原始序列。"""
        previous = self.get(settlement_id)
        # NOTE: 专门开马可跨重洗轮次重复实体ID；牌源模块按每轮校验。
        cards = tuple(ordered_source)
        selection = select_horses(
            previous.plan,
            source_id,
            cards,
            previous.win.previous_relative_seat,
        )
        event = {
            "op": "BIND",
            "settlement_id": settlement_id,
            "source_id": source_id,
            "ordered_source": list(cards),
        }
        if previous.selection is not None:
            old_event = next(
                item
                for item in self._events
                if item["op"] == "BIND" and item["settlement_id"] == settlement_id
            )
            if old_event != event:
                raise RuleError("settlement.conflict")
            return previous
        record = SettlementRecord(
            previous.win,
            previous.plan,
            selection,
            liability=previous.liability,
        )
        self._records[settlement_id] = record
        self._events.append(event)
        return record

    def bind_multi(
        self,
        settlement_id: str,
        source_id: str,
        ordered_source: Iterable[int],
    ) -> MultiSettlementRecord:
        """绑定多响统一牌序；专门牌序必须完整，重试不可改牌。"""
        previous = self.get_multi(settlement_id)
        # NOTE: 多响专门开马同样允许不同重洗轮次复用实体ID。
        cards = tuple(ordered_source)
        selection = select_shared_horses(previous.plan, source_id, cards)
        event = {
            "op": "BIND_MULTI",
            "settlement_id": settlement_id,
            "source_id": source_id,
            "ordered_source": list(cards),
        }
        if previous.selection is not None:
            old_event = next(
                item
                for item in self._events
                if item["op"] == "BIND_MULTI"
                and item["settlement_id"] == settlement_id
            )
            if old_event != event:
                raise RuleError("settlement.conflict")
            return previous
        record = MultiSettlementRecord(previous.win, previous.plan, selection)
        self._multi_records[settlement_id] = record
        self._events.append(event)
        return record

    def pay(self, settlement_id: str) -> tuple[int, ...]:
        """只对已绑定普通账单提交一次；重复支付返回原收据。"""
        previous = self.get(settlement_id)
        if previous.receipt is not None:
            return previous.receipt
        if previous.selection is None:
            raise RuleError("settlement.unbound")
        if previous.liability is None:
            receipt = ordinary_payments(previous.win, previous.selection.hit_count)
        elif isinstance(previous.liability, KongDrawLiability):
            receipt = kong_draw_payments(
                previous.win, previous.selection.hit_count, previous.liability
            )
        else:
            receipt = twelve_grounded_payments(
                previous.win, previous.selection.hit_count, previous.liability
            )
        self._records[settlement_id] = SettlementRecord(
            previous.win,
            previous.plan,
            previous.selection,
            receipt,
            previous.liability,
        )
        self._events.append({"op": "PAY", "settlement_id": settlement_id})
        return receipt

    def pay_multi(self, settlement_id: str) -> tuple[int, ...]:
        """对多响统一马牌只提交一次；各赢家按冻结的本局位置计算m。"""
        previous = self.get_multi(settlement_id)
        if previous.receipt is not None:
            return previous.receipt
        if previous.selection is None:
            raise RuleError("settlement.unbound")
        hits = {
            item.winner: previous.selection.hit_count_for(
                previous.win.relative_seat(item.winner)
            )
            for item in previous.win.winners
        }
        receipt = multi_discard_payments(previous.win, hits)
        self._multi_records[settlement_id] = MultiSettlementRecord(
            previous.win,
            previous.plan,
            previous.selection,
            receipt,
        )
        self._events.append({"op": "PAY_MULTI", "settlement_id": settlement_id})
        return receipt

    def next_hand_reveal(self, settlement_id: str) -> dict[str, Any] | None:
        """仅普通单赢家下局手牌来源可公开命中牌；多响不足分支不借下局手牌。"""
        record = self.get(settlement_id)
        selection = record.selection
        if selection is None or record.plan.source is not HorseSource.NEXT_HAND:
            return None
        counts = Counter(tile_type(card) for card in selection.hits)
        return {
            "event": "REVEAL_RETAINED_HORSES",
            "source_hand_id": selection.source_id,
            "settles_hand_id": record.win.hand_id,
            "holder": record.win.winner,
            "cards": [
                {"kind": kind, "count": count}
                for kind, count in sorted(counts.items())
            ],
        }
