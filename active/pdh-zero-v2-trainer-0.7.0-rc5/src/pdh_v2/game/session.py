"""连续对局、抓马结算、虚拟散场发牌与庄位迁移。"""
from dataclasses import dataclass
import random

from pdh_v2.config import DECK_SIZE, PLAYERS
from pdh_v2.dealer import DealerState
from pdh_v2.game.engine import MahjongHand
from pdh_v2.game.state import HandOutcome
from pdh_v2.horses import HorseSource
from pdh_v2.ledger import SettlementBook
from pdh_v2.qualifications import WinMethod
from pdh_v2.settlement import DrawRecord, MultiDiscardWin, MultiWinner, OrdinaryWin


@dataclass(frozen=True, slots=True)
class ResolvedHand:
    """已完成抓马和付款的手局结果。"""

    outcome: HandOutcome
    receipt: tuple[int, ...]
    next_dealer: DealerState
    horse_source: HorseSource | None
    virtual_next_hand: bool = False


class GameSession:
    """单线程参考会话；墙、专门开马和策略随机流应保持分离。"""

    def __init__(self, seed: int, dealer: int = 0) -> None:
        self.seed = seed
        self.wall_rng = random.Random(seed)
        self.horse_rng = random.Random(seed ^ 0x5A17A11)
        self.dealer_state = DealerState(dealer)
        self.scores = [0] * PLAYERS
        self.book = SettlementBook()
        self.hand_index = 0
        self.events: list[dict[str, object]] = []
        self.current_hand = self._new_hand(virtual=False)

    @staticmethod
    def _shuffle(rng: random.Random) -> tuple[int, ...]:
        cards = list(range(DECK_SIZE))
        rng.shuffle(cards)
        return tuple(cards)

    def _new_hand(self, virtual: bool) -> MahjongHand:
        hand = MahjongHand(
            f"H{self.hand_index:08d}",
            self.dealer_state.player,
            self._shuffle(self.wall_rng),
        )
        self.events.append(
            {
                "event": "HAND_DEALT",
                "hand_id": hand.hand_id,
                "dealer": self.dealer_state.player,
                "virtual": virtual,
                "wall": list(hand.wall.order),
            }
        )
        self.hand_index += 1
        return hand

    def _separate_sequence(self, count: int) -> tuple[int, ...]:
        """N>136时每136张重新洗同一副牌，直至开足；每轮牌序进入审计事件。"""
        cards: list[int] = []
        cycles: list[list[int]] = []
        while len(cards) < count:
            cycle = list(self._shuffle(self.horse_rng))
            cycles.append(cycle)
            cards.extend(cycle)
        selected = tuple(cards[:count])
        self.events.append(
            {
                "event": "SEPARATE_HORSE_DECK",
                "required": count,
                "cycles": cycles,
                "selected": list(selected),
            }
        )
        return selected

    def _prepare_next_hand(
        self,
        next_dealer: DealerState,
        virtual: bool,
    ) -> MahjongHand:
        previous = self.dealer_state
        self.dealer_state = next_dealer
        hand = self._new_hand(virtual=virtual)
        self.dealer_state = previous
        return hand

    def _ordinary_source(
        self,
        outcome: HandOutcome,
        winner: int,
        record,
        next_hand: MahjongHand | None,
    ) -> tuple[str, tuple[int, ...]]:
        if record.plan.source is HorseSource.OLD_WALL:
            return f"{outcome.hand_id}:OLD_WALL", outcome.old_wall
        if record.plan.source is HorseSource.NEXT_HAND:
            if next_hand is None:
                raise RuntimeError("next hand required")
            return f"{next_hand.hand_id}:DEAL", tuple(next_hand.players[winner].concealed)
        return f"{outcome.hand_id}:SEPARATE", self._separate_sequence(record.plan.required)

    def resolve(self, continue_play: bool = True) -> ResolvedHand:
        """结算当前终局；散场仍会生成仅用于结算的虚拟下一把发牌。"""
        outcome = self.current_hand.outcome
        if outcome is None:
            raise RuntimeError("hand is not terminal")
        if outcome.draw:
            self.book.record_draw(
                DrawRecord(
                    f"D:{outcome.hand_id}",
                    outcome.hand_id,
                    self.dealer_state.player,
                    self.dealer_state.continuation,
                )
            )
            next_dealer = self.dealer_state.after_draw()
            self.dealer_state = next_dealer
            if continue_play:
                self.current_hand = self._new_hand(virtual=False)
            return ResolvedHand(outcome, (0, 0, 0, 0), next_dealer, None)

        winners = tuple(item.player for item in outcome.winners)
        next_dealer = (
            self.dealer_state.after_win(winners[0])
            if len(winners) == 1
            else self.dealer_state.after_multi_win(winners)
        )
        continuation = next_dealer.continuation
        old_available = len(outcome.old_wall)
        next_hand: MahjongHand | None = None
        virtual = False
        if len(winners) == 1:
            item = outcome.winners[0]
            win = OrdinaryWin(
                f"S:{outcome.hand_id}",
                outcome.hand_id,
                item.player,
                self.dealer_state.player,
                continuation,
                item.score.multiplier,
                1,
                WinMethod(outcome.method),
                outcome.discarder,
            )
            if outcome.kong_provider is not None:
                record = self.book.begin_kong_draw(win, old_available, outcome.kong_provider)
            elif outcome.twelve_grounded is not None:
                record = self.book.begin_twelve_grounded(
                    win, old_available, outcome.twelve_grounded
                )
            else:
                record = self.book.begin(win, old_available)
            if record.plan.source is HorseSource.NEXT_HAND:
                virtual = not continue_play
                next_hand = self._prepare_next_hand(next_dealer, virtual=virtual)
            source_id, source = self._ordinary_source(
                outcome, item.player, record, next_hand
            )
            record = self.book.bind(win.settlement_id, source_id, source)
            receipt = self.book.pay(win.settlement_id)
            reveal = self.book.next_hand_reveal(win.settlement_id)
            if reveal is not None and next_hand is not None:
                # NOTE: 只公开实际参与前N张且命中的实体，不能泄露同牌种的其他暗牌。
                if record.selection is None:
                    raise RuntimeError("bound selection required")
                next_hand.players[item.player].revealed_retained.update(
                    record.selection.hits
                )
                self.events.append(reveal)
            source_kind = record.plan.source
        else:
            multi = MultiDiscardWin(
                f"M:{outcome.hand_id}",
                outcome.hand_id,
                self.dealer_state.player,
                continuation,
                1,
                outcome.discarder,
                tuple(
                    MultiWinner(item.player, item.score.multiplier)
                    for item in outcome.winners
                ),
            )
            record = self.book.begin_multi(multi, old_available)
            if record.plan.source is HorseSource.OLD_WALL:
                source_id, source = f"{outcome.hand_id}:OLD_WALL", outcome.old_wall
            else:
                source_id = f"{outcome.hand_id}:SEPARATE"
                source = self._separate_sequence(record.plan.required)
            self.book.bind_multi(multi.settlement_id, source_id, source)
            receipt = self.book.pay_multi(multi.settlement_id)
            source_kind = record.plan.source
        for seat, value in enumerate(receipt):
            self.scores[seat] += value
        self.dealer_state = next_dealer
        if continue_play:
            self.current_hand = next_hand if next_hand is not None else self._new_hand(False)
        return ResolvedHand(outcome, receipt, next_dealer, source_kind, virtual)
