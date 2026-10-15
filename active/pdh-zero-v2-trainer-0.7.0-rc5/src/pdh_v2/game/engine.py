"""无吃牌、支持多响与抢加杠的136张参考手局状态机。"""
from dataclasses import asdict
from itertools import combinations
from typing import Iterable

from pdh_v2.config import PLAYERS
from pdh_v2.contracts import RuleError, integer, text
from pdh_v2.game.actions import Action, ActionKind
from pdh_v2.game.digest import hand_state_digest
from pdh_v2.game.state import (
    HandOutcome,
    KongContext,
    MeldRecord,
    PendingAddedKong,
    PendingDiscard,
    Phase,
    PlayerState,
    ResponseState,
    WinnerResult,
)
from pdh_v2.game.wall import DrawnTile, Wall
from pdh_v2.qualifications import WinMethod
from pdh_v2.responsibility import (
    KongDrawLiability,
    TwelveGroundedLiability,
    lock_kong_draw_liability,
    lock_twelve_grounded_liability,
)
from pdh_v2.scoring import ScoreContext, score_hand
from pdh_v2.tiles import Meld, MeldKind, WinningHand, tile_type


class MahjongHand:
    """单手确定性状态机；所有外部动作必须先出现在legal_actions中。"""

    def __init__(self, hand_id: str, dealer: int, wall_order: tuple[int, ...]) -> None:
        self.hand_id = text(hand_id, "hand_id")
        self.dealer = integer(dealer, "dealer", 0, PLAYERS - 1)
        self.wall = Wall(wall_order)
        dealt = self.wall.deal(self.dealer)
        self.players = [PlayerState(list(cards)) for cards in dealt]
        self.current_player = self.dealer
        self.phase = Phase.ACTION
        self.pending_discard: PendingDiscard | None = None
        self.pending_added_kong: PendingAddedKong | None = None
        self.response: ResponseState | None = None
        self.last_draw = DrawnTile(
            self.dealer,
            dealt[self.dealer][-1],
            "FRONT",
            self.wall.remaining + 1,
        )
        self.kong_context: KongContext | None = None
        self.twelve_liabilities: dict[int, TwelveGroundedLiability] = {}
        self.pass_win_locked = [False] * PLAYERS
        self.outcome: HandOutcome | None = None
        self.events: list[dict[str, object]] = [
            {"event": "DEAL", "dealer": self.dealer, "hands": [list(x) for x in dealt]}
        ]

    def _melds(self, player: int) -> tuple[Meld, ...]:
        return tuple(record.meld for record in self.players[player].melds)

    def _winning_hand(self, player: int, incoming: int | None = None) -> WinningHand:
        cards = tuple(self.players[player].concealed + ([] if incoming is None else [incoming]))
        return WinningHand(cards, self._melds(player))

    def _score(
        self,
        player: int,
        method: WinMethod,
        incoming: int | None = None,
    ):
        if incoming is None and (
            self.last_draw.player != player
            or self.last_draw.tile not in self.players[player].concealed
        ):
            return None
        winning = incoming if incoming is not None else self.last_draw.tile
        wall_before = self.last_draw.wall_before
        return score_hand(
            ScoreContext(
                self._winning_hand(player, incoming),
                winning,
                method,
                wall_before,
                self.kong_context is not None and incoming is None,
            )
        )

    def _matching(self, player: int, kind: int) -> tuple[int, ...]:
        return tuple(card for card in self.players[player].concealed if tile_type(card) == kind)

    def _action_phase_actions(self, player: int) -> list[Action]:
        if player != self.current_player:
            return []
        state = self.players[player]
        result = [Action(player, ActionKind.DISCARD, (card,)) for card in state.concealed]
        if self._score(player, WinMethod.SELF_DRAW) is not None:
            result.append(Action(player, ActionKind.WIN))
        by_kind: dict[int, list[int]] = {}
        for card in state.concealed:
            by_kind.setdefault(tile_type(card), []).append(card)
        if self.wall.remaining > 0:
            for cards in by_kind.values():
                if len(cards) == 4:
                    result.append(Action(player, ActionKind.CLOSED_KONG, tuple(cards)))
        for index, record in enumerate(state.melds):
            if self.wall.remaining > 0 and record.meld.kind is MeldKind.PONG:
                matches = self._matching(player, tile_type(record.meld.tiles[0]))
                for card in matches:
                    result.append(Action(player, ActionKind.ADDED_KONG, (card,), index))
        return result

    def _discard_response_actions(self, player: int) -> list[Action]:
        if self.response is None or not self.response.queue or player != self.response.queue[0]:
            return []
        pending = self.pending_discard
        if pending is None:
            raise RuleError("state.invalid")
        result = [Action(player, ActionKind.PASS)]
        if (
            not self.pass_win_locked[player]
            and self._score(player, WinMethod.DISCARD_WIN, pending.tile) is not None
        ):
            result.append(Action(player, ActionKind.WIN))
        matches = self._matching(player, tile_type(pending.tile))
        result.extend(
            Action(player, ActionKind.PONG, tuple(choice))
            for choice in combinations(matches, 2)
        )
        if self.wall.remaining > 0:
            result.extend(
                Action(player, ActionKind.OPEN_KONG, tuple(choice))
                for choice in combinations(matches, 3)
            )
        return result

    def _added_response_actions(self, player: int) -> list[Action]:
        if self.response is None or not self.response.queue or player != self.response.queue[0]:
            return []
        pending = self.pending_added_kong
        if pending is None:
            raise RuleError("state.invalid")
        result = [Action(player, ActionKind.PASS)]
        if (
            not self.pass_win_locked[player]
            and self._score(player, WinMethod.ROB_KONG, pending.tile) is not None
        ):
            result.append(Action(player, ActionKind.WIN))
        return result

    def legal_actions(self, player: int) -> tuple[Action, ...]:
        """返回当前玩家全部真实实体动作。"""
        integer(player, "player", 0, PLAYERS - 1)
        if self.phase is Phase.TERMINAL:
            return ()
        if self.phase is Phase.ACTION:
            return tuple(self._action_phase_actions(player))
        if self.phase is Phase.DISCARD_RESPONSE:
            return tuple(self._discard_response_actions(player))
        return tuple(self._added_response_actions(player))

    def _record(self, action: Action) -> None:
        self.events.append({
            "event": "ACTION",
            "action": {
                "player": action.player,
                "kind": action.kind.value,
                "tiles": list(action.tiles),
                "meld_index": action.meld_index,
            },
        })

    def step(self, action: Action) -> HandOutcome | None:
        """执行一个合法动作；终局时返回HandOutcome。"""
        if action not in self.legal_actions(action.player):
            raise RuleError("action.illegal")
        self._record(action)
        if self.phase is Phase.ACTION:
            self._step_action(action)
        else:
            self._step_response(action)
        return self.outcome

    def _step_action(self, action: Action) -> None:
        if action.kind is ActionKind.DISCARD:
            self._discard(action.player, action.tiles[0])
        elif action.kind is ActionKind.CLOSED_KONG:
            self._closed_kong(action.player, action.tiles)
        elif action.kind is ActionKind.ADDED_KONG:
            self._begin_added_kong(action)
        elif action.kind is ActionKind.WIN:
            self._self_draw_win(action.player)
        else:
            raise RuleError("action.illegal")

    def _discard(self, player: int, card: int) -> None:
        state = self.players[player]
        state.concealed.remove(card)
        state.discards.append(card)
        self.kong_context = None
        self.pending_discard = PendingDiscard(player, card)
        queue = [(player + offset) % PLAYERS for offset in range(1, PLAYERS)]
        self.response = ResponseState(player, queue)
        self.phase = Phase.DISCARD_RESPONSE

    def _closed_kong(self, player: int, cards: tuple[int, ...]) -> None:
        state = self.players[player]
        for card in cards:
            state.concealed.remove(card)
        state.melds.append(MeldRecord(Meld(MeldKind.CLOSED_KONG, cards)))
        self.kong_context = KongContext(MeldKind.CLOSED_KONG.value, player, None)
        self._supplement_draw(player)

    def _begin_added_kong(self, action: Action) -> None:
        if action.meld_index is None:
            raise RuleError("action.meld_index")
        self.pending_added_kong = PendingAddedKong(
            action.player,
            action.tiles[0],
            action.meld_index,
        )
        queue = [(action.player + offset) % PLAYERS for offset in range(1, PLAYERS)]
        self.response = ResponseState(action.player, queue)
        self.phase = Phase.ADDED_KONG_RESPONSE

    def _step_response(self, action: Action) -> None:
        if self.response is None:
            raise RuleError("state.invalid")
        had_win = any(item.kind is ActionKind.WIN for item in self.legal_actions(action.player))
        if had_win and action.kind is not ActionKind.WIN:
            self.pass_win_locked[action.player] = True
        self.response.responses[action.player] = action
        self.response.queue.pop(0)
        if self.response.queue:
            return
        if self.phase is Phase.DISCARD_RESPONSE:
            self._resolve_discard_response()
        else:
            self._resolve_added_response()

    def _resolve_discard_response(self) -> None:
        if self.response is None or self.pending_discard is None:
            raise RuleError("state.invalid")
        wins = [
            action.player
            for action in self.response.responses.values()
            if action.kind is ActionKind.WIN
        ]
        if wins:
            self._finish_external_wins(wins, WinMethod.DISCARD_WIN, self.pending_discard.tile)
            return
        claims = [
            action for action in self.response.responses.values()
            if action.kind in (ActionKind.PONG, ActionKind.OPEN_KONG)
        ]
        if not claims:
            source = self.pending_discard.player
            self._clear_response()
            self._normal_draw((source + 1) % PLAYERS)
            return
        # NOTE: 房规碰优先于杠；同类按距离弃牌者最近者确定。
        pongs = [action for action in claims if action.kind is ActionKind.PONG]
        candidates = pongs or claims
        source = self.pending_discard.player
        chosen = min(candidates, key=lambda item: (item.player - source) % PLAYERS)
        self._apply_discard_claim(chosen)

    def _apply_discard_claim(self, action: Action) -> None:
        if self.pending_discard is None:
            raise RuleError("state.invalid")
        source = self.pending_discard.player
        card = self.pending_discard.tile
        state = self.players[action.player]
        for own in action.tiles:
            state.concealed.remove(own)
        kind = MeldKind.PONG if action.kind is ActionKind.PONG else MeldKind.OPEN_KONG
        liability = lock_twelve_grounded_liability(
            self.hand_id,
            action.player,
            source,
            self._melds(action.player),
            kind,
        )
        state.melds.append(MeldRecord(Meld(kind, action.tiles + (card,)), source))
        if liability is not None:
            self.twelve_liabilities[action.player] = liability
        self.players[source].discards.remove(card)
        self._clear_response()
        self.current_player = action.player
        self.phase = Phase.ACTION
        if kind is MeldKind.OPEN_KONG:
            self.kong_context = KongContext(kind.value, action.player, source)
            self._supplement_draw(action.player)

    def _resolve_added_response(self) -> None:
        if self.response is None or self.pending_added_kong is None:
            raise RuleError("state.invalid")
        wins = [
            action.player for action in self.response.responses.values()
            if action.kind is ActionKind.WIN
        ]
        if wins:
            self._finish_external_wins(wins, WinMethod.ROB_KONG, self.pending_added_kong.tile)
            return
        pending = self.pending_added_kong
        state = self.players[pending.player]
        record = state.melds[pending.meld_index]
        if record.meld.kind is not MeldKind.PONG:
            raise RuleError("state.invalid")
        state.concealed.remove(pending.tile)
        state.melds[pending.meld_index] = MeldRecord(
            Meld(MeldKind.ADDED_KONG, record.meld.tiles + (pending.tile,)),
            record.provider,
        )
        self._clear_response()
        self.kong_context = KongContext(MeldKind.ADDED_KONG.value, pending.player, None)
        self._supplement_draw(pending.player)

    def _clear_response(self) -> None:
        self.pending_discard = None
        self.pending_added_kong = None
        self.response = None

    def _normal_draw(self, player: int) -> None:
        if self.wall.remaining == 0:
            self._finish_draw()
            return
        draw = self.wall.draw_front(player)
        self.players[player].concealed.append(draw.tile)
        self.pass_win_locked[player] = False
        self.last_draw = draw
        self.kong_context = None
        self.current_player = player
        self.phase = Phase.ACTION
        self.events.append({"event": "DRAW", **asdict(draw)})

    def _supplement_draw(self, player: int) -> None:
        if self.wall.remaining == 0:
            self._finish_draw()
            return
        draw = self.wall.draw_back(player)
        self.players[player].concealed.append(draw.tile)
        self.pass_win_locked[player] = False
        self.last_draw = draw
        self.current_player = player
        self.phase = Phase.ACTION
        self.events.append({"event": "SUPPLEMENT_DRAW", **asdict(draw)})

    def _self_draw_win(self, player: int) -> None:
        result = self._score(player, WinMethod.SELF_DRAW)
        if result is None:
            raise RuleError("action.illegal")
        source = self.last_draw.source
        old_wall = self.wall.available_after(source)
        provider = None
        if self.kong_context is not None:
            liability = lock_kong_draw_liability(
                self.hand_id,
                player,
                self.kong_context.provider,
                MeldKind(self.kong_context.kind),
            )
            provider = liability.responsible if liability is not None else None
        twelve = None if provider is not None else self.twelve_liabilities.get(player)
        self.outcome = HandOutcome(
            self.hand_id,
            WinMethod.SELF_DRAW.value,
            (WinnerResult(player, result),),
            None,
            False,
            source,
            old_wall,
            twelve,
            provider,
        )
        self.phase = Phase.TERMINAL

    def _finish_external_wins(
        self,
        winners: Iterable[int],
        method: WinMethod,
        incoming: int,
    ) -> None:
        results = []
        for player in winners:
            result = self._score(player, method, incoming)
            if result is None:
                raise RuleError("action.illegal")
            results.append(WinnerResult(player, result))
        source_player = (
            self.pending_discard.player
            if self.pending_discard is not None
            else self.pending_added_kong.player
        )
        source = self.last_draw.source
        self.outcome = HandOutcome(
            self.hand_id,
            method.value,
            tuple(results),
            source_player,
            False,
            source,
            self.wall.available_after(source),
        )
        self.phase = Phase.TERMINAL
        self._clear_response()

    def _finish_draw(self) -> None:
        source = self.last_draw.source
        self.outcome = HandOutcome(
            self.hand_id,
            "DRAW",
            (),
            None,
            True,
            source,
            self.wall.available_after(source),
        )
        self.phase = Phase.TERMINAL


    def validate_invariants(self) -> None:
        """验证136张实体恰好存在一处；用于随机压力测试和回放门禁。"""
        cards: list[int] = list(self.wall.order[self.wall.front : self.wall.back + 1])
        for player in self.players:
            cards.extend(player.concealed)
            cards.extend(player.discards)
            for record in player.melds:
                cards.extend(record.meld.tiles)
        if len(cards) != 136 or len(set(cards)) != 136 or set(cards) != set(range(136)):
            raise RuleError("state.tiles")

    def state_digest(self) -> dict[str, object]:
        """返回完整可规范JSON化状态摘要，供牌谱重放比较。"""
        return hand_state_digest(self)
