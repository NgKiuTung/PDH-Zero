"""重放手局并独立重建本手结算；验证跨局亮马初始信息及最终付款。"""
from __future__ import annotations

import hashlib
import json

from pdh_v2.config import SOURCE_SHA256
from pdh_v2.game.actions import Action, ActionKind
from pdh_v2.game.engine import MahjongHand
from pdh_v2.journal import SNAPSHOT_SCHEMA_VERSION, load_snapshot
from pdh_v2.replay import HandTranscript


def _canonical(value) -> bytes:
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,
                      allow_nan=False).encode()


def audit_terminal_record(record: dict) -> dict:
    """校验新版本私有牌谱；缺少结算事实时拒绝把它当完整审计通过。"""
    transcript = HandTranscript.from_json(_canonical(record['transcript']))
    audit = record.get('audit')
    if not isinstance(audit,dict):
        raise ValueError('record has no settlement audit evidence')
    reveals = audit['initial_revealed_retained']
    if len(reveals) != 4:
        raise ValueError('invalid public initial state')
    hand = MahjongHand(transcript.hand_id,transcript.dealer,transcript.wall)
    for player,cards in zip(hand.players,reveals):
        if not set(cards).issubset(player.concealed):
            raise ValueError('initial public card does not belong to hand')
        player.revealed_retained.update(cards)
    for item in transcript.actions:
        hand.step(Action(item['player'],ActionKind(item['kind']),tuple(item['tiles']),
                         item['meld_index']))
    if hashlib.sha256(_canonical(hand.state_digest())).hexdigest() != transcript.state_sha256:
        raise ValueError('hand replay digest mismatch')
    payload={'schema_version':SNAPSHOT_SCHEMA_VERSION,'rule_source_sha256':SOURCE_SHA256,
             'events':audit['settlement_events']}
    envelope={'payload':payload,'sha256':hashlib.sha256(_canonical(payload)).hexdigest()}
    book=load_snapshot(_canonical(envelope))
    if list(book.balances) != record['receipt']:
        raise ValueError('settlement receipt mismatch')
    return {'hand_id':transcript.hand_id,'status':'PASS','receipt':record['receipt']}
