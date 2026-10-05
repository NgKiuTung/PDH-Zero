"""私有结算审计派生支付来源；只用于评估，不把隐藏真值送入Actor。"""
from __future__ import annotations

import hashlib
import json

from pdh_v2.config import SOURCE_SHA256
from pdh_v2.journal import SNAPSHOT_SCHEMA_VERSION, load_snapshot
from pdh_v2.qualifications import WinMethod
from pdh_v2.responsibility import KongDrawLiability, TwelveGroundedLiability


def _book(events):
    payload = dict(schema_version=SNAPSHOT_SCHEMA_VERSION,
                   rule_source_sha256=SOURCE_SHA256, events=events)
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':'),
                     ensure_ascii=False, allow_nan=False).encode()
    return load_snapshot(json.dumps({'payload': payload,
                                    'sha256': hashlib.sha256(raw).hexdigest()}).encode())


def payment_attribution(terminal, seat: int) -> dict:
    """重建冻结账本后拆出每笔支付，必须与四家receipt完全对齐。"""
    if terminal.audit is None:
        raise ValueError('loss attribution requires settlement audit')
    events = terminal.audit['settlement_events']
    book = _book(events)
    if tuple(book.balances) != tuple(terminal.receipt):
        raise ValueError('audit/receipt mismatch')
    payments = []
    for event in events:
        if event['op'] == 'PAY_MULTI':
            rec = book.get_multi(event['settlement_id'])
            for winner in rec.win.winners:
                hits = rec.selection.hit_count_for(rec.win.relative_seat(winner.winner))
                payments.append(_payment(rec.win.discarder, winner.winner,
                    rec.win.base_units, winner.multiplier, hits, 1,
                    'rob_kong' if terminal.method == 'ROB_KONG' else 'discard_multi',
                    rec.plan.source.value, rec.plan.continuation))
        elif event['op'] == 'PAY':
            rec = book.get(event['settlement_id'])
            reason = ('ordinary_self_draw' if rec.win.method is WinMethod.SELF_DRAW
                      else 'rob_kong' if rec.win.method is WinMethod.ROB_KONG
                      else 'ordinary_discard')
            if rec.win.method is WinMethod.SELF_DRAW:
                if isinstance(rec.liability, KongDrawLiability):
                    reason = 'kong_draw_liability'
                elif isinstance(rec.liability, TwelveGroundedLiability):
                    reason = 'twelve_grounded_liability'
            unit = rec.win.base_units * rec.win.multiplier * (rec.selection.hit_count + 1)
            for payer, value in enumerate(rec.receipt):
                if value < 0:
                    if (-value) % unit:
                        raise ValueError('non-integral liability share')
                    payments.append(_payment(payer, rec.win.winner, rec.win.base_units,
                        rec.win.multiplier, rec.selection.hit_count, (-value) // unit,
                        reason, rec.plan.source.value, rec.plan.continuation))
    reconstructed = [0, 0, 0, 0]
    for payment in payments:
        reconstructed[payment['payer']] -= payment['amount']
        reconstructed[payment['winner']] += payment['amount']
    if tuple(reconstructed) != tuple(terminal.receipt):
        raise ValueError('payment decomposition mismatch')
    losses = [p for p in payments if p['payer'] == seat]
    return {'payments': payments, 'candidate_loss': sum(p['amount'] for p in losses),
            'candidate_loss_before_horses': sum(p['before_horses'] for p in losses),
            'candidate_loss_horse_uplift': sum(p['horse_uplift'] for p in losses),
            'settlement_audit_verified': True}


def _payment(payer, winner, base, multiplier, hits, shares, reason, source, continuation):
    before = base * multiplier * shares
    return dict(payer=payer, winner=winner, amount=before * (hits + 1),
                before_horses=before, horse_uplift=before * hits, horse_hits=hits,
                multiplier=multiplier, shares=shares, cause=reason,
                horse_source=source, continuation=continuation)


def summarize_losses(rows: list[dict]) -> dict:
    """按互斥支付原因累计；倍率与马只作正交维度，避免重复加总损失。"""
    causes = {}
    for row in rows:
        for payment in row.get('payments', []):
            if payment['payer'] != row['candidate_seat']:
                continue
            item = causes.setdefault(payment['cause'], {'payments': 0, 'amount': 0,
                'before_horses': 0, 'horse_uplift': 0})
            item['payments'] += 1
            for name in ('amount', 'before_horses', 'horse_uplift'):
                item[name] += payment[name]
    return {'losses_by_cause': causes,
            'negative_score_hands': sum(row['score'] < 0 for row in rows),
            'conditional_mean_loss': (
                sum(-row['score'] for row in rows if row['score'] < 0)
                / max(1, sum(row['score'] < 0 for row in rows))),
            'loss_audit_scope': 'recorded_settlement_not_inferred_from_score'}