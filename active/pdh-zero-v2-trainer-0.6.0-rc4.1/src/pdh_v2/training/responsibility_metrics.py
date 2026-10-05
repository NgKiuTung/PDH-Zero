"""PDH研究代码：账本自责分与百手防御率；责任不是可避免性或因果贡献。"""
from __future__ import annotations

import math
import numpy as np

DIRECT_CAUSES = frozenset(("ordinary_discard", "discard_multi", "rob_kong",
                          "twelve_grounded_liability", "kong_draw_liability"))
SELF_DRAW_CAUSE = "ordinary_self_draw"


def responsibility_row(row: dict) -> dict:
    """对一个玩家—手局逐笔归类；多响金额累加、责任事件手局去重。"""
    if row.get("settlement_audit_verified") is not True or "payments" not in row:
        raise ValueError("responsibility.payment_evidence_missing")
    seat = row["candidate_seat"]
    if type(seat) is not int or not 0 <= seat < 4:
        raise ValueError("responsibility.seat_invalid")
    direct = base = horse = share = income = outgoing = 0
    balances = [0, 0, 0, 0]
    for payment in row["payments"]:
        cause = payment["cause"]
        if cause not in DIRECT_CAUSES | {SELF_DRAW_CAUSE}:
            raise ValueError("responsibility.unknown_cause")
        payer, winner = payment["payer"], payment["winner"]
        if (type(payer) is not int or type(winner) is not int
                or payer not in range(4) or winner not in range(4) or payer == winner):
            raise ValueError("responsibility.payment_seat_invalid")
        amount, principal, uplift = (payment[k] for k in
                                     ("amount", "before_horses", "horse_uplift"))
        if any(type(x) is not int or x < 0 for x in (amount, principal, uplift)):
            raise ValueError("responsibility.amount_invalid")
        if principal + uplift != amount:
            raise ValueError("responsibility.decomposition_mismatch")
        balances[payer] -= amount
        balances[winner] += amount
        if winner == seat:
            income += amount
        if payer != seat:
            continue
        outgoing += amount
        if cause in DIRECT_CAUSES:
            direct += amount
            base += principal
            horse += uplift
        else:
            share += amount
    if balances[seat] != row["score"] or outgoing != row.get("candidate_loss", outgoing):
        raise ValueError("responsibility.score_mismatch")
    return {"direct_loss": direct, "direct_principal": base, "direct_horse_uplift": horse,
            "ordinary_self_draw_share": share, "total_outgoing": outgoing,
            "total_income": income, "direct_loss_hand": int(direct > 0),
            "scope": "rules_liability_not_causal_or_avoidable_error"}


def _distribution(values: list[int]) -> dict:
    data = np.asarray(values, dtype=float)
    positive = data[data > 0]
    return {"denominator": "all_candidate_seat_hands", "p95": float(np.quantile(data, .95)),
            "p99": float(np.quantile(data, .99)), "maximum": int(data.max()),
            "positive_hands": len(positive),
            "mean_if_positive": float(positive.mean()) if len(positive) else None}


def summarize_responsibility(rows: list[dict]) -> dict:
    """完整账本加总及最差实际净分手局分解；供牌关联单列，不改奖励。"""
    if not rows:
        raise ValueError("responsibility.empty_rows")
    derived = [responsibility_row(row) for row in rows]
    sums = {key: sum(r[key] for r in derived) for key in (
        "direct_loss", "direct_principal", "direct_horse_uplift", "ordinary_self_draw_share",
        "total_outgoing", "total_income", "direct_loss_hand")}
    order = sorted(range(len(rows)), key=lambda i: (
        rows[i]["score"], rows[i].get("wall_seed", i), rows[i]["candidate_seat"]))
    tails = {}
    for fraction in (.01, .05):
        indices = order[:max(1, math.ceil(len(rows)*fraction))]
        causes = {}
        for i in indices:
            for payment in rows[i]["payments"]:
                if payment["payer"] == rows[i]["candidate_seat"]:
                    causes[payment["cause"]] = causes.get(payment["cause"], 0)+payment["amount"]
        tails[str(fraction)] = {"hands": len(indices), "losses_by_cause": causes,
            "direct_loss": sum(derived[i]["direct_loss"] for i in indices),
            "ordinary_self_draw_share": sum(
                derived[i]["ordinary_self_draw_share"] for i in indices),
            "scope": "lowest_net_score_hands_ties_by_wall_seat"}
    return {"schema_version": 1, "candidate_seat_hands": len(rows), **sums,
        "defense_loss_per_100_seat_hands": 100*sums["direct_loss"]/len(rows),
        "direct_loss_hand_rate": sums["direct_loss_hand"]/len(rows),
        "direct_loss_distribution": _distribution([r["direct_loss"] for r in derived]),
        "ordinary_self_draw_distribution": _distribution(
            [r["ordinary_self_draw_share"] for r in derived]),
        "net_score_tail_breakdown": tails,
        "scope": "monitoring_only_not_reward_or_gate",
        "ordinary_self_draw_is_not_proven_luck_or_zero_contribution": True}
