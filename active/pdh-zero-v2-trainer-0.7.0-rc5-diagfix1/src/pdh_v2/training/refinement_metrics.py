"""统一指标分母；分开局数、学习座位-局、唯一弃牌与训练遍历次数。"""
from __future__ import annotations
from pdh_v2.game.actions import ActionKind


def collection_metrics(collected, collection_seconds: float) -> dict:
    """零奖励局和无决策座位都计入实际手局，不能让分母随课程静默变化。"""
    positions = collected.learner_seat_hands
    count = collected.hands
    actions = {kind.value: collected.action_counts[i] for i,kind in enumerate(ActionKind)}
    return {
        'hands':count, 'learner_seat_hands':positions,
        'learner_count_histogram':list(collected.learner_seat_histogram),
        'learner_decisions':collected.decisions,
        'environment_decisions':collected.environment_decisions,
        'heuristic_decisions':collected.heuristic_decisions,
        'learner_mean_score':collected.learner_score_sum/max(1,positions),
        'learner_win_rate_per_seat':collected.learner_win_count/max(1,positions),
        'learner_point_shoot_rate_per_seat':collected.learner_point_shoots/max(1,positions),
        'any_learner_win_rate_per_table':collected.learner_wins/max(1,count),
        'any_learner_point_shoot_rate_per_table':collected.learner_point_shoots/max(1,count),
        'draw_rate':collected.draws/max(1,count), 'raw_score_sum':list(collected.raw_score_sum),
        'action_counts':actions, 'pong_opportunities':collected.pong_opportunities,
        'pong_acceptance_rate':actions['PONG']/collected.pong_opportunities
            if collected.pong_opportunities else None,
        'win_opportunities':collected.win_opportunities, 'wins_taken':collected.wins_taken,
        'win_capture_rate':collected.wins_taken/collected.win_opportunities
            if collected.win_opportunities else None,
        'risk_samples_unique':collected.risk_samples,
        'risk_positives_unique':collected.risk_positives,
        'collection_seconds':collection_seconds,
        'learner_decisions_per_collection_second':collected.decisions/collection_seconds,
    }
