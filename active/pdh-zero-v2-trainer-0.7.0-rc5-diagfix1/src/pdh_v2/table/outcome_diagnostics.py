"""PDH research code: candidate-only patterns/income/opportunities, never an Actor reward."""
from __future__ import annotations

from collections import Counter
from itertools import zip_longest
import json
from pathlib import Path

import numpy as np

from pdh_v2.strategy import ROUTE_NAMES, route_state
from pdh_v2.table.data import records, restore_record
from pdh_v2.training.candidate_store import file_digest
from pdh_v2.training.refinement_control import atomic_json


def candidate_outcome(record: dict, result: dict) -> dict:
    """Attribute only this candidate's win; opponent patterns cannot inflate its statistics."""
    seat = result['candidate_seat']
    winners = [w for w in record['audit']['outcome']['winners'] if w['player'] == seat]
    if bool(winners) != bool(result['win']) or len(winners) > 1:
        raise ValueError('rc5.candidate_winner_identity_mismatch')
    incoming = [p for p in result['payments'] if p['winner'] == seat]
    outgoing = sum(p['amount'] for p in result['payments'] if p['payer'] == seat)
    income = sum(p['amount'] for p in incoming)
    if income-outgoing != result['score']:
        raise ValueError('rc5.candidate_income_reconstruction_mismatch')
    score = winners[0]['score'] if winners else None
    if score and any(p['multiplier'] != score['multiplier'] for p in incoming):
        raise ValueError('rc5.candidate_multiplier_mismatch')
    return dict(won=bool(winners), income=income, outgoing=outgoing,
                combination='+'.join(sorted(i['pattern'] for i in score['items']))
                    if score else None,
                multiplier=score['multiplier'] if score else None,
                method=record['method'] if winners else None,
                horse_hits=sorted({p['horse_hits'] for p in incoming}))


def summarize_candidate_outcomes(directory: Path) -> dict:
    """Summarize all synchronized terminal/result rows; never select only difficult hands."""
    replay, results = directory/'hands.jsonl.gz', directory/'results.jsonl'
    combinations, multipliers, income = Counter(), Counter(), Counter()
    actions, horse_hits = Counter(), Counter()
    starting = np.zeros(8)
    count = wins = draws = 0
    with results.open() as stream:
        for record, line in zip_longest(records(replay), stream):
            if record is None or line is None:
                raise ValueError('rc5.evaluation_diagnostic_row_count_mismatch')
            row = json.loads(line)
            if (row['state_sha256'] != record['transcript']['state_sha256']
                    or row['wall_seed'] != record['audit']['session_seed']
                    or row['env_index'] != record['env_index']):
                raise ValueError('rc5.evaluation_diagnostic_row_identity_mismatch')
            own = candidate_outcome(record, row)
            seat = row['candidate_seat']
            session, _ = restore_record(record)
            starting += route_state(session.current_hand.players[seat].concealed, ()).potentials
            own_actions = (a['kind'] for a in record['transcript']['actions']
                           if a['player'] == seat)
            actions.update(own_actions)
            count, wins, draws = count+1, wins+int(own['won']), draws+int(record['draw'])
            if own['won']:
                combinations[own['combination']] += 1
                multipliers[str(own['multiplier'])] += 1
                income[own['method']] += own['income']
                horse_hits.update(map(str, own['horse_hits']))
    report = dict(scope='candidate_only_posthoc_not_causal_attribution_or_reward', hands=count,
                  candidate_wins=wins, table_draws=draws, winning_combinations=dict(combinations),
                  winning_multipliers=dict(multipliers), candidate_income_by_method=dict(income),
                  candidate_actions=dict(actions), candidate_win_horse_hits=dict(horse_hits),
                  initial_route_mean=dict(zip(ROUTE_NAMES, (starting/max(1, count)).tolist())),
                  replay_sha256=file_digest(replay), results_sha256=file_digest(results),
                  tail_replay_index='worst_hands.json', modifies_training_or_gate=False)
    atomic_json(directory/'candidate-outcomes.json', report)
    return report