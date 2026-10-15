"""固定初始庄位、零连庄、同墙四座位的受控单手评估；保存逐局和私有审计。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Callable

import torch
import numpy as np

from pdh_v2.rollout import BatchedSessions
from pdh_v2.rollout.parallel_env import ParallelSessions
from pdh_v2.training.collector import RolloutCollector
from pdh_v2.training.events import ReplayWriter
from pdh_v2.training.paired_metrics import summarize
from pdh_v2.training.loss_attribution import payment_attribution
from pdh_v2.training.responsibility_metrics import responsibility_row
from pdh_v2.training.wall_registry import wall_fingerprint


@torch.no_grad()
def paired_evaluate(model, device, output: Path, hands: int = 1024,
                    environments: int = 64, workers: int = 4, seed: int = 950_000,
                    progress: Callable[[int, int], None] | None = None,
                    check_stop: Callable[[], None] | None = None,
                    transport: str = 'pipe',
                    on_terminal: Callable | None = None,
                    opponent_factory: Callable | None = None) -> tuple[list[dict], dict]:
    """每个seed恰好四座位；不沿用按先完成手数排空的有偏评估分配。"""
    if hands < 4 or hands % 4 or environments < 1:
        raise ValueError("evaluation hands must be a positive multiple of four")
    output.mkdir(parents=True, exist_ok=False)
    rows: list[dict] = []
    started = time.monotonic()
    model.eval()
    replay = ReplayWriter(output/'hands.jsonl.gz')
    backend = None
    try:
        with (output/'results.jsonl').open('x', encoding='utf-8') as stream:
            for offset in range(0, hands, environments):
                if check_stop is not None:
                    check_stop()
                count = min(environments, hands-offset)
                seeds = tuple(seed+((offset+i)//4)*1_000_003 for i in range(count))
                if backend is None or backend.count != count:
                    if backend is not None:
                        backend.close()
                    backend = (ParallelSessions(count, seed, workers, seeds,
                               check_stop=check_stop, transport=transport)
                               if workers > 0 else BatchedSessions(count, seed, seeds))
                else:
                    backend.reset(seeds)
                # NOTE(rc5): default path is unchanged; each new wall wave resets opponents' GRUs.
                opponent = opponent_factory(count) if opponent_factory is not None else None
                hidden = model.initial_state(count, device)
                while not backend.all_inactive():
                    if check_stop is not None:
                        check_stop()
                    if progress is not None:
                        progress(len(rows), hands)
                    requests = backend.requests()
                    active = [r for r in requests if r.view.player == (offset+r.env_index)%4]
                    choices = {}
                    if active:
                        observations = torch.from_numpy(np.stack([
                            np.asarray(r.view.observation, dtype=np.float32)
                            for r in active])).to(device)
                        features, owners, counts = RolloutCollector._pack_action_features(active)
                        ids = torch.tensor([r.env_index for r in active], device=device)
                        logits, _, _, states = model.forward_step(observations,
                            hidden.index_select(0, ids), features.to(device), owners.to(device))
                        hidden.index_copy_(0, ids, states)
                        # NOTE: GPU结果一次回传，避免每桌一个.item()同步。
                        data = logits.cpu(); cursor = 0
                        for request, n in zip(active, counts):
                            choices[request.env_index] = int(data[cursor:cursor+n].argmax())
                            cursor += n
                    other = tuple(r for r in requests if r.env_index not in choices)
                    choices.update(backend.heuristic_actions(other) if opponent is None
                                   else opponent(backend, other))
                    terminals = backend.step(choices, drain=True)
                    if on_terminal is not None:
                        for terminal in terminals:
                            on_terminal(terminal)
                    replay.write_many(terminals, update=0)
                    for terminal in terminals:
                        seat = (offset+terminal.env_index)%4
                        won = seat in terminal.winners
                        if sum(terminal.receipt) != 0:
                            raise RuntimeError('non-zero-sum evaluation')
                        row = dict(wall_seed=seeds[terminal.env_index], candidate_seat=seat,
                            score=terminal.receipt[seat], win=int(won), draw=int(terminal.draw),
                            point_shoot=int(terminal.discarder == seat),
                            self_draw=int(won and terminal.method=='SELF_DRAW'),
                            discard_win=int(won and terminal.discarder is not None),
                            env_index=terminal.env_index, wave_offset=offset,
                            state_sha256=terminal.transcript.state_sha256,
                            action_events=len(terminal.transcript.actions),
                            wall_type_sha256=wall_fingerprint(terminal.transcript.wall),
                            **payment_attribution(terminal, seat))
                        row['responsibility'] = responsibility_row(row)
                        chain = (terminal.audit or {}).get('supply_chain')
                        row['supply_association_observed'] = chain is not None
                        row['candidate_supply_to_self_draw_winner'] = (
                            any(e['external_supply'] and e['provider'] == seat
                                and e['claimant'] in terminal.winners for e in chain)
                            if chain is not None and terminal.method == 'SELF_DRAW' else None)
                        rows.append(row)
                        stream.write(json.dumps(row, sort_keys=True)+'\n')
                        stream.flush()
    finally:
        if backend is not None:
            backend.close()
    rows.sort(key=lambda r:(r['wall_seed'],r['candidate_seat']))
    # NOTE: 明细仍以results为权威；索引只保存最差手局的key，避免复制一份牌谱。
    worst = sorted(rows, key=lambda row: (row['score'], row['wall_seed'],
                                        row['candidate_seat']))[:min(50, len(rows))]
    tail_keys = [{key: row[key] for key in ('wall_seed', 'candidate_seat', 'score',
                 'state_sha256')} for row in worst if row['score'] < 0]
    (output / 'worst_hands.json').write_text(json.dumps({
        'source': 'results.jsonl', 'keys': tail_keys,
        'scope': 'diagnostic_tail_index_not_an_independent_sample'}, indent=2) + '\n')
    summary = {**summarize(rows), 'seconds':time.monotonic()-started,
               'protocol':'controlled_single_hand_same_wall_four_seats_v1',
               'seed':seed, 'initial_dealer':0, 'initial_continuation':0,
               'unit':'complete_natural_hand_not_isolated_decision',
               'unique_starting_walls':len({r['wall_type_sha256'] for r in rows}),
               'seat_variants_per_wall':4,
               'action_events_total':sum(r['action_events'] for r in rows),
               'results_sha256':hashlib.sha256((output/'results.jsonl').read_bytes()).hexdigest()}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return rows, summary
