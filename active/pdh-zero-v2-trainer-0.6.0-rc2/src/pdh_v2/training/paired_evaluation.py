"""固定初始庄位、零连庄、同墙四座位的受控单手评估；保存逐局和私有审计。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Callable

import torch

from pdh_v2.rollout import BatchedSessions
from pdh_v2.rollout.parallel_env import ParallelSessions
from pdh_v2.training.collector import RolloutCollector
from pdh_v2.training.events import ReplayWriter
from pdh_v2.training.paired_metrics import summarize


@torch.no_grad()
def paired_evaluate(model, device, output: Path, hands: int = 1024,
                    environments: int = 64, workers: int = 4, seed: int = 950_000,
                    progress: Callable[[int, int], None] | None = None,
                    check_stop: Callable[[], None] | None = None) -> tuple[list[dict], dict]:
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
                    backend = (ParallelSessions(count, seed, workers, seeds, check_stop=check_stop)
                               if workers > 0 else BatchedSessions(count, seed, seeds))
                else:
                    backend.reset(seeds)
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
                        observations = torch.tensor([r.view.observation for r in active],
                                                    dtype=torch.float32, device=device)
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
                    choices.update(backend.heuristic_actions(other))
                    terminals = backend.step(choices, drain=True)
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
                            state_sha256=terminal.transcript.state_sha256)
                        rows.append(row)
                        stream.write(json.dumps(row, sort_keys=True)+'\n')
                        stream.flush()
    finally:
        if backend is not None:
            backend.close()
    rows.sort(key=lambda r:(r['wall_seed'],r['candidate_seat']))
    summary = {**summarize(rows), 'seconds':time.monotonic()-started,
               'protocol':'controlled_single_hand_same_wall_four_seats_v1',
               'seed':seed, 'initial_dealer':0, 'initial_continuation':0,
               'results_sha256':hashlib.sha256((output/'results.jsonl').read_bytes()).hexdigest()}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return rows, summary
