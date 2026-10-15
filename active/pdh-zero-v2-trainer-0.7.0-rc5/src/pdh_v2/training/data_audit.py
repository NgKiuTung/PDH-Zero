"""PDH研究代码：离线验证手局粒度、真实牌墙重复与样本级隔离；不训练模型。"""
from __future__ import annotations

from collections import Counter
import gzip
import json
from pathlib import Path
import statistics

from pdh_v2.replay.audit import audit_terminal_record
from pdh_v2.training.wall_registry import wall_fingerprint


def _read_replays(path: Path, replay_samples: int) -> tuple[dict, set[str]]:
    fingerprints: Counter[str] = Counter()
    lengths: list[int] = []
    verified = 0
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            transcript = row['transcript']
            fingerprints[wall_fingerprint(transcript['wall'])] += 1
            lengths.append(len(transcript['actions']))
            if not (row.get('audit') or {}).get('outcome'):
                raise ValueError('data_audit.missing_terminal_outcome')
            if verified < replay_samples:
                audit_terminal_record(row)
                verified += 1
    if not lengths or min(lengths) < 1:
        raise ValueError('data_audit.empty_or_actionless_replay')
    return {'complete_hand_records': len(lengths),
            'unique_starting_type_walls': len(fingerprints),
            'wall_multiplicity': dict(Counter(fingerprints.values())),
            'action_events_min': min(lengths), 'action_events_max': max(lengths),
            'action_events_mean': statistics.mean(lengths),
            'action_events_median': statistics.median(lengths),
            'sampled_full_rule_replays_verified': verified}, set(fingerprints)


def audit_run(run: Path, replay_samples: int = 8) -> dict:
    """全量数手数和指纹，抽样按冻结规则重放；不把抽样通过写成全量重放。"""
    if replay_samples < 0:
        raise ValueError('data_audit.negative_replay_samples')
    paths = sorted((run / 'replays').glob('*.jsonl.gz'))
    paths += sorted((run / 'evaluations').glob('*/hands.jsonl.gz'))
    if not paths:
        raise ValueError('data_audit.no_replays')
    stats = {}
    training = set()
    eval_sets = {}
    for path in paths:
        summary, walls = _read_replays(path, replay_samples)
        relative = str(path.relative_to(run))
        stats[relative] = summary
        if path.parent.name == 'replays':
            training.update(walls)
        else:
            eval_sets[relative] = walls
    overlap = {name: len(training & walls) for name, walls in eval_sets.items()}
    return {'status': 'PASS' if not any(overlap.values()) else 'OVERLAP',
            'scope': 'this_run_logged_starting_walls_not_all_model_history',
            'replays': stats, 'training_development_starting_wall_overlap': overlap,
            'fixed_development_reuse_is_intentional': True,
            'generalization_verified': False}
