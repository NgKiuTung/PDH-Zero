"""从完整逐局开发评估重算门禁；不把人工中断或局部文件当作晋级。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pdh_v2.training.paired_metrics import development_gate, paired_comparison


def _read_rows(path: Path) -> list[dict]:
    summary = json.loads((path/'summary.json').read_text(encoding='utf-8'))
    raw = (path/'results.jsonl').read_bytes()
    if hashlib.sha256(raw).hexdigest() != summary['results_sha256']:
        raise ValueError('evaluation results hash mismatch')
    rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    if len(rows) != summary['hands']:
        raise ValueError('evaluation count mismatch')
    return rows


def review_run(run: Path) -> dict:
    """只读审查：PASS仍不授权长训练；INCOMPLETE必须保留实际状态。"""
    manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('scope') != 'solo_stability_refinement_candidate':
        raise ValueError('not a refinement run')
    if manifest.get('status') != 'completed':
        return dict(gate='INCOMPLETE', status=manifest.get('status'), long_training_allowed=False)
    decisions = manifest.get('total_learner_decisions', 0)
    baseline = run/'evaluations'/'baseline'
    candidates = [run/'evaluations'/f'final-d{decisions:012d}',
                  run/'evaluations'/f'd{decisions:012d}']
    candidate = next((p for p in candidates if (p/'summary.json').is_file()), None)
    if not decisions or not manifest.get('accepted_updates') or candidate is None:
        return dict(gate='INCOMPLETE', reason='no complete accepted-update evaluation',
                    long_training_allowed=False)
    result = paired_comparison(_read_rows(baseline), _read_rows(candidate))
    return dict(gate=development_gate(result), comparison=result,
                evaluation=str(candidate), baseline=str(baseline),
                protocol='repeated_development_set_not_confirmatory_test',
                long_training_allowed=False)
