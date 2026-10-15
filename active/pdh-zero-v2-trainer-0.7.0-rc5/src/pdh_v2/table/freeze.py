"""PDH research code: freeze all predeclared seeds, never cherry-pick a smoke winner."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid

from pdh_v2.table.data import contract
from pdh_v2.table.study_contract import check_identity
from pdh_v2.training.candidate_store import file_digest
from pdh_v2.training.refinement_control import atomic_json, utc


def confined_file(root: Path, relative: str) -> Path:
    """Reject traversal and external symlinks in machine-readable candidate pointers."""
    path = (root / relative).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise ValueError('rc5.candidate_path_outside_evidence')
    return path


def freeze_study(root: Path, study: Path, output: Path, hands: int = 4096,
                 max_seconds: int = 21600) -> dict:
    """Freeze only a completed non-smoke study with every seed and opponent gate passed."""
    manifest = json.loads((study/'manifest.json').read_text())
    report = json.loads((study/'report.json').read_text())
    config = manifest['config']
    if (report['status'] != 'PASS' or config['smoke_only'] or len(config['seeds']) < 3
            or report['completed_arms'] != 3*len(config['seeds'])
            or hands < 4096 or hands % 4 or max_seconds < 1
            or manifest['baseline_sha256'] != contract()['baseline']['sha256']):
        raise ValueError('rc5.study_not_eligible_for_final_freeze')
    check_identity(root, manifest['protected_source'])
    selected = []
    for seed in config['seeds']:
        trial = json.loads((study/f'seed-{seed}/joint/trial.json').read_text())
        if trial['decision']['status'] != 'keep_for_independent_confirmation':
            raise ValueError('rc5.not_all_seed_gates_passed')
        for mode in ('baseline', 'joint'):
            item = json.loads((study/f'seed-{seed}/{mode}/trial.json').read_text())
            path = confined_file(study, item['checkpoint'])
            if file_digest(path) != item['checkpoint_sha256']:
                raise ValueError('rc5.candidate_digest_mismatch')
            selected.append((seed, mode, path, item['checkpoint_sha256']))
    output.mkdir(parents=True, exist_ok=False)
    candidates = []
    for seed, mode, path, digest in selected:
        name = f'{seed}-{mode}.pt'
        shutil.copyfile(path, output/name)
        (output/(name+'.sha256')).write_text(digest+'  '+name+'\n')
        candidates.append(dict(seed=seed, mode=mode, file=name, sha256=digest))
    frozen = dict(schema='rc5-final-freeze-v1', freeze_id=uuid.uuid4().hex, created_at=utc(),
                  study_manifest_sha256=file_digest(study/'manifest.json'),
                  protected_source=manifest['protected_source'], candidates=candidates,
                  baseline_sha256=manifest['baseline_sha256'], seeds=config['seeds'],
                  opponents=config['opponents'], hands_per_condition=hands,
                  max_seconds=max_seconds, historical_coverage='partial',
                  protocol='all_seeds_all_opponents_paired_fresh_walls_no_training',
                  final_seeds_generated_after_freeze=True, deployment_promoted=False)
    atomic_json(output/'freeze.json', frozen)
    (output/'freeze.sha256').write_text(file_digest(output/'freeze.json')+'  freeze.json\n')
    return frozen


def load_freeze(root: Path, path: Path) -> dict:
    """Validate frozen identities before generating or revealing any new test seed."""
    if file_digest(path/'freeze.json') != (path/'freeze.sha256').read_text().split()[0]:
        raise ValueError('rc5.freeze_digest_mismatch')
    frozen = json.loads((path/'freeze.json').read_text())
    if (frozen['schema'] != 'rc5-final-freeze-v1'
            or frozen['baseline_sha256'] != contract()['baseline']['sha256']):
        raise ValueError('rc5.freeze_identity_mismatch')
    check_identity(root, frozen['protected_source'])
    for item in frozen['candidates']:
        if file_digest(confined_file(path, item['file'])) != item['sha256']:
            raise ValueError('rc5.candidate_digest_mismatch')
    return frozen