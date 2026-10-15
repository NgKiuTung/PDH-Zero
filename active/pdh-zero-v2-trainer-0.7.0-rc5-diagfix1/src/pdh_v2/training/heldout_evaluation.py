"""冻结候选的新牌墙验证；不做优化器更新，不按盲测结果再次挑候选。"""
from __future__ import annotations

from dataclasses import asdict
import fcntl
import json
import math
from pathlib import Path
import subprocess

import torch

from pdh_v2.config import SOURCE_SHA256
from pdh_v2.policy import CandidateActorCritic, ModelConfig
from pdh_v2.training.candidate_store import file_digest
from pdh_v2.training.paired_evaluation import paired_evaluate
from pdh_v2.training.paired_metrics import paired_comparison, development_gate
from pdh_v2.training.refinement_artifacts import commit_best
from pdh_v2.training.refinement_control import Progress, StopRequested, atomic_json
from pdh_v2.training.refinement_gate import read_evaluation
from pdh_v2.training.random_streams import new_seed
from pdh_v2.training.data_lineage import import_lineage
from pdh_v2.training.wall_registry import WallRegistry, exposed_walls


def load_evaluation_model(path: Path, device: torch.device):
    """加载明确指定的冻结模型；校验哈希、房规、结构和有限性，不加载优化器。"""
    if not path.is_file() or path.stat().st_size > 1024**3:
        raise ValueError('missing or oversized checkpoint')
    expected = path.with_suffix(path.suffix + '.sha256').read_text().split()[0]
    if file_digest(path) != expected:
        raise ValueError('checkpoint hash mismatch')
    data = torch.load(path, map_location='cpu', weights_only=True)
    if data.get('schema_version') != 1 or data.get('rule_source_sha256') != SOURCE_SHA256:
        raise ValueError('checkpoint schema/rule identity mismatch')
    config = ModelConfig(**data['model_config'])
    if config.risk_policy_weight != 0:
        raise ValueError('evaluation requires already declared diagnostic-only risk policy')
    model = CandidateActorCritic(config).to(device)
    model.load_state_dict(data['model'], strict=True)
    if any(not torch.isfinite(t).all() for t in model.parameters()):
        raise ValueError('non-finite checkpoint')
    return model.eval().requires_grad_(False)


def validate_new_walls(run: Path, seed: int, hands: int) -> set[int]:
    """与实际开发集逐局key比较，不只比较首个seed；四座位共享一个牌墙组。"""
    if type(seed) is not int or seed < 0 or hands < 4 or hands % 4:
        raise ValueError('invalid heldout seed/hands')
    proposed = {seed + i * 1_000_003 for i in range(hands // 4)}
    summaries = list((run / 'evaluations').glob('*/summary.json'))
    if not summaries:
        raise ValueError('source run has no completed evaluation to check overlap')
    used = set()
    for summary in summaries:
        used.update(row['wall_seed'] for row in read_evaluation(summary.parent))
    if proposed & used:
        raise ValueError('heldout wall seeds overlap development walls')
    return proposed


def resolve_candidate(run: Path, candidate: Path | None) -> Path:
    """默认只取已冻结的开发候选；旧rc2没有此指针，必须显式指定其真实文件。"""
    if candidate is None:
        selection = json.loads((run / 'selected_development.json').read_text())
        if selection.get('label') == 'baseline':
            raise ValueError('no non-baseline candidate selected; inspect development evidence')
        candidate = run / selection['checkpoint']
        if file_digest(candidate) != selection['checkpoint_sha256']:
            raise ValueError('selection hash mismatch')
    resolved = candidate.resolve()
    if (run / 'checkpoints').resolve() not in resolved.parents:
        raise ValueError('candidate must be inside source run/checkpoints')
    return resolved


def reserve_walls(root: Path, seed: int, hands: int, output: Path) -> None:
    """同一个独立验证根目录内，使用过或中断暴露过的牌墙不能再次称为新牌墙。"""
    path = root / 'wall_reservations.json'
    records = json.loads(path.read_text()) if path.exists() else []
    proposed = {seed + i * 1_000_003 for i in range(hands // 4)}
    for record in records:
        used = {record['seed'] + i * 1_000_003 for i in range(record['hands'] // 4)}
        if proposed & used:
            raise ValueError('heldout walls already reserved; do not repeatedly tune on them')
    records.append(dict(seed=seed, hands=hands, output=str(output)))
    atomic_json(path, records)


def _device(name):
    device = torch.device(name)
    if device.type == 'cuda':
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid',
                                 '--format=csv,noheader,nounits'],
                                check=True, capture_output=True, text=True, timeout=10)
        if result.stdout.strip() or not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable or occupied; no process was killed')
    return device


def evaluate_frozen_pair(run: Path, output: Path, candidate: Path | None = None,
                         hands: int = 4096, seed: int | None = None, workers: int = 4,
                         device_name: str = 'cuda', max_seconds: int = 5400,
                         max_cvar_drop: float = 2.0, install_signals: bool = True) -> dict:
    """一次比较baseline与事先指定的候选；受总预算约束，不修改源实验。"""
    if type(workers) is not int or workers < 0 or max_seconds <= 0:
        raise ValueError('nonnegative workers and positive wall budget required')
    if not math.isfinite(max_cvar_drop) or max_cvar_drop < 0:
        raise ValueError('invalid tail-risk budget')
    run = run.resolve()
    seed = new_seed() if seed is None else seed
    source_manifest = json.loads((run / 'manifest.json').read_text())
    if source_manifest.get('status') != 'completed':
        raise ValueError('freeze and complete the source run before heldout evaluation')
    validate_new_walls(run, seed, hands)
    candidate = resolve_candidate(run, candidate)
    baseline = run / 'checkpoints' / 'checkpoint_initial.pt'
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / '.heldout.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        device = _device(device_name)
        output.mkdir(exist_ok=False)
        reserve_walls(output.parent, seed, hands, output)
        clock = Progress(output, max_seconds)
        if install_signals:
            clock.install_signals()
        report = dict(status='running', scope='frozen_pair_new_wall_single_training_seed',
                      hands=hands, seed=seed, source_run=str(run),
                      long_training_allowed=False, max_cvar_drop=max_cvar_drop)
        atomic_json(output / 'manifest.json', report)
        registry = WallRegistry(run.parent / '.wall-exposures.sqlite3')
        try:
            clock.stage('audit_data_lineage')
            report['data_lineage'] = import_lineage(
                registry, run / 'selected_development.json', clock.check)
            def _record(terminal):
                audit = terminal.audit or {}
                if not audit.get('outcome') or 'session_seed' not in audit:
                    raise ValueError('data_protocol.incomplete_heldout_hand')
                transcript = {'hand_id': terminal.hand_id, 'wall': terminal.transcript.wall}
                for hand_id, wall in exposed_walls(transcript, audit):
                    origin = f"{audit['session_seed']}:{hand_id}"
                    registry.register(wall, 'heldout', str(output.resolve()), origin)
            rows = {}
            configs = []
            torch.set_num_threads(1)
            for label, path in (('baseline', baseline), ('candidate', candidate)):
                clock.stage('load_checkpoint', candidate=label)
                model = load_evaluation_model(path, device)
                configs.append(asdict(model.config))
                if configs[0] != configs[-1]:
                    raise ValueError('effective model configuration differs')
                report[label + '_sha256'] = file_digest(path)
                commit_best(path, output / (label + '.pt'))
                atomic_json(output / 'manifest.json', report)
                clock.stage('evaluation', evaluation=label, total_hands=hands)
                rows[label], report[label] = paired_evaluate(model, device, output / label,
                    hands=hands, environments=min(64, hands), workers=workers, seed=seed,
                    progress=clock.tick, check_stop=clock.check, transport='shared',
                    on_terminal=_record)
                clock.event('evaluation_completed', evaluation=label, **report[label])
                del model
            comparison = paired_comparison(rows['baseline'], rows['candidate'])
            comparison['scope'] = 'fresh_walls_frozen_pair_not_multiple_training_seeds'
            report.update(status='completed', comparison=comparison,
                          gate=development_gate(comparison, max_cvar_drop=max_cvar_drop))
        except (StopRequested, KeyboardInterrupt) as exc:
            report.update(status='stopped', gate='INCOMPLETE', reason=str(exc))
        except BaseException as exc:
            report.update(status='failed', gate='INCOMPLETE', error=str(exc))
            raise
        finally:
            report['known_wall_identity_check'] = (
                'PASS' if report['status'] == 'completed' else 'INCOMPLETE')
            report['registered_unique_exposed_walls'] = registry.counts(str(output.resolve()))
            registry.__exit__(None, None, None)
            report['elapsed_wall_seconds'] = clock.elapsed
            atomic_json(output / 'manifest.json', report)
            atomic_json(output / 'report.json', report)
            atomic_json(output / 'status.json', report)
            clock.restore_signals()
        return report