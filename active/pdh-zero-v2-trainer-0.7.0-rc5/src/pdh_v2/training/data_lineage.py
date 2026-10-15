"""PDH研究代码：只读导入可用祖先牌谱；缺失历史明确标记，不能追认完全隔离。"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Callable

from pdh_v2.training.candidate_store import file_digest
from pdh_v2.training.wall_registry import WallRegistry, exposed_walls


def _run_for_source(source: Path) -> Path | None:
    if source.name == 'selected_development.json':
        return source.parent
    for parent in source.parents:
        if parent.name == 'checkpoints':
            return parent.parent
    return None


def import_lineage(registry: WallRegistry, source: Path,
                   check: Callable[[], None] | None = None) -> dict:
    """导入源run及可达祖先的训练/评估牌墙指纹；不读取或修改模型权重。"""
    root = _run_for_source(source.resolve())
    pending = [root] if root is not None else []
    visited: set[Path] = set()
    imported: list[dict] = []
    missing: list[str] = [] if root is not None else [str(source)]
    while pending:
        run = pending.pop().resolve()
        if run in visited:
            continue
        if len(visited) >= 100:
            raise ValueError('data_protocol.lineage_exceeds_100_runs')
        visited.add(run)
        if check is not None:
            check()
        manifest_path = run / 'manifest.json'
        if not manifest_path.is_file():
            missing.append(str(manifest_path))
            continue
        manifest = json.loads(manifest_path.read_text())
        paths = sorted(set((run / 'replays').glob('*.jsonl.gz'))
                       | set((run / 'evaluations').glob('*/hands.jsonl.gz')))
        if not paths:
            missing.append(str(run / '{replays,evaluations}'))
        for path in paths:
            if check is not None:
                check()
            digest = file_digest(path)
            if registry.history_loaded(digest):
                imported.append({'path': str(path), 'sha256': digest, 'cached': True})
                continue
            count = 0
            try:
                with gzip.open(path, 'rt', encoding='utf-8') as handle:
                    for line in handle:
                        row = json.loads(line)
                        for _, wall in exposed_walls(row['transcript'], row.get('audit')):
                            registry.import_history_wall(wall, str(run))
                        count += 1
                        if check is not None and count % 256 == 0:
                            check()
                if count == 0:
                    raise ValueError('data_protocol.empty_history_replay')
                registry.finish_history(digest, path, count)
            except BaseException:
                registry.connection.rollback()
                raise
            imported.append({'path': str(path), 'sha256': digest, 'records': count})
        config = manifest.get('config', {})
        # NOTE: rc1教师轨迹只存内存；不能因为PPO牌谱存在，就宣称所有预训练暴露已登记。
        if config.get('warmup_hands', 0):
            missing.append(str(run / 'UNRECORDED_LEGACY_TEACHER_WALLS'))
        if any((run / 'evaluations').glob('evaluation-*.json')):
            missing.append(str(run / 'UNRECORDED_LEGACY_EVALUATION_WALLS'))
        ancestor = config.get('initial_selection') or config.get('warmup_checkpoint')
        if ancestor:
            parent = _run_for_source(Path(ancestor))
            if parent is not None:
                pending.append(parent)
            else:
                missing.append(str(ancestor))
    return {'files': imported, 'missing_history': sorted(set(missing)),
            'historical_coverage': 'partial' if missing else 'accessible_lineage',
            'claim': 'known_logged_wall_exclusion_not_proof_of_all_past_exposure'}
