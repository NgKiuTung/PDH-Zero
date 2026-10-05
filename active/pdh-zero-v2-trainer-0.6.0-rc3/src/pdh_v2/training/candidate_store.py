"""评估候选不可覆盖存档；名义开发候选与通过统计门禁的权重分开。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil

from pdh_v2.training.refinement_artifacts import commit_best
from pdh_v2.training.refinement_control import atomic_json


def file_digest(path: Path) -> str:
    """流式校验权重，不为了计算哈希把大检查点再复制进内存。"""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_label(label: str) -> None:
    """日志标签只能定位本次候选子目录，拒绝路径和任意文件名。"""
    if not isinstance(label, str) or not re.fullmatch(
            r'baseline|(?:final-)?d[0-9]{12}', label):
        raise ValueError('invalid candidate label')


class CandidateStore:
    """单写入者目录；评估失败或INCONCLUSIVE也永久保留原始候选。"""

    def __init__(self, run: Path) -> None:
        self.run = run
        self.root = run / 'checkpoints' / 'candidates'
        self.root.mkdir(parents=True, exist_ok=True)

    def register(self, source: Path, label: str, update: int, decisions: int) -> Path:
        """在评估之前保存权重和身份；同名、路径穿越、无校验来源均拒绝。"""
        validate_label(label)
        expected = source.with_suffix(source.suffix + '.sha256').read_text().split()[0]
        actual = file_digest(source)
        if actual != expected:
            raise ValueError('candidate source hash mismatch')
        directory = self.root / label
        directory.mkdir(exist_ok=False)
        target = directory / 'model.pt'
        # NOTE: 先完整落盘临时文件再发布；后续latest覆盖不会改变这个实体。
        temporary = directory / 'model.tmp'
        with source.open('rb') as incoming, temporary.open('xb') as outgoing:
            shutil.copyfileobj(incoming, outgoing)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        os.replace(temporary, target)
        if file_digest(target) != actual:
            raise ValueError('candidate copy hash mismatch')
        target.with_suffix('.pt.sha256').write_text(actual + '  model.pt\n')
        atomic_json(directory / 'candidate.json', {
            'label': label, 'update': update, 'learner_decisions': decisions,
            'checkpoint_sha256': actual, 'status': 'awaiting_evaluation',
            'checkpoint': str(target.relative_to(self.run)),
        })
        return target

    def complete(self, label: str, summary: dict, comparison: dict, gate: str) -> None:
        """绑定同一候选的评估结果哈希，不覆盖已有完整评估。"""
        validate_label(label)
        if gate not in {'BASELINE', 'PASS', 'REGRESSION', 'INCONCLUSIVE'}:
            raise ValueError('unknown candidate evaluation gate')
        path = self.root / label / 'candidate.json'
        record = json.loads(path.read_text())
        if record['status'] != 'awaiting_evaluation':
            raise ValueError('candidate evaluation already committed')
        if file_digest(self.root / label / 'model.pt') != record['checkpoint_sha256']:
            raise ValueError('candidate changed during evaluation')
        record.update(status='evaluated', summary=summary, comparison=comparison, gate=gate)
        atomic_json(path, record)

    def select(self, label: str, qualified: bool = False) -> None:
        """保存指针及便利副本；开发集最佳不宣称为已验证最优或训练恢复点。"""
        validate_label(label)
        record = json.loads((self.root / label / 'candidate.json').read_text())
        if record['status'] != 'evaluated':
            raise ValueError('cannot select an unevaluated checkpoint')
        kind = 'qualified' if qualified else 'selected_development'
        if qualified and record['gate'] != 'PASS':
            raise ValueError('qualified selection requires PASS')
        expected_path = self.root / label / 'model.pt'
        if (self.run / record['checkpoint']).resolve() != expected_path.resolve():
            raise ValueError('candidate record points to a different checkpoint')
        if file_digest(expected_path) != record['checkpoint_sha256']:
            raise ValueError('candidate checkpoint hash mismatch')
        atomic_json(self.run / (kind + '.json'), {
            **record, 'selection_scope': kind, 'long_training_allowed': False,
        })
        commit_best(self.run / record['checkpoint'],
                    self.run / 'checkpoints' / ('checkpoint_' + kind + '.pt'))