"""PDH研究代码：每次新运行使用新随机流，固定开发配对与独立验证严格分层。"""
from __future__ import annotations

from pathlib import Path

from pdh_v2.training.random_streams import derive_seed
from pdh_v2.training.data_lineage import import_lineage
from pdh_v2.training.refinement_control import atomic_json
from pdh_v2.training.wall_registry import WallRegistry, exposed_walls


class DataProtocol:
    """训练完成局在进入PPO前查重；固定开发集复用明确登记，不计作新增独立牌墙。"""

    def __init__(self, config, clock):
        self.config = config
        self.prefix = str(config.output.resolve())
        self.registry = WallRegistry(config.output.parent / '.wall-exposures.sqlite3')
        self.completed_hands = 0
        self.completed_actions = 0
        self.report = {
            'schema_version': 1, 'training_unit': 'complete_hand_seat_sequence',
            'recurrent_memory': 'within_hand_reset_at_terminal',
            'cross_hand_state': 'dealer_continuation_scores_and_retained_reveals',
            'cross_hand_return_optimization': False,
            'run_seed': config.seed, 'seed_origin': config.seed_origin,
            'development_seed': config.evaluation_seed,
            'development_policy': 'fixed_within_run_repeated_not_confirmatory',
            'evaluation_hands': config.evaluation_hands,
            'unique_development_starting_walls': config.evaluation_hands // 4,
            'seat_variants_per_wall': 4,
            'fresh_frozen_evaluation_required': True,
            'generalization_verified': False,
            'registry': str(self.registry.path),
        }
        try:
            clock.stage('audit_data_lineage')
            source = config.initial_selection or config.warmup_checkpoint
            self.report['lineage'] = import_lineage(self.registry, source, clock.check)
            self.write()
        except BaseException:
            self.registry.__exit__(None, None, None)
            raise

    def close(self) -> None:
        """释放本任务登记连接；不是删除证据。"""
        self.registry.__exit__(None, None, None)

    def training_seeds(self) -> tuple[int, ...]:
        """桌号独立于Worker划分；同会话牌墙RNG继续前进，不每个更新重新洗同一局。"""
        return tuple(derive_seed(self.config.seed, 'train-session', i)
                     for i in range(self.config.environments))

    def record(self, terminal, purpose: str) -> None:
        """终局才登记完整轨迹；预发下一手与随后实际打该手按同一origin识别。"""
        audit = terminal.audit or {}
        if not audit.get('outcome') or 'session_seed' not in audit:
            raise ValueError('data_protocol.requires_complete_terminal_and_session_seed')
        session = audit['session_seed']
        transcript = {'hand_id': terminal.transcript.hand_id,
                      'wall': terminal.transcript.wall}
        owner = f'{self.prefix}:{purpose}'
        for hand_id, wall in exposed_walls(transcript, audit):
            self.registry.register(wall, purpose, owner, f'{session}:{hand_id}')
        if purpose == 'train':
            self.completed_hands += 1
            self.completed_actions += len(terminal.transcript.actions)

    def write(self) -> None:
        """由运行边界落盘，逐手查重不产生逐动作日志或同步写大量统计文件。"""
        self.report['registered_unique_exposed_walls'] = self.registry.counts(self.prefix + ':')
        self.report['completed_training_hands'] = self.completed_hands
        self.report['completed_training_action_events'] = self.completed_actions
        atomic_json(self.config.output / 'data-integrity.json', self.report)
