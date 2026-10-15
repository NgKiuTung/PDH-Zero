"""随机流、跨用途指纹隔离、历史范围和显式复现回归。"""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pdh_v2.game.session import GameSession
from pdh_v2.training.data_lineage import import_lineage
from pdh_v2.training.random_streams import derive_seed
from pdh_v2.training.refinement_control import RefinementConfig
from pdh_v2.training.wall_registry import WallRegistry, exposed_walls, wall_fingerprint


class RandomStreamTests(unittest.TestCase):
    def test_new_runs_do_not_use_fixed_date_seed(self):
        with patch('pdh_v2.training.random_streams.new_seed', side_effect=[431, 902]):
            first = RefinementConfig(Path('one'), Path('input.pt'))
            second = RefinementConfig(Path('two'), Path('input.pt'))
        self.assertNotEqual(first.seed, second.seed)
        self.assertNotEqual(first.evaluation_seed, second.evaluation_seed)
        self.assertEqual(first.seed_origin, 'generated')

    def test_explicit_seed_reproduces_stream_identity(self):
        first = RefinementConfig(Path('one'), Path('input.pt'), seed=314)
        second = RefinementConfig(Path('two'), Path('input.pt'), seed=314)
        self.assertEqual(first.seed, second.seed)
        self.assertEqual(first.evaluation_seed, second.evaluation_seed)
        self.assertEqual(first.seed_origin, 'explicit')

    def test_purposes_have_distinct_seed_streams(self):
        values = {derive_seed(21, role) for role in ('train', 'development', 'heldout')}
        self.assertEqual(len(values), 3)

    def test_sessions_have_distinct_seeds(self):
        self.assertEqual(len({derive_seed(9, 'train-session', i) for i in range(64)}), 64)

    def test_negative_seed_is_rejected(self):
        with self.assertRaises(ValueError):
            RefinementConfig(Path('one'), Path('input.pt'), seed=-1)

    def test_bool_seed_is_rejected(self):
        with self.assertRaises(ValueError):
            derive_seed(True, 'train')

    def test_singleton_eval_is_not_4096_cut_questions(self):
        with self.assertRaises(ValueError):
            RefinementConfig(Path('one'), Path('input.pt'), evaluation_hands=1)

    def test_changing_run_seed_changes_starting_wall(self):
        a = GameSession(derive_seed(10, 'train-session')).current_hand.wall.order
        b = GameSession(derive_seed(11, 'train-session')).current_hand.wall.order
        self.assertNotEqual(wall_fingerprint(a), wall_fingerprint(b))


class WallRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.registry = WallRegistry(self.root / 'walls.sqlite3')
        self.wall = list(range(136))

    def tearDown(self):
        self.registry.__exit__(None, None, None)
        self.temp.cleanup()

    def test_swapping_identical_entity_copies_is_not_a_new_wall(self):
        other = self.wall.copy()
        other[0], other[1] = other[1], other[0]
        self.assertEqual(wall_fingerprint(self.wall), wall_fingerprint(other))

    def test_invalid_deck_rejected(self):
        with self.assertRaises(ValueError):
            wall_fingerprint([0] * 136)

    def test_fixed_development_pair_can_reuse_wall(self):
        first = self.registry.register(self.wall, 'development', 'run:dev', 'wall0:H0')
        second = self.registry.register(self.wall, 'development', 'run:dev', 'wall0:H0')
        self.assertEqual(first, second)
        self.assertEqual(self.registry.counts('run'), {'development': 1})

    def test_different_training_hand_duplicate_rejected(self):
        self.registry.register(self.wall, 'train', 'run:train', 'session0:H0')
        with self.assertRaisesRegex(ValueError, 'reused'):
            self.registry.register(self.wall, 'train', 'run:train', 'session0:H1')

    def test_train_to_development_overlap_rejected(self):
        self.registry.register(self.wall, 'train', 'run:train', 'session0:H0')
        with self.assertRaisesRegex(ValueError, 'reused'):
            self.registry.register(self.wall, 'development', 'run:dev', 'wall0:H0')

    def test_development_to_train_overlap_rejected(self):
        self.registry.register(self.wall, 'development', 'run:dev', 'wall0:H0')
        with self.assertRaisesRegex(ValueError, 'reused'):
            self.registry.register(self.wall, 'train', 'run:train', 'session0:H0')

    def test_heldout_cannot_reuse_training_wall(self):
        self.registry.register(self.wall, 'train', 'run:train', 'session0:H0')
        with self.assertRaisesRegex(ValueError, 'reused'):
            self.registry.register(self.wall, 'heldout', 'test:new', 'wall0:H0')

    def test_second_run_cannot_silently_reuse_explicit_seed_wall(self):
        self.registry.register(self.wall, 'train', 'run1:train', 'session0:H0')
        with self.assertRaisesRegex(ValueError, 'reused'):
            self.registry.register(self.wall, 'train', 'run2:train', 'session0:H0')

    def test_next_deal_then_played_same_hand_is_not_false_duplicate(self):
        self.registry.register(self.wall, 'train', 'run:train', 'session0:H1')
        self.registry.register(self.wall, 'train', 'run:train', 'session0:H1')
        self.assertEqual(self.registry.counts('run'), {'train': 1})

    def test_origin_must_not_change_wall(self):
        self.registry.register(self.wall, 'train', 'run:train', 'session0:H0')
        other = self.wall.copy()
        other[0], other[4] = other[4], other[0]
        with self.assertRaisesRegex(ValueError, 'origin_changed'):
            self.registry.register(other, 'train', 'run:train', 'session0:H0')

    def test_legacy_history_blocks_known_walls(self):
        self.registry.import_history_wall(self.wall, 'old-run')
        self.registry.connection.commit()
        with self.assertRaisesRegex(ValueError, 'reused'):
            self.registry.register(self.wall, 'train', 'new-run:train', 'session0:H0')

    def test_next_hand_and_horse_exposure_are_included(self):
        transcript = {'wall': self.wall, 'hand_id': 'H0'}
        audit = {'session_events': [
            {'event': 'HAND_DEALT', 'hand_id': 'H1', 'wall': self.wall},
            {'event': 'SEPARATE_HORSE_DECK', 'cycles': [self.wall]}]}
        self.assertEqual([key for key, _ in exposed_walls(transcript, audit)],
                         ['H0', 'H1', 'horse:H0:0'])

    def test_missing_history_is_reported_not_assumed_clean(self):
        result = import_lineage(self.registry, self.root / 'missing/selected_development.json')
        self.assertEqual(result['historical_coverage'], 'partial')
        self.assertTrue(result['missing_history'])

    def test_history_import_is_idempotent_and_content_bound(self):
        run = self.root / 'source'
        (run / 'replays').mkdir(parents=True)
        (run / 'manifest.json').write_text('{"config":{}}')
        with gzip.open(run / 'replays/hands.jsonl.gz', 'wt') as stream:
            stream.write(json.dumps({'transcript': {'hand_id': 'H0', 'wall': self.wall}}))
        first = import_lineage(self.registry, run / 'selected_development.json')
        second = import_lineage(self.registry, run / 'selected_development.json')
        self.assertEqual(first['files'][0]['records'], 1)
        self.assertTrue(second['files'][0]['cached'])

    def test_bad_gzip_is_not_marked_imported(self):
        run = self.root / 'bad'
        (run / 'replays').mkdir(parents=True)
        (run / 'manifest.json').write_text('{"config":{}}')
        (run / 'replays/hands.jsonl.gz').write_bytes(b'not gzip')
        with self.assertRaises(gzip.BadGzipFile):
            import_lineage(self.registry, run / 'selected_development.json')
        count = self.registry.connection.execute('SELECT COUNT(*) FROM history_files').fetchone()[0]
        self.assertEqual(count, 0)


class LegacyExposureTests(unittest.TestCase):
    def test_legacy_teacher_exposure_remains_unknown_even_with_run_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'run'
            run.mkdir()
            (run / 'manifest.json').write_text('{"config":{"warmup_hands":2048}}')
            with WallRegistry(root / 'walls.sqlite3') as registry:
                report = import_lineage(registry, run / 'selected_development.json')
            self.assertEqual(report['historical_coverage'], 'partial')
            self.assertTrue(any('TEACHER' in path for path in report['missing_history']))

    def test_summary_only_legacy_evaluation_is_not_full_wall_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'run'
            (run / 'evaluations').mkdir(parents=True)
            (run / 'manifest.json').write_text('{"config":{}}')
            (run / 'evaluations/evaluation-d0.json').write_text('{}')
            with WallRegistry(root / 'walls.sqlite3') as registry:
                report = import_lineage(registry, run / 'selected_development.json')
            self.assertTrue(any('EVALUATION' in path for path in report['missing_history']))
