"""Known-exposure and source-identity guards must fail closed."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from pdh_v2.table.exposure import ExposureStore


def replay(path):
    row = dict(transcript=dict(hand_id='H0', wall=list(range(136))), audit={}, receipt=[0]*4)
    with gzip.open(path, 'wt') as stream:
        stream.write(json.dumps(row)+'\n')


class Rc5ExposureTests(unittest.TestCase):
    def test_renaming_heldout_cannot_change_its_role(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'renamed-train.jsonl.gz'
            replay(source)
            with ExposureStore(root / 'registry.sqlite3') as store:
                store.import_file(source, 'heldout_used')

                with self.assertRaises(ValueError):
                    store.import_file(source, 'train')
                with self.assertRaises(ValueError):
                    store.require_training_source(source)

    def test_historical_wall_cannot_enter_fresh_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'history.jsonl.gz'
            replay(source)
            with ExposureStore(root / 'registry.sqlite3') as store:
                store.import_file(source, 'heldout_used')

                with self.assertRaises(ValueError):
                    store.register(list(range(136)), 'train', 'run', 'hand')
                with self.assertRaises(ValueError):
                    store.check_label_wall(list(range(136)))

    def test_partial_gzip_does_not_mark_file_imported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'broken.gz'
            source.write_bytes(b'not gzip')
            with ExposureStore(root / 'registry.sqlite3') as store:

                with self.assertRaises(OSError):
                    store.import_file(source, 'train')

                self.assertEqual(store.connection.execute(
                    'SELECT COUNT(*) FROM rc5_sources').fetchone()[0], 0)

    def test_explicit_paired_reuse_is_allowed_but_split_change_is_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            with ExposureStore(Path(tmp) / 'registry.sqlite3') as store:
                wall = list(range(136))
                first = store.register(wall, 'train', 'paired-study', 'seed:hand')

                second = store.register(wall, 'train', 'paired-study', 'seed:hand')

                self.assertEqual(first, second)
                with self.assertRaises(ValueError):
                    store.register(wall, 'development', 'paired-study', 'seed:hand')