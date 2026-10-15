"""Tensor-only boundaries reject poisoned features and checkpoint state before inference."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from pdh_v2.table.dataset import load_dataset
from pdh_v2.table.tensor_checks import finite_state, validate_dataset_tensors
from pdh_v2.table.data import contract
from pdh_v2.training.candidate_store import file_digest
from pdh_v2.training.refinement_control import atomic_json


def dataset_fixture():
    return dict(schema='rc5-auxiliary-v2', baseline=contract()['baseline']['sha256'],
                source_sha256=contract()['training_replay_sha256'], groups=['train', 'valid'],
                features=dict(observation=torch.zeros(2, 622), history=torch.zeros(2, 384)),
                labels=dict(target=torch.zeros(2, 3, 153), mask=torch.ones(2, 3, 153)),
                validation=torch.tensor([False, True]), closed=torch.ones(2, 3, dtype=torch.bool))


def write_fixture(root, data):
    path = root/'dataset.pt'
    torch.save(data, path)
    atomic_json(path.with_suffix('.json'), dict(dataset_sha256=file_digest(path)))
    return path


class Rc5TensorGuardTests(unittest.TestCase):
    def test_valid_namespaces_and_shapes(self):
        data = dataset_fixture()
        validate_dataset_tensors(data)
        self.assertNotIn('labels', data['features'])

    def test_private_feature_namespace_rejected(self):
        data = dataset_fixture()
        data['features']['future_wall'] = torch.zeros(136)
        with self.assertRaisesRegex(ValueError, 'namespace'):
            validate_dataset_tensors(data)

    def test_nonfinite_features_rejected(self):
        data = dataset_fixture()
        data['features']['history'][0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            validate_dataset_tensors(data)

    def test_shape_rejected(self):
        data = dataset_fixture()
        data['features']['history'] = torch.zeros(2, 1)
        with self.assertRaisesRegex(ValueError, 'shape'):
            validate_dataset_tensors(data)

    def test_mask_rejected(self):
        data = dataset_fixture()
        data['labels']['mask'][0, 0, 0] = 2
        with self.assertRaisesRegex(ValueError, 'mask'):
            validate_dataset_tensors(data)

    def test_float_validation_rejected(self):
        data = dataset_fixture()
        data['validation'] = data['validation'].float()
        with self.assertRaisesRegex(ValueError, 'dtype'):
            validate_dataset_tensors(data)

    def test_empty_state_rejected(self):
        with self.assertRaises(ValueError):
            finite_state({})

    def test_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_fixture(Path(tmp), dataset_fixture())
            data = load_dataset(path)
            self.assertEqual(data['groups'], ['train', 'valid'])

    def test_whole_session_overlap_rejected(self):
        data = dataset_fixture()
        data['groups'] = ['same_session', 'same_session']
        with tempfile.TemporaryDirectory() as tmp:
            path = write_fixture(Path(tmp), data)
            with self.assertRaisesRegex(ValueError, 'overlap'):
                load_dataset(path)

    def test_old_semantics_rejected(self):
        data = dataset_fixture()
        data['schema'] = 'rc5-auxiliary-v1'
        with tempfile.TemporaryDirectory() as tmp:
            path = write_fixture(Path(tmp), data)
            with self.assertRaisesRegex(ValueError, 'provenance'):
                load_dataset(path)

    def test_hash_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_fixture(Path(tmp), dataset_fixture())
            path.write_bytes(b'not the original file')
            with patch('torch.load') as loader:
                with self.assertRaisesRegex(ValueError, 'digest'):
                    load_dataset(path)
                loader.assert_not_called()