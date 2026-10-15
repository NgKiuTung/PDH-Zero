"""PDH research code: fail-closed tensor boundaries before fitting or Actor deployment."""
from __future__ import annotations

import torch


def finite_state(state: dict) -> None:
    """Reject non-tensor or nonfinite state entries, even with a matching file digest."""
    if not state or any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all()
                        for v in state.values()):
        raise ValueError('rc5.nonfinite_or_invalid_checkpoint_tensor')


def validate_dataset_tensors(data: dict) -> None:
    """Only public observation/history features are accepted; labels are a separate namespace."""
    features, labels = data['features'], data['labels']
    if set(features) != {'observation', 'history'} or set(labels) != {'target', 'mask'}:
        raise ValueError('rc5.auxiliary_tensor_namespace_invalid')
    count = len(data['groups'])
    expected = ((features['observation'], (count, 622)),
                (features['history'], (count, 384)),
                (labels['target'], (count, 3, 153)), (labels['mask'], (count, 3, 153)),
                (data['validation'], (count,)), (data['closed'], (count, 3)))
    if count < 2 or any(tuple(t.shape) != shape for t, shape in expected):
        raise ValueError('rc5.auxiliary_tensor_shape_invalid')
    finite_state(dict(observation=features['observation'], history=features['history'], **labels))
    if data['validation'].dtype != torch.bool or data['closed'].dtype != torch.bool:
        raise ValueError('rc5.auxiliary_split_dtype_invalid')
    if not bool(((labels['mask'] == 0) | (labels['mask'] == 1)).all()):
        raise ValueError('rc5.auxiliary_mask_invalid')