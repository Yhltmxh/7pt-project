"""Augmentation semantics and reproducibility, using real image operations."""
import random

import numpy as np
from PIL import Image
import pytest
import torch

from src.config import ConfigError, apply_overrides, load_config, validate_config
from src.data.transforms import build_transforms


def image(height=32, width=40):
    return Image.fromarray(np.random.default_rng(17).integers(0, 256, (height, width, 3), dtype=np.uint8))


def config():
    return apply_overrides(load_config('configs/base.yaml'), {'preprocess.image_size': 32})['preprocess']


def test_same_image_receives_same_geometry_and_appearance_in_both_modalities():
    transform = build_transforms(config(), training=True)
    sample = image()
    for _ in range(20):
        result = transform({'clinical': sample, 'dermoscopic': sample})
        assert torch.equal(result['clinical'], result['dermoscopic'])
        assert result['clinical'].shape == (3, 32, 32)
        assert result['clinical'].dtype == torch.float32 and torch.isfinite(result['clinical']).all()


def test_restore_python_rng_replays_full_albumentations_sequence():
    transform = build_transforms(config(), training=True)
    sample = image()
    state = random.getstate()
    sequence = [transform({'clinical': sample})['clinical'] for _ in range(5)]
    random.setstate(state)
    replay = build_transforms(config(), training=True)
    assert all(torch.equal(expected, replay({'clinical': sample})['clinical']) for expected in sequence)
    assert any(not torch.equal(sequence[0], item) for item in sequence[1:])


def test_different_modality_sizes_and_single_modalities_are_supported():
    transform = build_transforms(config(), training=True)
    for images in ({'clinical': image(28, 37), 'dermoscopic': image(40, 30)},
                   {'clinical': image()}, {'dermoscopic': image()}):
        result = transform(images)
        assert set(result) == set(images)
        assert all(value.shape == (3, 32, 32) for value in result.values())
    with pytest.raises(ValueError, match='inputs'):
        transform({'diagnosis': image()})


def test_validation_is_deterministic_and_does_not_consume_global_rng():
    transform = build_transforms(config(), training=False)
    sample = image()
    random_state, numpy_state = random.getstate(), np.random.get_state()
    outputs = [transform({'clinical': sample})['clinical'] for _ in range(3)]
    assert all(torch.equal(outputs[0], value) for value in outputs[1:])
    assert random.getstate() == random_state
    current = np.random.get_state()
    assert current[0] == numpy_state[0] and np.array_equal(current[1], numpy_state[1])
    assert current[2:] == numpy_state[2:]
    assert [type(t).__name__ for t in transform.pipeline.transforms] == ['Resize', 'Normalize', 'ToTensorV2']


@pytest.mark.parametrize('field,value', [
    ('affine_probability', 1.1), ('brightness_limit', -1), ('scale', [1.1, .9]),
    ('crop_scale', [.8, 1.2]), ('blur_limit', 4), ('noise_std_range', [0, 2]),
    ('rotate', float('nan')),
])
def test_invalid_augmentation_parameters_are_rejected(field, value):
    with pytest.raises(ConfigError):
        load_config('configs/base.yaml', {f'preprocess.isic2025.{field}': value})


def test_policy_and_loader_contracts_reject_incompatible_settings():
    for overrides in ({'preprocess.geometry_coupling': 'independent'},
                      {'preprocess.appearance_coupling': 'none'},
                      {'train.persistent_workers': True}, {'train.prefetch_factor': 0},
                      {'preprocess.isic2025.typo': .5}):
        with pytest.raises(ConfigError):
            load_config('configs/base.yaml', overrides)
    cfg = load_config('configs/base.yaml', {'preprocess.train_augmentation': 'flips',
                                          'preprocess.appearance_coupling': 'none'})
    validate_config(cfg)
    assert build_transforms(cfg['preprocess'], True).appearance_coupling == 'none'
