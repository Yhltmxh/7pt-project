"""Development CB Focal boundaries, fold-local weights and exact recovery."""
import csv
import json
from collections import Counter

import numpy as np
import pytest
import torch

from scripts import train_server
from scripts.run_experiments import build_plan, expand_suite
from scripts.smoke_test import create_fixture
from src.config import ConfigError, apply_overrides, load_config, validate_config
from src.data.manifest import prepare_dataset, read_manifest
from src.data.splits import make_dev_folds
from src.provenance import file_hash
from src.train import make_loader, run_training
from src.train_utils import load_checkpoint, prediction_cross_entropy
from tests.test_cv_splits import write_folds


def test_cb_profile_schedules_only_seed42_b04_five_folds(tmp_path):
    args = train_server.parser().parse_args([
        '--profile', 'b04_cb_focal', '--data-root', 'data/derm7pt',
        '--job-dir', str(tmp_path / 'job')])
    directory, job = train_server._resolve_job(args)
    plan = build_plan(expand_suite(directory / 'suite.yaml',
        only=job['settings']['only'], overrides=job['overrides']))
    assert {(r['spec_id'], r['seed'], r['fold']) for r in plan['runs']} == {
        ('B04', 42, fold) for fold in range(5)}
    for cfg in plan['resolved_configs'].values():
        assert cfg['loss']['diagnosis'] == 'cb_focal'
        assert cfg['loss']['cb_beta'] == .99 and cfg['loss']['focal_gamma'] == 2
        assert cfg['model']['head_dropout'] == .2 and cfg['train']['mixup_alpha'] == 0
        assert cfg['train']['batch_size'] == 16 and cfg['train']['gradient_accumulation'] == 1
        assert cfg['train']['ema']['decay'] == .99 and cfg['train']['epochs'] == 25
        assert cfg['evaluation']['track_regular'] and cfg['evaluation']['track_train']
    for flags in (['--only', 'B02'], ['--seeds', '101'], ['--split-seed', '42'],
                  ['--protocol', 'official_holdout']):
        invalid = train_server.parser().parse_args([
            '--profile', 'b04_cb_focal', '--data-root', 'data/derm7pt',
            '--job-dir', str(tmp_path / 'invalid'), *flags])
        with pytest.raises(ConfigError, match='requires only B04'):
            train_server._resolve_job(invalid)
        assert not (tmp_path / 'invalid').exists()
    assert load_config('configs/base.yaml')['loss']['diagnosis'] == 'cross_entropy'
    assert load_config('configs/development/b04_regularized.yaml')['train']['mixup_alpha'] == .2


@pytest.mark.parametrize('key,value', [
    ('loss.cb_beta', None), ('loss.cb_beta', 1), ('loss.cb_beta', -.1),
    ('loss.cb_beta', True), ('loss.cb_beta', float('nan')),
    ('loss.focal_gamma', None), ('loss.focal_gamma', -.1),
    ('loss.focal_gamma', float('inf')), ('loss.focal_gamma', True),
    ('loss.class_weights_source', 'all_cases'), ('stage', 'draft'), ('stage', 'frozen'),
])
def test_cb_candidate_rejects_invalid_or_unregistered_settings(key, value):
    cfg = apply_overrides(load_config('configs/development/b04_cb_focal.yaml'), {key: value})
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_ce_still_refuses_focal_parameters():
    with pytest.raises(ConfigError, match='cross entropy baseline'):
        load_config('configs/base.yaml', {'loss.cb_beta': .99})


def test_cb_uses_training_fold_weights_without_mixup_and_resumes_exactly(tmp_path, monkeypatch):
    import src.losses as losses
    import src.train_utils as train_utils
    torch.set_num_threads(2)
    root = tmp_path / 'synthetic'
    manifest = create_fixture(root, n_cases=36, train_count=21, valid_count=10)
    # A genuinely imbalanced synthetic development pool; test cases stay separate.
    meta_file = root / 'meta/meta.csv'
    with meta_file.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for index in (0, 5):
        rows[index]['diagnosis'] = 'clark nevus'
    with meta_file.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert prepare_dataset(root, manifest, require_official_counts=False)['passed']
    data_rows = read_manifest(manifest / 'samples.csv')
    folds = write_folds(tmp_path / 'folds.csv', make_dev_folds(data_rows, 5, 20261004))
    cfg = load_config('configs/development/b04_cb_focal.yaml', {
        'dataset.root': str(root), 'dataset.manifest': str(manifest / 'samples.csv'),
        'dataset.label_map': str(manifest / 'label_map.json'),
        'dataset.clinical_substitution_policy': 'retain_official',
        'dataset.split_protocol': 'dev_cv5', 'dataset.folds_file': str(folds), 'dataset.cv_fold': 2,
        'model.clinical_encoder': 'tiny_cnn', 'model.dermoscopic_encoder': 'tiny_cnn',
        'model.pretrained_id': None, 'preprocess.image_size': 32,
        'train.epochs': 2, 'train.warmup_epochs': 0, 'train.batch_size': 3,
        'train.num_workers': 0, 'train.device': 'cpu',
        'experiment.output_root': str(tmp_path / 'runs')})
    training = make_loader(cfg, 'train').dataset.rows
    count = Counter(r['diagnosis_index'] for r in training)
    expected_counts = [count[c] for c in range(5)]
    assert len(set(expected_counts)) > 1
    assert all(r['official_split'] != 'test' for r in training)
    original_build = losses.build_loss
    observed = []

    def record_loss(config, class_counts=None):
        assert class_counts == expected_counts
        criterion = original_build(config, class_counts)
        assert isinstance(criterion.diagnosis_loss, losses.CBFocalLoss)
        assert criterion.diagnosis_loss.alpha.mean().item() == pytest.approx(1)
        observed.append(criterion.diagnosis_loss.alpha.detach().clone())
        return criterion

    def forbid_mixup(inputs, targets, alpha):
        assert alpha == 0
        return inputs, targets, targets, 1.

    monkeypatch.setattr(losses, 'build_loss', record_loss)
    monkeypatch.setattr(train_utils, 'mixup_inputs', forbid_mixup)
    full = run_training(cfg, 42, output_dir=tmp_path / 'full')
    resumed = run_training(cfg, 42, output_dir=tmp_path / 'resumed', stop_after_epoch=1)
    run_training(cfg, 42, output_dir=resumed, resume=resumed / 'checkpoints/final_regular.pt')
    assert len(observed) == 3
    a = load_checkpoint(full / 'checkpoints/final_regular.pt')
    b = load_checkpoint(resumed / 'checkpoints/final_regular.pt')
    for group in ('model', 'ema'):
        assert all(torch.equal(a[group][key], b[group][key]) for key in a[group])
    assert a['history'] == b['history']
    assert all(r['train_accuracy_definition'] == 'augmented_hard_label_accuracy' for r in a['history'])
    with np.load(full / 'predictions/valid_ema_argmax_notta.npz', allow_pickle=False) as prediction:
        assert a['history'][-1]['valid_loss'] == pytest.approx(prediction_cross_entropy(prediction))
    provenance = json.loads((full / 'provenance.json').read_text())
    assert provenance['class_counts'] == expected_counts and not provenance['test_used']
    assert provenance['diagnosis_loss'] == 'cb_focal'
    assert provenance['diagnosis_class_weights'] == observed[0].tolist()
    assert provenance['diagnosis_class_weight_normalization'] == 'mean_one'
    before = file_hash(full / 'status.json')
    changed = apply_overrides(cfg, {'loss.diagnosis': 'cross_entropy',
                                    'loss.cb_beta': None, 'loss.focal_gamma': None})
    with pytest.raises(ValueError, match='identity'):
        run_training(changed, 42, output_dir=full, resume=full / 'checkpoints/final_regular.pt')
    assert file_hash(full / 'status.json') == before
