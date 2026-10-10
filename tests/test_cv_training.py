"""Tiny synthetic checks for fold-aware training, resume and inference."""

import copy
import csv
import json
import zipfile

import numpy as np
import pytest
import torch

from scripts.smoke_test import create_fixture
from src.config import apply_overrides, load_config
from src.data.manifest import read_manifest
from src.data.splits import make_dev_folds
from src.infer import run_inference
from src.provenance import file_hash, make_identity, run_name
from src.train import make_loader, preflight, run_training
from src.train_utils import load_checkpoint
from tests.test_cv_splits import SPLIT_SEED, write_folds


@pytest.fixture
def cv_cfg(tmp_path):
    torch.set_num_threads(2)
    root = tmp_path / 'synthetic_data'
    manifest_dir = create_fixture(root, n_cases=30, train_count=15, valid_count=10)
    rows = read_manifest(manifest_dir / 'samples.csv')
    folds_file = write_folds(tmp_path / 'folds.csv', make_dev_folds(rows, 5, SPLIT_SEED))
    return apply_overrides(load_config('configs/base.yaml'), {
        'stage': 'development', 'experiment.id': 'B04', 'experiment.seeds': [42],
        'dataset.root': str(root), 'dataset.manifest': str(manifest_dir / 'samples.csv'),
        'dataset.label_map': str(manifest_dir / 'label_map.json'),
        'dataset.clinical_substitution_policy': 'retain_official',
        'dataset.split_protocol': 'dev_cv5', 'dataset.folds_file': str(folds_file),
        'dataset.cv_fold': 2, 'dataset.split_seed': SPLIT_SEED,
        'model.clinical_encoder': 'tiny_cnn', 'model.dermoscopic_encoder': 'tiny_cnn',
        'model.pretrained_id': None, 'preprocess.image_size': 32,
        'train.epochs': 2, 'train.warmup_epochs': 0, 'train.batch_size': 3,
        'train.gradient_accumulation': 3, 'train.num_workers': 0, 'train.device': 'cpu',
        'train.backbone_learning_rate': .001, 'train.head_learning_rate': .001,
        'train.ema.decay': .9, 'experiment.output_root': str(tmp_path / 'runs')})


def test_preflight_and_loaders_use_fold_training_counts_and_seal_test(cv_cfg):
    report = preflight(cv_cfg)
    assert report['counts'] == {'train': 15, 'valid': 10, 'test': 5}
    assert report['selection_counts'] == {'train': 20, 'valid': 5}
    assert report['training_diagnosis_counts'] == [4] * 5
    assert report['protocol'] == 'dev_cv5' and report['fold'] == 2
    assert report['eval_partition'] == 'dev_cv_valid' and len(report['folds_hash']) == 64
    train, valid = make_loader(cv_cfg, 'train'), make_loader(cv_cfg, 'valid')
    assert set(train.dataset.sample_ids).isdisjoint(valid.dataset.sample_ids)
    assert any(row['official_split'] == 'valid' for row in train.dataset.rows)
    assert all(row['official_split'] != 'test' for row in train.dataset.rows + valid.dataset.rows)
    with pytest.raises(ValueError, match='test remains sealed'):
        make_loader(cv_cfg, 'test')
    inputs, targets, ids = next(iter(train))
    assert set(inputs) == {'clinical', 'dermoscopic', 'modality_mask'}
    assert 'diagnosis' in targets and 'sample_id' in ids


@pytest.mark.parametrize('num_workers', [0, 2])
def test_fold_training_export_and_exact_resume(cv_cfg, tmp_path, num_workers):
    cv_cfg = apply_overrides(cv_cfg, {'train.num_workers': num_workers})
    loader = make_loader(cv_cfg, 'train')
    assert loader.num_workers == num_workers and not loader.persistent_workers
    assert loader.prefetch_factor == (2 if num_workers else None) and not loader.pin_memory
    full = run_training(cv_cfg, 42, output_dir=tmp_path / 'full')
    resumed = run_training(cv_cfg, 42, output_dir=tmp_path / 'resumed', stop_after_epoch=1)
    checkpoint_path = resumed / 'checkpoints/final_regular.pt'
    run_training(cv_cfg, 42, output_dir=resumed, resume=checkpoint_path)
    full_cp, resumed_cp = [load_checkpoint(path / 'checkpoints/final_regular.pt') for path in (full, resumed)]
    for field in ('model', 'ema'):
        assert full_cp[field].keys() == resumed_cp[field].keys()
        assert all(torch.equal(full_cp[field][key], resumed_cp[field][key]) for key in full_cp[field])
    assert full_cp['history'] == resumed_cp['history']
    assert full_cp['ema_updates'] == resumed_cp['ema_updates'] == 6
    assert full_cp['protocol'] == 'dev_cv5' and full_cp['fold'] == full_cp['cv_fold'] == 2
    provenance = json.loads((full / 'provenance.json').read_text())
    status = json.loads((full / 'status.json').read_text())
    assert provenance['class_counts'] == [4] * 5
    assert provenance['class_counts_source'] == 'current_training_partition_only'
    assert len(provenance['training_sample_ids']) == 20 and len(provenance['validation_sample_ids']) == 5
    assert not provenance['test_used']
    assert status['status'] == 'complete' and status['fold'] == status['cv_fold'] == 2
    prediction_path = full / 'predictions/valid_ema_argmax_notta.npz'
    with np.load(prediction_path, allow_pickle=False) as artifact:
        assert artifact['schema_version'].item() == 1
        assert artifact['fold'].item() == artifact['cv_fold'].item() == 2
        assert artifact['protocol'].item() == 'dev_cv5'
        assert artifact['eval_partition'].item() == 'dev_cv_valid'
        assert len(artifact['sample_id']) == 5
        for field in ('identity_hash', 'config_hash', 'source_hash', 'manifest_hash',
                      'label_map_hash', 'raw_data_hash', 'folds_hash', 'split_hash'):
            assert artifact[field].item() == provenance[field]
        before = artifact['logits'].copy()
    assert run_inference(cv_cfg, full / 'checkpoints/final_ema.pt') == prediction_path
    with np.load(prediction_path, allow_pickle=False) as artifact:
        assert np.array_equal(before, artifact['logits'])
    metrics = json.loads((full / 'metrics/valid_ema_argmax_notta.json').read_text())
    assert metrics['eval_partition'] == 'dev_cv_valid' and metrics['fold'] == 2
    for final_evaluation in (False, True):
        with pytest.raises(ValueError, match='official test remains sealed'):
            run_inference(cv_cfg, full / 'checkpoints/final_ema.pt', 'test', final_evaluation)
    original_status = (resumed / 'status.json').read_bytes()
    original_config = (resumed / 'resolved_config.yaml').read_bytes()
    wrong_fold = apply_overrides(cv_cfg, {'dataset.cv_fold': 1})
    with pytest.raises(ValueError, match='identity'):
        run_training(wrong_fold, 42, output_dir=resumed, resume=checkpoint_path)
    assert (resumed / 'status.json').read_bytes() == original_status
    assert (resumed / 'resolved_config.yaml').read_bytes() == original_config
    with pytest.raises(ValueError, match='identity'):
        run_inference(wrong_fold, full / 'checkpoints/final_ema.pt')


def test_fold_and_training_seed_are_separate_identity_dimensions(cv_cfg):
    manifest, label_map = cv_cfg['dataset']['manifest'], cv_cfg['dataset']['label_map']
    a = make_identity(cv_cfg, 42, manifest, label_map)
    b = make_identity(cv_cfg, 777, manifest, label_map)
    next_fold = apply_overrides(cv_cfg, {'dataset.cv_fold': 3})
    c = make_identity(next_fold, 42, manifest, label_map)
    assert a['split_hash'] == b['split_hash'] == c['split_hash']
    assert a['folds_hash'] == b['folds_hash'] == c['folds_hash']
    assert len({a['identity_hash'], b['identity_hash'], c['identity_hash']}) == 3
    assert run_name(cv_cfg, 42, a['identity_hash']).startswith('B04_seed42_fold2_')
    assert run_name(next_fold, 42, c['identity_hash']).startswith('B04_seed42_fold3_')
    assert make_loader(cv_cfg, 'valid').dataset.sample_ids == make_loader(copy.deepcopy(cv_cfg), 'valid').dataset.sample_ids


@pytest.mark.parametrize('ema_state', ['older_epoch', 'missing'])
def test_resume_after_final_regular_commit_recovers_ema_without_training(cv_cfg, tmp_path, monkeypatch, ema_state):
    import src.train_utils as train_utils
    run_dir = tmp_path / 'interrupted_final_save'
    save_checkpoint = train_utils.save_checkpoint

    def fail_final_ema(path, payload):
        if path.name == 'final_ema.pt' and payload['epoch'] == cv_cfg['train']['epochs']:
            raise RuntimeError('injected final EMA save failure')
        return save_checkpoint(path, payload)

    monkeypatch.setattr(train_utils, 'save_checkpoint', fail_final_ema)
    with pytest.raises(RuntimeError, match='injected final EMA'):
        run_training(cv_cfg, 42, output_dir=run_dir)
    regular_path, ema_path = [run_dir / f'checkpoints/final_{weight_type}.pt'
                              for weight_type in ('regular', 'ema')]
    regular = load_checkpoint(regular_path)
    assert regular['epoch'] == 2 and load_checkpoint(ema_path)['epoch'] == 1
    failed_status = json.loads((run_dir / 'status.json').read_text())
    assert failed_status['status'] == 'failed' and failed_status['completed_epochs'] == 2
    if ema_state == 'missing':
        ema_path.unlink()
    previous_hash = file_hash(ema_path) if ema_path.exists() else None
    regular_hash = file_hash(regular_path)
    monkeypatch.setattr(train_utils, 'save_checkpoint', save_checkpoint)

    def forbid_training(*args, **kwargs):
        raise AssertionError('A completed Regular checkpoint must resume without another training epoch')

    monkeypatch.setattr(train_utils, 'train_epoch', forbid_training)
    run_training(cv_cfg, 42, output_dir=run_dir, resume=regular_path)
    restored = load_checkpoint(ema_path)
    assert restored['epoch'] == 2 and restored['weight_type'] == 'ema'
    assert restored['identity_hash'] == regular['identity_hash']
    assert set(restored['model']) == set(regular['ema'])
    assert all(torch.equal(restored['model'][key], value) for key, value in regular['ema'].items())
    assert not any(key in restored for key in ('optimizer', 'scheduler', 'scaler', 'rng', 'ema'))
    assert file_hash(regular_path) == regular_hash
    status = json.loads((run_dir / 'status.json').read_text())
    provenance = json.loads((run_dir / 'provenance.json').read_text())
    assert status['status'] == 'complete' and status['completed_epochs'] == 2
    event = provenance['checkpoint_recovery_events'][-1]
    assert event['reason'] == ema_state and event['previous_checkpoint_hash'] == previous_hash
    assert event['regular_checkpoint_hash'] == regular_hash and event['restored_epoch'] == 2
    assert event['restored_checkpoint_hash'] == file_hash(ema_path)
    with (run_dir / 'history.csv').open(newline='') as stream:
        assert [int(row['epoch']) for row in csv.DictReader(stream)] == [1, 2]
    prediction_path = run_dir / 'predictions/valid_ema_argmax_notta.npz'
    with np.load(prediction_path, allow_pickle=False) as artifact:
        assert artifact['checkpoint_hash'].item() == file_hash(ema_path)
        before = artifact['logits'].copy()
    ema_hash = file_hash(ema_path)
    run_inference(cv_cfg, ema_path)
    # A second resume must preserve the valid EMA file's bytes and bound hash.
    run_training(cv_cfg, 42, output_dir=run_dir, resume=regular_path)
    assert file_hash(ema_path) == ema_hash
    with np.load(prediction_path, allow_pickle=False) as artifact:
        assert np.array_equal(before, artifact['logits'])


def test_prediction_metrics_missing_is_recovered_but_changed_metrics_are_preserved(cv_cfg, tmp_path):
    run_dir = run_training(cv_cfg, 42, output_dir=tmp_path / 'missing_metrics')
    checkpoint = run_dir / 'checkpoints/final_ema.pt'
    prediction = run_dir / 'predictions/valid_ema_argmax_notta.npz'
    metrics = run_dir / 'metrics/valid_ema_argmax_notta.json'
    original_metrics = json.loads(metrics.read_text())
    prediction_hash = file_hash(prediction)
    metrics.unlink()
    assert run_inference(cv_cfg, checkpoint) == prediction
    assert json.loads(metrics.read_text()) == original_metrics
    assert file_hash(prediction) == prediction_hash
    changed = {**original_metrics, 'seed': 777}
    metrics.write_text(json.dumps(changed))
    metrics_hash = file_hash(metrics)
    with pytest.raises(ValueError, match='metrics changed'):
        run_inference(cv_cfg, checkpoint)
    assert file_hash(prediction) == prediction_hash and file_hash(metrics) == metrics_hash


def test_prediction_write_interruption_never_publishes_a_partial_archive(cv_cfg, tmp_path, monkeypatch):
    run_dir = run_training(cv_cfg, 42, output_dir=tmp_path / 'atomic_predictions')
    checkpoint = run_dir / 'checkpoints/final_ema.pt'
    prediction = run_dir / 'predictions/valid_ema_argmax_notta.npz'
    metrics = run_dir / 'metrics/valid_ema_argmax_notta.json'
    prediction.unlink()
    metrics.unlink()
    original_files = set(prediction.parent.iterdir())

    def abort_zip(stream, **artifact):
        assert stream.name != str(prediction)
        stream.write(b'PK\x03\x04incomplete synthetic archive')
        raise KeyboardInterrupt('injected prediction write interruption')

    with monkeypatch.context() as patch:
        patch.setattr(np, 'savez_compressed', abort_zip)
        with pytest.raises(KeyboardInterrupt, match='injected prediction'):
            run_inference(cv_cfg, checkpoint)
    assert not prediction.exists() and not metrics.exists()
    assert set(prediction.parent.iterdir()) == original_files
    assert run_inference(cv_cfg, checkpoint) == prediction
    assert metrics.is_file()
    # A damaged formal archive is retained for inspection and is never replaced.
    prediction.write_bytes(b'PK\x03\x04damaged archive')
    damaged_hash = file_hash(prediction)
    metrics_hash = file_hash(metrics)
    with pytest.raises((ValueError, OSError, zipfile.BadZipFile)):
        run_inference(cv_cfg, checkpoint)
    assert file_hash(prediction) == damaged_hash and file_hash(metrics) == metrics_hash
