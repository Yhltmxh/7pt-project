"""Synthetic end-to-end validation; never consumes Derm7pt benchmark cases."""
from __future__ import annotations

import argparse
import csv
import copy
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def create_fixture(root, *, n_cases=20, train_count=10, valid_count=5):
    import numpy as np
    from PIL import Image
    from src.data.labels import ATTRIBUTE_CLASSES
    from src.data.manifest import prepare_dataset
    (root / 'images').mkdir(parents=True)
    (root / 'meta').mkdir()
    rng = np.random.default_rng(413)
    diagnoses = ['basal cell carcinoma', 'clark nevus', 'melanoma', 'dermatofibroma', 'seborrheic keratosis']
    rows = []
    if train_count <= 0 or valid_count <= 0 or train_count + valid_count >= n_cases:
        raise ValueError('Synthetic fixture requires nonempty train/valid/test')
    for index in range(n_cases):
        row = dict(case_num=str(index), case_id=f'synthetic_{index}', diagnosis=diagnoses[index % 5],
                   clinic=f'c{index}.png', derm=f'd{index}.png', sex='female', location='back', elevation='flat')
        row.update({name: 'absent' for name in ATTRIBUTE_CLASSES})
        rows.append(row)
        for modality in ('c', 'd'):
            Image.fromarray(rng.integers(0, 256, (40, 48, 3), dtype=np.uint8)).save(root / 'images' / f'{modality}{index}.png')
    with (root / 'meta/meta.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for split, positions in [('train', range(train_count)),
            ('valid', range(train_count, train_count + valid_count)),
            ('test', range(train_count + valid_count, n_cases))]:
        with (root / f'meta/{split}_indexes.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['indexes'])
            writer.writerows([position] for position in positions)
    manifest_dir = root / 'manifests'
    audit = prepare_dataset(root, manifest_dir, require_official_counts=False)
    if not audit['passed']:
        raise AssertionError(f'Synthetic fixture failed technical audit: {audit}')
    return manifest_dir


def run_smoke(output_dir=None, device='cpu'):
    import numpy as np
    import torch
    from src.config import apply_overrides, load_config, validate_config
    from src.infer import run_inference
    from src.provenance import environment_info, write_json
    from src.train import run_training
    from src.train_utils import load_checkpoint
    torch.set_num_threads(2)
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA smoke requested but unavailable')
    base = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix='derm7pt-smoke-'))
    base.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='attempt-', dir=base))
    root = work / 'synthetic_data'
    manifest_dir = create_fixture(root)
    cfg = apply_overrides(load_config('configs/base.yaml'), {
        'stage': 'development', 'dataset.root': str(root.resolve()),
        'dataset.manifest': str(manifest_dir / 'samples.csv'),
        'dataset.label_map': str(manifest_dir / 'label_map.json'),
        'dataset.clinical_substitution_policy': 'retain_official',
        'model.clinical_encoder': 'tiny_cnn', 'model.dermoscopic_encoder': 'tiny_cnn',
        'model.pretrained_id': None, 'preprocess.image_size': 32,
        'train.epochs': 2, 'train.warmup_epochs': 0, 'train.batch_size': 3,
        'train.gradient_accumulation': 3, 'train.num_workers': 0, 'train.device': device,
        'train.backbone_learning_rate': .001, 'train.head_learning_rate': .001,
        'train.ema.decay': .9, 'experiment.output_root': str(work / 'runs')})
    validate_config(cfg, require_ready=True)
    for name, modalities in [('B01', ['clinical']), ('B02', ['dermoscopic'])]:
        single = apply_overrides(cfg, {'experiment.id': name, 'model.modalities': modalities, 'model.fusion': 'single'})
        run_training(single, 101, output_dir=work / name)
    uninterrupted = run_training(cfg, 101, output_dir=work / 'full')
    resumed = run_training(cfg, 101, output_dir=work / 'resumed', stop_after_epoch=1)
    checkpoint_path = resumed / 'checkpoints/final_regular.pt'
    run_training(cfg, 101, output_dir=resumed, resume=checkpoint_path)
    a, b = (load_checkpoint(path / 'checkpoints/final_regular.pt') for path in (uninterrupted, resumed))
    for key in a['model']:
        if not torch.equal(a['model'][key], b['model'][key]):
            raise AssertionError(f'Resumed regular weights differ: {key}')
    for key in a['ema']:
        if not torch.equal(a['ema'][key], b['ema'][key]):
            raise AssertionError(f'Resumed EMA weights differ: {key}')
    if a['history'] != b['history'] or a['ema_updates'] != b['ema_updates']:
        raise AssertionError('Resumed history/EMA update count differs')
    saved = np.load(uninterrupted / 'predictions/valid_ema_argmax_notta.npz', allow_pickle=False)['logits'].copy()
    run_inference(cfg, uninterrupted / 'checkpoints/final_ema.pt')
    reloaded = np.load(uninterrupted / 'predictions/valid_ema_argmax_notta.npz', allow_pickle=False)['logits']
    if not np.array_equal(saved, reloaded):
        raise AssertionError('Reloaded EMA predictions differ')
    changed = apply_overrides(cfg, {'train.head_learning_rate': .002})
    original_status = (resumed / 'status.json').read_bytes()
    original_config = (resumed / 'resolved_config.yaml').read_bytes()
    try:
        run_training(changed, 101, output_dir=work / 'resumed', resume=checkpoint_path)
    except ValueError as error:
        if 'identity' not in str(error):
            raise
    else:
        raise AssertionError('Config drift was accepted on resume')
    if original_status != (resumed / 'status.json').read_bytes() or original_config != (resumed / 'resolved_config.yaml').read_bytes():
        raise AssertionError('Rejected resume modified the existing run')
    try:
        run_inference(cfg, uninterrupted / 'checkpoints/final_ema.pt', partition='test')
    except ValueError as error:
        if 'Test requires' not in str(error):
            raise
    else:
        raise AssertionError('Development test inference was accepted')
    report = {'passed': True, 'synthetic_only': True, 'device': device,
        'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
        'checks': ['B01/B02/B04 training', 'finite loss and gradients', 'Regular and EMA export',
            'exact epoch-boundary resume', 'identical reloaded EMA predictions',
            'reject config drift', 'reject development test inference', 'uneven gradient accumulation window'],
        'environment': environment_info(), 'work_dir': str(work.resolve())}
    write_json(base / 'report.json', report)
    print(f'Synthetic {device} smoke passed: {base / "report.json"}')
    return report


def run_cv_smoke(output_dir=None, device='cpu'):
    """Real five-fold orchestration on tiny synthetic images; no benchmark data."""
    import csv
    import json
    import uuid
    import numpy as np
    import torch
    from unittest.mock import patch
    from scripts.run_experiments import main as run_suite
    from scripts.collect_results import collect_suite
    from src.config import apply_overrides, load_config, save_config
    from src.data.manifest import read_manifest
    from src.data.splits import make_dev_folds
    from src.infer import run_inference
    from src.provenance import environment_info, file_hash, write_json
    from src.train import run_training
    from src.train_utils import load_checkpoint
    import yaml

    torch.set_num_threads(2)
    base = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix='derm7pt-cv-smoke-'))
    base.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='attempt-', dir=base)).resolve()
    root = work / 'synthetic_data'
    manifest_dir = create_fixture(root, n_cases=30, train_count=15, valid_count=10)
    rows = read_manifest(manifest_dir / 'samples.csv')
    folds = make_dev_folds(rows, n_folds=5, seed=20261004)
    folds_file = work / 'folds.csv'
    with folds_file.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('sample_id', 'lesion_id', 'source_row_index', 'dev_fold'))
        writer.writeheader()
        writer.writerows(folds)
    cfg = apply_overrides(load_config('configs/base.yaml'), {
        'stage': 'development', 'dataset.root': str(root),
        'dataset.manifest': str(manifest_dir / 'samples.csv'),
        'dataset.label_map': str(manifest_dir / 'label_map.json'),
        'dataset.split_protocol': 'dev_cv5', 'dataset.folds_file': str(folds_file),
        'dataset.cv_fold': None, 'dataset.split_seed': 20261004,
        'dataset.clinical_substitution_policy': 'retain_official',
        'model.clinical_encoder': 'tiny_cnn', 'model.dermoscopic_encoder': 'tiny_cnn',
        'model.pretrained_id': None, 'preprocess.image_size': 32,
        'train.epochs': 2, 'train.warmup_epochs': 0, 'train.batch_size': 3,
        'train.gradient_accumulation': 3, 'train.num_workers': 0, 'train.device': device,
        'train.backbone_learning_rate': .001, 'train.head_learning_rate': .001,
        'train.ema.decay': .9, 'experiment.seeds': [42], 'experiment.output_root': str(work / 'runs')})
    base_file, suite_file = work / 'base.yaml', work / 'suite.yaml'
    save_config(cfg, base_file)
    suite_id = 'smoke_cv42_' + uuid.uuid4().hex
    suite = {'suite_id': suite_id, 'base_config': str(base_file), 'protocol_config': None, 'seeds': [42],
        'experiments': [
            {'id': 'B01', 'parent': 'base', 'overrides': {'model.modalities': ['clinical'], 'model.fusion': 'single'}},
            {'id': 'B02', 'parent': 'base', 'overrides': {'model.modalities': ['dermoscopic'], 'model.fusion': 'single'}},
            {'id': 'B04', 'parent': 'base', 'overrides': {'model.fusion': 'mean'}}]}
    suite_file.write_text(yaml.safe_dump(suite, sort_keys=False), encoding='utf-8')

    def stop_once(config, seed, output_dir=None, resume=None):
        assert seed == 42 and resume is None
        run_training(config, seed, output_dir=output_dir, stop_after_epoch=1)
        raise RuntimeError('Synthetic CV interruption after epoch 1')

    argv = ['--suite', str(suite_file)]
    with patch('src.train.run_training', stop_once):
        if run_suite(argv) != 2:
            raise AssertionError('Synthetic CV interruption was not recorded')
    if run_suite(argv + ['--resume']) != 0:
        raise AssertionError('Synthetic CV suite could not resume')
    suite_dir = Path(__file__).resolve().parents[1] / 'outputs/suites' / suite_id
    collection = collect_suite(suite_dir)
    if not collection['all_preregistered_seeds_complete'] or collection['formal']:
        raise AssertionError('Synthetic CV collection incomplete or incorrectly marked formal')
    plan = json.loads((suite_dir / 'plan.json').read_text(encoding='utf-8'))
    if plan['run_count'] != 15 or any(run['seed'] != 42 for run in plan['runs']):
        raise AssertionError('CV plan does not contain exactly 15 seed42 runs')
    first = plan['runs'][0]
    first_cfg = plan['resolved_configs'][first['config_key']]
    resumed = Path(first['run_dir'])
    reference = run_training(first_cfg, 42, output_dir=work / 'uninterrupted_reference')
    a, b = (load_checkpoint(path / 'checkpoints/final_regular.pt') for path in (reference, resumed))
    for key in ('model', 'ema'):
        if any(not torch.equal(a[key][name], b[key][name]) for name in a[key]):
            raise AssertionError(f'CV resumed {key} differs from uninterrupted training')
    if a['history'] != b['history']:
        raise AssertionError('CV resumed history differs')
    old_status = (resumed / 'status.json').read_bytes()
    try:
        run_inference(first_cfg, resumed / 'checkpoints/final_ema.pt', partition='test', final_evaluation=True)
    except ValueError:
        pass
    else:
        raise AssertionError('CV test inference was accepted')
    wrong_fold = apply_overrides(first_cfg, {'dataset.cv_fold': 1})
    try:
        run_training(wrong_fold, 42, output_dir=resumed, resume=resumed / 'checkpoints/final_regular.pt')
    except ValueError:
        pass
    else:
        raise AssertionError('Cross-fold checkpoint reuse was accepted')
    if (resumed / 'status.json').read_bytes() != old_status:
        raise AssertionError('Rejected cross-fold resume changed the run')
    expected = {row['sample_id'] for row in rows if row['official_split'] in ('train', 'valid')}
    oof_files = sorted((suite_dir / 'oof').glob('*.npz'))
    if len(oof_files) != 3:
        raise AssertionError('Expected three pooled OOF artifacts')
    for path in oof_files:
        with np.load(path, allow_pickle=False) as artifact:
            ids = list(map(str, artifact['sample_id']))
            if len(ids) != len(expected) or set(ids) != expected:
                raise AssertionError('OOF has missing, repeated, or test cases')
    hashes = {run['run_dir']: file_hash(Path(run['run_dir']) / 'checkpoints/final_regular.pt') for run in plan['runs']}
    with patch('src.train.run_training', side_effect=AssertionError('Complete CV run must not retrain')):
        if run_suite(argv + ['--resume']) != 0:
            raise AssertionError('Complete CV verification/reuse failed')
    if any(file_hash(Path(path) / 'checkpoints/final_regular.pt') != sha for path, sha in hashes.items()):
        raise AssertionError('Complete CV reuse changed checkpoints')
    report = {'passed': True, 'synthetic_only': True, 'device': device, 'training_seed': 42,
        'split_seed': 20261004, 'n_folds': 5, 'run_count': 15, 'development_cases': len(expected),
        'suite_dir': str(suite_dir), 'work_dir': str(work), 'environment': environment_info(),
        'checks': ['B01/B02/B04 five-fold training', 'exact epoch-boundary CV recovery',
                   'pooled OOF coverage', 'strict collection', 'reject cross-fold reuse',
                   'reject CV test inference', 'verified complete reuse'], 'formal': False}
    write_json(base / 'report.json', report)
    print(f'Synthetic seed42 five-fold smoke passed: {base / "report.json"}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--protocol', choices=['official_holdout', 'dev_cv5'], default='official_holdout')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    (run_cv_smoke if args.protocol == 'dev_cv5' else run_smoke)(args.output_dir, args.device)


if __name__ == '__main__':
    main()
