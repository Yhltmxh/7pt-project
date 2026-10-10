"""One train/valid entry point for official holdout and test-sealed development CV."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from src.config import load_config, resolve_path, save_config, validate_config
from src.provenance import environment_info, file_hash, git_info, make_identity, run_name, utc_now, write_json


def _write_history(path, history):
    """Publish the history derived from the last committed Regular checkpoint."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, encoding='utf-8',
                                         newline='', delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _ensure_final_ema_checkpoint(regular, path, provenance, provenance_path):
    """Reconcile an inference-only EMA artifact with committed Regular state.

    A missing or older EMA can follow interruption between the two checkpoint
    writes. An incompatible or same-epoch inconsistent artifact fails closed.
    """
    import torch
    from src.train_utils import load_checkpoint, save_checkpoint
    path = Path(path)
    if regular.get('weight_type') != 'regular' or regular.get('ema') is None:
        raise ValueError('EMA recovery requires a trusted Regular payload with EMA state')
    expected = {key: value for key, value in regular.items()
                if key not in ('optimizer', 'scheduler', 'scaler', 'rng', 'ema')}
    expected.update(model=regular['ema'], weight_type='ema')
    previous_hash, previous_epoch = None, None
    if path.exists():
        previous_hash = file_hash(path)
        existing = load_checkpoint(path, regular['identity_hash'], regular['class_order'])
        previous_epoch = existing.get('epoch')
        if (set(existing) != set(expected) or existing.get('weight_type') != 'ema'
                or isinstance(previous_epoch, bool) or not isinstance(previous_epoch, int)
                or not 0 < previous_epoch <= regular['epoch']):
            raise ValueError('Existing EMA checkpoint is incompatible with committed Regular state')
        static_fields = ('schema_version', 'identity_hash', 'class_order', 'resolved_config',
                         'seed', 'fold', 'cv_fold', 'protocol', 'initial_model_hash',
                         'best_state', 'selection_protocol', 'weight_type')
        if any(existing.get(key) != expected.get(key) for key in static_fields):
            raise ValueError('Existing EMA checkpoint metadata differs from committed Regular state')
        if previous_epoch == regular['epoch']:
            actual_state, expected_state = existing['model'], expected['model']
            if (any(existing[key] != value for key, value in expected.items() if key != 'model')
                    or set(actual_state) != set(expected_state)
                    or any(not isinstance(actual_state[key], torch.Tensor)
                           or actual_state[key].dtype != value.dtype
                           or actual_state[key].shape != value.shape
                           or not torch.equal(actual_state[key], value.detach().cpu())
                           for key, value in expected_state.items())):
                raise ValueError('Existing final EMA checkpoint differs from committed Regular EMA state')
            return
        if (existing['history'] != regular['history'][:previous_epoch]
                or existing['ema_updates'] != sum(item['optimizer_updates'] for item in existing['history'])):
            raise ValueError('Existing older EMA checkpoint history differs from committed Regular state')
    event = {'at': utc_now(), 'artifact': 'checkpoints/final_ema.pt',
             'reason': 'missing' if previous_hash is None else 'older_epoch',
             'previous_checkpoint_hash': previous_hash, 'previous_epoch': previous_epoch,
             'regular_checkpoint_hash': file_hash(path.with_name('final_regular.pt')),
             'restored_epoch': regular['epoch']}
    provenance.setdefault('checkpoint_recovery_events', []).append(event)
    # Record the superseded artifact before its atomic replacement.
    write_json(provenance_path, provenance)
    save_checkpoint(path, expected)
    event['restored_checkpoint_hash'] = file_hash(path)
    write_json(provenance_path, provenance)


def preflight(cfg):
    validate_config(cfg, require_ready=True)
    from src.data.manifest import read_manifest, validate_manifest_against_source
    from src.data.labels import CLASS_ORDER, label_contract
    from src.data.splits import select_config_partition, prediction_partition
    root = resolve_path(cfg['dataset']['root'])
    manifest, label_map = (resolve_path(cfg['dataset'][k]) for k in ('manifest', 'label_map'))
    if not root.is_dir():
        raise ValueError(f'Dataset root does not exist: {root}')
    rows = read_manifest(manifest)
    raw_hashes = validate_manifest_against_source(rows, root,
        require_official_counts=cfg['stage'] == 'frozen')
    audit = json.loads((manifest.parent / 'dataset_audit.json').read_text(encoding='utf-8'))
    if audit.get('manifest_sha256') != file_hash(manifest):
        raise ValueError('Data audit does not match the current manifest; rerun data preparation')
    if not audit.get('passed'):
        raise ValueError(f'Data technical audit requires correction: {audit.get("errors", [])}')
    if cfg['stage'] == 'frozen' and not audit.get('formal_ready'):
        raise ValueError('Data evidence is not formal-ready; resolve and document the substitution/identity audit')
    label_info = json.loads(label_map.read_text(encoding='utf-8'))
    declared = label_info.get('class_order')
    if list(declared or []) != list(CLASS_ORDER):
        raise ValueError('Label map class order differs from BCC/NEV/MEL/MISC/SK')
    if label_info != label_contract():
        raise ValueError('Label map differs from the versioned author mapping')
    samples = [r['sample_id'] for r in rows]
    if len(set(samples)) != len(samples):
        raise ValueError('Manifest sample IDs are duplicated')
    counts = {part: sum(r['official_split'] == part for r in rows) for part in ('train', 'valid', 'test')}
    if sum(counts.values()) != len(rows) or any(v == 0 for v in counts.values()):
        raise ValueError('Manifest must contain nonempty disjoint official train/valid/test partitions')
    seen_rows = [int(r['source_row_index']) for r in rows]
    if sorted(seen_rows) != list(range(len(rows))):
        raise ValueError('Manifest source row coverage is incomplete or duplicated')
    seen_hashes = {}
    for row in rows:
        if not 0 <= int(row['diagnosis_index']) < 5:
            raise ValueError('Invalid manifest diagnosis index')
        for modality in cfg['model']['modalities']:
            relative = row[f'{modality}_path']
            path = (root / relative).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError('Image path escapes dataset root')
            expected = row.get(f'{modality}_sha256')
            if not expected or not path.is_file() or file_hash(path) != expected:
                raise ValueError(f'Missing or changed image: {relative}')
            previous = seen_hashes.setdefault(expected, row['official_split'])
            if previous != row['official_split']:
                raise ValueError('Identical image content crosses official partitions; audit requires review')
    from src.data.labels import diagnosis_index
    for row in rows:
        if diagnosis_index(row['diagnosis_raw']) != int(row['diagnosis_index']):
            raise ValueError('Manifest diagnosis mapping mismatch')
    protocol = cfg['dataset']['split_protocol']
    if protocol == 'dev_cv5' and cfg['stage'] != 'development':
        raise ValueError('dev_cv5 is a development-only protocol; official test remains sealed')
    training_rows = select_config_partition(rows, cfg, 'train')
    validation_rows = select_config_partition(rows, cfg, 'valid')
    if not training_rows or not validation_rows:
        raise ValueError('Configured training and validation partitions must be nonempty')
    train_ids = {row['sample_id'] for row in training_rows}
    valid_ids = {row['sample_id'] for row in validation_rows}
    if train_ids & valid_ids:
        raise ValueError('Configured training and validation samples overlap')
    import torch
    device_setting = cfg['train']['device']
    device = 'cuda' if device_setting == 'auto' and torch.cuda.is_available() else device_setting
    if device == 'auto':
        device = 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    output_root = resolve_path(cfg['experiment']['output_root'])
    parent = output_root
    while not parent.exists():
        parent = parent.parent
    # ConvNeXt-Base two branches: conservative allowance for state+EMA+optimizer+final exports.
    estimate = 6 * 1024**3 if 'convnext_base' in (
        cfg['model']['clinical_encoder'], cfg['model']['dermoscopic_encoder']) else 50 * 1024**2
    free = shutil.disk_usage(parent).free
    if free < estimate:
        raise ValueError(f'Insufficient output disk: need approximately {estimate} bytes, available {free}')
    return {'counts': counts, 'device': device, 'storage_estimate_bytes': estimate,
            'selection_counts': {'train': len(training_rows), 'valid': len(validation_rows)},
            'training_diagnosis_counts': [sum(row['diagnosis_index'] == c for row in training_rows) for c in range(5)],
            'protocol': protocol, 'fold': cfg['dataset']['cv_fold'] if protocol == 'dev_cv5' else -1,
            'eval_partition': prediction_partition(cfg, 'valid'),
            'folds_hash': file_hash(resolve_path(cfg['dataset']['folds_file'])) if protocol == 'dev_cv5' else None,
            'raw_source_hashes': raw_hashes,
            'audit_status': audit.get('status'),
            'disk_free_bytes': free, 'manifest_hash': file_hash(manifest),
            'label_map_hash': file_hash(label_map), 'epochs': cfg['train']['epochs'],
            'test_evaluation': False}


def make_loader(cfg, partition, generator=None, training=False):
    import torch
    from src.data.derm7pt import Derm7ptDataset
    from src.data.transforms import build_transforms
    from src.train_utils import seed_worker
    dataset = Derm7ptDataset(manifest=resolve_path(cfg['dataset']['manifest']),
        root=resolve_path(cfg['dataset']['root']), partition=partition,
        modalities=cfg['model']['modalities'],
        transform=build_transforms(cfg['preprocess'], training=training), cfg=cfg)
    return torch.utils.data.DataLoader(dataset, batch_size=cfg['train']['batch_size'],
        shuffle=training, num_workers=cfg['train']['num_workers'], generator=generator,
        worker_init_fn=seed_worker,
        pin_memory=cfg['train']['pin_memory'] and cfg['train']['device'] != 'cpu',
        prefetch_factor=cfg['train']['prefetch_factor'] if cfg['train']['num_workers'] else None,
        persistent_workers=cfg['train']['persistent_workers'])


def run_training(cfg, seed, output_dir=None, resume=None, stop_after_epoch=None):
    import torch
    from src.data.labels import CLASS_ORDER
    from src.evaluation.metrics import classification_metrics
    from src.losses import build_loss
    from src.models.model import build_model
    from src.train_utils import (ModelEMA, build_optimizer, build_scheduler, capture_rng,
        collect_predictions, load_checkpoint, restore_rng, save_checkpoint, set_seed, state_hash, train_epoch)
    report = preflight(cfg)
    manifest, label_map = (resolve_path(cfg['dataset'][k]) for k in ('manifest', 'label_map'))
    identity = make_identity(cfg, seed, manifest, label_map)
    run_dir = Path(output_dir) if output_dir is not None else resolve_path(cfg['experiment']['output_root']) / (
        run_name(cfg, seed, identity['identity_hash']))
    protocol = cfg['dataset']['split_protocol']
    fold = cfg['dataset']['cv_fold'] if protocol == 'dev_cv5' else -1
    if run_dir.exists() and resume is None:
        raise FileExistsError(f'Run exists; use explicit checkpoint resume after checking identity: {run_dir}')
    if resume is not None and not run_dir.exists():
        raise ValueError('Resume requires the original run directory')
    # Validate before writing status/config so rejected resumes preserve existing work.
    resume_checkpoint = None
    existing = None
    if resume is not None:
        if Path(resume).resolve().parents[1] != run_dir.resolve():
            raise ValueError('Resume checkpoint must belong to the original run directory')
        resume_checkpoint = load_checkpoint(resume, identity['identity_hash'], CLASS_ORDER)
        if protocol == 'dev_cv5' and (resume_checkpoint.get('fold') != fold
                or resume_checkpoint.get('cv_fold') != fold or resume_checkpoint.get('protocol') != protocol):
            raise ValueError('Resume checkpoint CV fold/protocol mismatch')
        if resume_checkpoint.get('weight_type') != 'regular' or 'optimizer' not in resume_checkpoint:
            raise ValueError('Training resume requires final_regular.pt with full optimizer/RNG state')
        existing = json.loads((run_dir / 'provenance.json').read_text(encoding='utf-8'))
        if existing['identity_hash'] != identity['identity_hash']:
            raise ValueError('Existing run identity differs; refusing to replace its artifacts')
        if load_config(run_dir / 'resolved_config.yaml') != cfg:
            raise ValueError('Existing run configuration differs; refusing to replace its artifacts')
        if existing['environment'] != environment_info() or existing['device'] != report['device']:
            raise ValueError('Exact resume requires the original Python/package environment and device')
        if (isinstance(resume_checkpoint.get('epoch'), bool)
                or not isinstance(resume_checkpoint.get('epoch'), int)
                or not 0 < resume_checkpoint['epoch'] <= cfg['train']['epochs']):
            raise ValueError('Checkpoint epoch exceeds or is outside the configured training budget')
    run_dir.mkdir(parents=True, exist_ok=True)
    status = {'stage': 'training', 'status': 'training', 'seed': seed,
              'spec_id': cfg['experiment']['id'], 'started_at': utc_now(),
              'protocol': protocol, 'fold': fold, 'cv_fold': fold if protocol == 'dev_cv5' else None,
              'identity_hash': identity['identity_hash'],
              'completed_epochs': resume_checkpoint['epoch'] if resume_checkpoint is not None else 0}
    write_json(run_dir / 'status.json', status)
    save_config(cfg, run_dir / 'resolved_config.yaml')
    started = time.monotonic()
    try:
        set_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        device = torch.device(report['device'])
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        # A resumed model is initialized without fetching pretrained weights.
        model_cfg = dict(cfg['model'])
        if resume is not None:
            model_cfg['pretrained_id'] = 'none'
        model = build_model(model_cfg, num_classes=5).to(device)
        ema = ModelEMA(model, cfg['train']['ema']['decay']) if cfg['train']['ema']['enabled'] else None
        optimizer = build_optimizer(model, cfg['train'])
        scheduler = build_scheduler(optimizer, cfg['train'])
        scaler = torch.amp.GradScaler('cuda', init_scale=1024.,
            enabled=device.type == 'cuda' and cfg['train']['mixed_precision'])
        train_loader = make_loader(cfg, 'train', generator, True)
        valid_loader = make_loader(cfg, 'valid')
        counts = [sum(r['diagnosis_index'] == c for r in train_loader.dataset.rows) for c in range(5)]
        criterion = build_loss(cfg['loss'], class_counts=counts).to(device)
        history, start_epoch, initial_hash = [], 0, None
        if resume is not None:
            checkpoint = resume_checkpoint
            model.load_state_dict(checkpoint['model'])
            if ema is not None:
                ema.module.load_state_dict(checkpoint['ema'])
                ema.updates = checkpoint['ema_updates']
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            scaler.load_state_dict(checkpoint['scaler'])
            restore_rng(checkpoint['rng'], generator)
            history, start_epoch = checkpoint['history'], checkpoint['epoch']
            initial_hash = checkpoint['initial_model_hash']
        else:
            initial_hash = state_hash(model)
        provenance = {**identity, 'git': git_info(), 'environment': environment_info(),
            'seed': seed, 'fold': fold, 'protocol': protocol, 'preflight': report,
            'cv_fold': fold if protocol == 'dev_cv5' else None,
            'training_sample_ids': list(train_loader.dataset.sample_ids),
            'validation_sample_ids': list(valid_loader.dataset.sample_ids),
            'class_counts_source': 'current_training_partition_only', 'class_counts': counts,
            'pretrained_id': cfg['model']['pretrained_id'], 'initial_model_hash': initial_hash,
            'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'amp_initial_scale': 1024.,
            'selection': 'fixed_final_epoch', 'test_used': False}
        provenance['resume_events'] = (existing.get('resume_events', []) if existing else []) + (
            [{'at': utc_now(), 'checkpoint': str(Path(resume).resolve()),
              'checkpoint_hash': file_hash(resume), 'from_epoch': start_epoch}]
            if resume is not None else [])
        provenance['checkpoint_recovery_events'] = existing.get('checkpoint_recovery_events', []) if existing else []
        write_json(run_dir / 'provenance.json', provenance)
        if start_epoch > cfg['train']['epochs']:
            raise ValueError('Checkpoint epoch exceeds the configured training budget')
        for epoch in range(start_epoch, cfg['train']['epochs']):
            lr = [group['lr'] for group in optimizer.param_groups]
            losses = train_epoch(model, train_loader, optimizer, criterion, device, scaler, cfg['train'], ema)
            evaluated_model = ema.module if cfg['evaluation']['checkpoint_type'] == 'ema' else model
            valid_predictions = collect_predictions(evaluated_model, valid_loader, device,
                                                    progress_bar=cfg['train']['progress_bar'])
            metrics = classification_metrics(valid_predictions['logits'], valid_predictions['y_true'])
            history.append({'epoch': epoch + 1, 'loss': losses['total'], 'diagnosis_loss': losses['diagnosis'],
                'attribute_loss': losses['attributes'], 'backbone_lr': lr[0], 'head_lr': lr[1],
                'optimizer_updates': losses['optimizer_updates'],
                'skipped_optimizer_steps': losses['skipped_optimizer_steps'],
                'valid_checkpoint_type': cfg['evaluation']['checkpoint_type'],
                'train_accuracy': losses['accuracy'],
                'valid_accuracy': metrics['accuracy'],
                'valid_diagnosis_macro_f1': metrics['diagnosis_macro_f1']})
            scheduler.step()
            payload = {'schema_version': 1, 'identity_hash': identity['identity_hash'],
                'class_order': list(CLASS_ORDER), 'resolved_config': cfg, 'seed': seed, 'epoch': epoch + 1,
                'fold': fold, 'cv_fold': fold if protocol == 'dev_cv5' else None, 'protocol': protocol,
                'model': model.state_dict(), 'ema': ema.module.state_dict() if ema else None,
                'ema_updates': ema.updates if ema else 0, 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'rng': capture_rng(generator),
                'history': history, 'initial_model_hash': initial_hash, 'best_state': None,
                'selection_protocol': 'fixed_final_epoch', 'weight_type': 'regular'}
            save_checkpoint(run_dir / 'checkpoints/final_regular.pt', payload)
            status['completed_epochs'] = epoch + 1
            if ema:
                save_checkpoint(run_dir / 'checkpoints/final_ema.pt', {
                    key: value for key, value in {**payload, 'model': ema.module.state_dict(),
                        'weight_type': 'ema'}.items()
                    if key not in ('optimizer', 'scheduler', 'scaler', 'rng', 'ema')})
            _write_history(run_dir / 'history.csv', history)
            write_json(run_dir / 'status.json', status)
            print(f"epoch={epoch+1}/{cfg['train']['epochs']} backbone_lr={lr[0]:.2e} head_lr={lr[1]:.2e} "
                  f"loss={losses['total']:.6f} train_acc={losses['accuracy']:.4f} "
                  f"valid_acc={metrics['accuracy']:.4f} valid_macro_f1={metrics['diagnosis_macro_f1']:.6f} "
                  f"updates={losses['optimizer_updates']} amp_skips={losses['skipped_optimizer_steps']}", flush=True)
            if stop_after_epoch is not None and epoch + 1 >= stop_after_epoch and epoch + 1 < cfg['train']['epochs']:
                status.update(status='interrupted', stage='training', reason='Explicit epoch-boundary stop')
                write_json(run_dir / 'status.json', status)
                return run_dir
        final_regular = payload if start_epoch < cfg['train']['epochs'] else resume_checkpoint
        if ema is not None:
            _ensure_final_ema_checkpoint(final_regular, run_dir / 'checkpoints/final_ema.pt',
                                         provenance, run_dir / 'provenance.json')
        _write_history(run_dir / 'history.csv', final_regular['history'])
        status['completed_epochs'] = final_regular['epoch']
        status.update(status='trained', stage='trained')
        write_json(run_dir / 'status.json', status)
        from src.infer import export_predictions
        status.update(status='predicting', stage='predicting')
        write_json(run_dir / 'status.json', status)
        for weight_type, candidate in [('regular', model)] + ([('ema', ema.module)] if ema else []):
            export_predictions(candidate, valid_loader, device, run_dir,
                run_dir / f'checkpoints/final_{weight_type}.pt', seed, 'valid', weight_type,
                protocol=protocol, fold=fold, identity=identity)
        artifacts = {p.relative_to(run_dir).as_posix(): file_hash(p) for folder in ('checkpoints', 'predictions', 'metrics')
                     for p in (run_dir / folder).glob('*') if p.is_file()}
        status.update(status='complete', stage='complete', completed_at=utc_now(),
            duration_seconds=time.monotonic() - started, artifact_hashes=artifacts,
            peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0)
        write_json(run_dir / 'status.json', status)
        return run_dir
    except Exception as error:
        status.update(status='failed', stage=status['stage'], error=f'{type(error).__name__}: {error}', failed_at=utc_now())
        write_json(run_dir / 'status.json', status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/base.yaml')
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--output-dir')
    parser.add_argument('--resume', help='Trusted project final_regular.pt only')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.preflight_only:
        print(json.dumps(preflight(cfg), indent=2))
    else:
        print(run_training(cfg, args.seed, args.output_dir, args.resume))


if __name__ == '__main__':
    main()
