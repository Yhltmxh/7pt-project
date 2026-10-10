"""One inference entry point with explicit frozen-test boundary."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from src.config import load_config, resolve_path
from src.provenance import file_hash, make_identity, write_json


def export_predictions(model, loader, device, run_dir, checkpoint_path, seed, partition, weight_type,
                       *, protocol='official_holdout', fold=-1, identity=None):
    import numpy as np
    from src.data.labels import CLASS_ORDER
    from src.evaluation.metrics import validate_predictions
    from src.train_utils import collect_predictions
    from src.data.splits import prediction_partition
    eval_partition = prediction_partition({'dataset': {'split_protocol': protocol}}, partition)
    if protocol == 'dev_cv5' and (isinstance(fold, bool) or not isinstance(fold, int) or fold not in range(5)):
        raise ValueError('Development CV predictions require fold in 0..4')
    if protocol == 'dev_cv5' and (identity is None or identity.get('cv_fold') != fold):
        raise ValueError('Development CV predictions require matching fold identity metadata')
    artifact = collect_predictions(model, loader, device)
    artifact.update(schema_version=np.asarray(1), class_order=np.asarray(CLASS_ORDER), seed=np.asarray(seed), fold=np.asarray(fold),
                    protocol=np.asarray(protocol), eval_partition=np.asarray(eval_partition), checkpoint_hash=np.asarray(file_hash(checkpoint_path)),
                    checkpoint_type=np.asarray(weight_type), tta=np.asarray(False), decision_rule=np.asarray('argmax'))
    if protocol == 'dev_cv5':
        for field in ('identity_hash', 'config_hash', 'source_hash', 'manifest_hash', 'label_map_hash',
                      'raw_data_hash', 'folds_hash', 'split_hash', 'cv_fold'):
            if field not in identity:
                raise ValueError(f'Development CV prediction identity lacks {field}')
            artifact[field] = np.asarray(identity[field])
    expected_ids = loader.dataset.sample_ids
    metrics = validate_predictions(artifact, expected_ids)
    metrics.update(checkpoint_hash=str(artifact['checkpoint_hash']), checkpoint_type=weight_type,
                   eval_partition=eval_partition, protocol=protocol, fold=fold, seed=seed, tta=False, decision_rule='argmax')
    stem = f'{partition}_{weight_type}_argmax_notta'
    path = Path(run_dir) / 'predictions' / f'{stem}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(run_dir) / 'metrics' / f'{stem}.json'
    if metrics_path.exists() and json.loads(metrics_path.read_text(encoding='utf-8')) != metrics:
        raise ValueError('Existing prediction metrics changed; refusing overwrite')
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if set(existing.files) != set(artifact) or any(
                    not np.array_equal(existing[key], value) for key, value in artifact.items()):
                raise FileExistsError(f'Existing predictions differ; refusing overwrite: {path}')
        if not metrics_path.exists():
            write_json(metrics_path, metrics)
        return path
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('wb', dir=path.parent, suffix='.npz', delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **artifact)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    write_json(metrics_path, metrics)
    return path


def run_inference(cfg, checkpoint_path, partition='valid', final_evaluation=False):
    import torch
    from src.data.labels import CLASS_ORDER
    from src.models.model import build_model
    from src.train import make_loader, preflight
    from src.train_utils import load_checkpoint
    protocol = cfg['dataset']['split_protocol']
    if protocol == 'dev_cv5' and partition != 'valid':
        raise ValueError('Development CV only predicts its validation fold; official test remains sealed')
    if partition == 'test' and (not final_evaluation or cfg['stage'] != 'frozen'):
        raise ValueError('Test requires stage=frozen and explicit --final-evaluation')
    report = preflight(cfg)
    checkpoint = load_checkpoint(checkpoint_path, class_order=CLASS_ORDER)
    identity = make_identity(cfg, checkpoint['seed'], resolve_path(cfg['dataset']['manifest']), resolve_path(cfg['dataset']['label_map']))
    if checkpoint['identity_hash'] != identity['identity_hash']:
        raise ValueError('Inference source/config/data identity differs from checkpoint')
    if protocol == 'dev_cv5' and (checkpoint.get('fold') != cfg['dataset']['cv_fold']
            or checkpoint.get('cv_fold') != cfg['dataset']['cv_fold'] or checkpoint.get('protocol') != protocol):
        raise ValueError('Inference checkpoint CV fold/protocol mismatch')
    if checkpoint['epoch'] != cfg['train']['epochs']:
        raise ValueError('Inference requires the completed fixed-epoch checkpoint')
    model_cfg = dict(cfg['model'], pretrained_id='none')
    model = build_model(model_cfg, num_classes=5).to(report['device'])
    model.load_state_dict(checkpoint['model'])
    loader = make_loader(cfg, partition)
    run_dir = Path(checkpoint_path).resolve().parents[1]
    return export_predictions(model, loader, torch.device(report['device']), run_dir, checkpoint_path,
                              checkpoint['seed'], partition, checkpoint['weight_type'], protocol=protocol,
                              fold=cfg['dataset']['cv_fold'] if protocol == 'dev_cv5' else -1, identity=identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--partition', choices=['valid', 'test'], default='valid')
    parser.add_argument('--final-evaluation', action='store_true')
    args = parser.parse_args()
    print(run_inference(load_config(args.config), args.checkpoint, args.partition, args.final_evaluation))


if __name__ == '__main__':
    main()
