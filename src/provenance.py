"""Stable artifact identity and atomic JSON writes; no ML dependency."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, encoding='utf-8', delete=False) as f:
        temporary = f.name
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    os.replace(temporary, path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def source_snapshot():
    files = sorted(p for folder in ('src', 'scripts', 'configs')
                   for p in (PROJECT_ROOT / folder).rglob('*')
                   if p.is_file() and p.suffix in ('.py', '.yaml', '.yml'))
    files += [PROJECT_ROOT / 'requirements.txt']
    hashes = {p.relative_to(PROJECT_ROOT).as_posix(): file_hash(p) for p in files}
    return {'files': hashes, 'sha256': object_hash(hashes)}


def git_info():
    def run(*args):
        try:
            r = subprocess.run(['git', *args], cwd=PROJECT_ROOT, capture_output=True, text=True)
            return r.stdout.strip() if r.returncode == 0 else None
        except FileNotFoundError:
            return None
    return {'revision': run('rev-parse', 'HEAD'), 'status': run('status', '--porcelain')}


def environment_info():
    packages = {}
    for name in ('torch', 'torchvision', 'numpy', 'Pillow', 'PyYAML', 'scikit-learn'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {'python': sys.version, 'executable': sys.executable, 'packages': packages}


def make_identity(cfg, seed, manifest, label_map):
    source = source_snapshot()
    data_root = Path(cfg['dataset']['root']).expanduser()
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    raw = {name: file_hash(data_root / 'meta' / name) for name in (
        'meta.csv', 'train_indexes.csv', 'valid_indexes.csv', 'test_indexes.csv')}
    components = {'config_hash': object_hash(cfg), 'source_hash': source['sha256'],
                  'manifest_hash': file_hash(manifest), 'label_map_hash': file_hash(label_map),
                  'raw_data_hash': object_hash(raw),
                  'seed': int(seed)}
    if cfg['dataset']['split_protocol'] == 'dev_cv5':
        folds = Path(cfg['dataset']['folds_file']).expanduser()
        if not folds.is_absolute():
            folds = PROJECT_ROOT / folds
        fold = cfg['dataset'].get('cv_fold')
        if type(fold) is not int or not 0 <= fold < 5:
            raise ValueError('CV run identity requires dataset.cv_fold in 0..4')
        components.update(folds_hash=file_hash(folds), cv_fold=fold)
        components['split_hash'] = object_hash({'protocol': 'dev_cv5',
            'folds_hash': components['folds_hash'], 'split_seed': cfg['dataset']['split_seed']})
    return {'identity_hash': object_hash(components), **components, 'source': source, 'raw_data_files': raw}


def run_name(cfg, seed, identity_hash):
    """A single canonical path convention for trainer and experiment runner."""
    name = f"{cfg['experiment']['id']}_seed{seed}"
    if cfg['dataset']['split_protocol'] == 'dev_cv5':
        fold = cfg['dataset'].get('cv_fold')
        if type(fold) is not int or not 0 <= fold < 5:
            raise ValueError('CV run path requires dataset.cv_fold in 0..4')
        name += f'_fold{fold}'
    return f'{name}_{identity_hash[:12]}'
