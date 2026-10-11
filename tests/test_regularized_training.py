"""Verify paired Mixup, seed-42-only scheduling, and recoverable diagnostics."""
import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.config import ConfigError, apply_overrides, load_config, validate_config
from src.train_utils import capture_rng, restore_rng, mixup_inputs, prediction_cross_entropy, load_checkpoint
from scripts import train_server


def test_mixup_pairs_both_modalities_and_keeps_ids_and_masks_out(monkeypatch):
    monkeypatch.setattr(np.random, 'beta', lambda *_: .25)
    monkeypatch.setattr(torch, 'randperm', lambda n, **_: torch.tensor([2, 0, 1]))
    clinical = torch.arange(3.).view(3,1,1,1)
    derm = clinical + 10
    mask = torch.ones(3,2,dtype=torch.bool)
    labels = torch.tensor([0,1,2])
    inputs = {'clinical':clinical,'dermoscopic':derm,'modality_mask':mask}
    targets = {'diagnosis':labels}
    mixed,a,b,lam = mixup_inputs(inputs,targets,.2)
    torch.testing.assert_close(mixed['dermoscopic']-mixed['clinical'],torch.full_like(clinical,10))
    torch.testing.assert_close(mixed['clinical'], .25*clinical+.75*clinical[[2,0,1]])
    assert torch.equal(b['diagnosis'],torch.tensor([2,0,1])) and a is targets
    assert mixed['modality_mask'] is mask and torch.equal(inputs['clinical'],clinical)
    assert set(mixed)==set(inputs) and lam==.25


def test_mixup_rng_replay_and_singleton():
    torch.manual_seed(42);np.random.seed(42)
    generator=torch.Generator().manual_seed(42)
    inputs={'clinical':torch.arange(12.).reshape(4,3)};targets={'diagnosis':torch.arange(4)}
    state=capture_rng(generator)
    a=mixup_inputs(inputs,targets,.2)
    restore_rng(state,generator)
    b=mixup_inputs(inputs,targets,.2)
    assert a[3]==b[3] and torch.equal(a[0]['clinical'],b[0]['clinical'])
    for alpha in (0,.2):
        x={'clinical':torch.zeros(1,3)};y={'diagnosis':torch.zeros(1,dtype=torch.long)}
        out=mixup_inputs(x,y,alpha)
        assert out[0] is x and out[1] is y and out[3]==1.


@pytest.mark.parametrize('key,value',[('model.head_dropout',1.),('model.head_dropout',-.1),
                                     ('train.mixup_alpha',-.1),('evaluation.track_regular',1)])
def test_invalid_regularization_is_refused(key,value):
    cfg=apply_overrides(load_config('configs/base.yaml'),{key:value})
    with pytest.raises(ConfigError):validate_config(cfg)


def test_profile_is_exactly_b04_seed42_five_folds(tmp_path):
    args=train_server.parser().parse_args(['--profile','b04_regularized','--data-root','data/derm7pt',
                                          '--job-dir',str(tmp_path/'job')])
    directory,job=train_server._resolve_job(args)
    from scripts.run_experiments import expand_suite,build_plan
    plan=build_plan(expand_suite(directory/'suite.yaml',only=job['settings']['only'],overrides=job['overrides']))
    assert {(r['spec_id'],r['seed'],r['fold']) for r in plan['runs']}=={('B04',42,f) for f in range(5)}
    assert job['settings']['batch_size']==16 and job['settings']['gradient_accumulation']==1
    for cfg in plan['resolved_configs'].values():
        assert cfg['model']['head_dropout']==.5 and cfg['train']['mixup_alpha']==.2
        assert cfg['train']['ema']['decay']==.99 and cfg['evaluation']['track_train']
        assert cfg['evaluation']['checkpoint_type']=='ema' and cfg['train']['epochs']==25
    for flags in (['--only','B01'],['--seeds','101'],['--protocol','official_holdout']):
        bad=train_server.parser().parse_args(['--profile','b04_regularized','--data-root','data/derm7pt',
                       '--job-dir',str(tmp_path/'invalid'),*flags])
        with pytest.raises(ConfigError,match='requires only B04'):train_server._resolve_job(bad)
        assert not (tmp_path/'invalid').exists()


def test_regularized_training_diagnostics_and_exact_resume(tmp_path):
    from scripts.smoke_test import create_fixture
    from src.train import run_training
    torch.set_num_threads(2)
    root=tmp_path/'data';manifest=create_fixture(root)
    cfg=apply_overrides(load_config('configs/development/b04_regularized.yaml'),{
        'dataset.root':str(root),'dataset.manifest':str(manifest/'samples.csv'),
        'dataset.label_map':str(manifest/'label_map.json'),'dataset.clinical_substitution_policy':'retain_official',
        'model.clinical_encoder':'tiny_cnn','model.dermoscopic_encoder':'tiny_cnn','model.pretrained_id':None,
        'preprocess.image_size':32,'train.device':'cpu','train.num_workers':0,
        'train.epochs':2,'train.warmup_epochs':0,'train.batch_size':3,
        'experiment.output_root':str(tmp_path/'runs')})
    full=run_training(cfg,42,output_dir=tmp_path/'full')
    resumed=run_training(cfg,42,output_dir=tmp_path/'resumed',stop_after_epoch=1)
    run_training(cfg,42,output_dir=resumed,resume=resumed/'checkpoints/final_regular.pt')
    a=load_checkpoint(full/'checkpoints/final_regular.pt');b=load_checkpoint(resumed/'checkpoints/final_regular.pt')
    for group in ('model','ema'):
        assert all(torch.equal(a[group][key],b[group][key]) for key in a[group])
    assert a['history']==b['history']
    for row in a['history']:
        assert row['train_accuracy_definition']=='mixup_weighted_label_accuracy'
        assert all(np.isfinite(row[k]) for k in ['valid_regular_loss','valid_loss','train_regular_eval_loss'])
        assert 0<=row['train_regular_eval_accuracy']<=1
    for weight in ('ema','regular'):
        pred=np.load(full/f'predictions/valid_{weight}_argmax_notta.npz',allow_pickle=False)
        assert pred['checkpoint_type'].item()==weight and len(pred['sample_id'])==5
        assert np.isfinite(prediction_cross_entropy(pred))
    assert json.loads((full/'provenance.json').read_text())['test_used'] is False
