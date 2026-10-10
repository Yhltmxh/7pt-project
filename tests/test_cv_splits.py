"""Five development folds preserve official rows and never expose test cases."""

import copy
import csv

import pytest

from src.data.splits import (
    DEV_FOLD_FIELDS, make_dev_folds, prediction_partition, read_dev_folds,
    select_config_partition, validate_dev_folds,
)


SPLIT_SEED = 20261004


def cv_rows():
    return [dict(sample_id=f's{index}', lesion_id=f'l{index}', source_row_index=index,
                 official_split='train' if index < 15 else 'valid' if index < 25 else 'test',
                 diagnosis_index=index % 5, exclusion_reason='',
                 clinical_sha256=f'clinical_{index}', dermoscopic_sha256=f'dermoscopic_{index}')
            for index in range(30)]


def write_folds(path, assignments):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=DEV_FOLD_FIELDS)
        writer.writeheader()
        writer.writerows(assignments)
    return path


def cv_config(path, fold=2):
    return {'dataset': {'split_protocol': 'dev_cv5', 'folds_file': str(path),
                        'cv_fold': fold, 'split_seed': SPLIT_SEED},
            'experiment': {'seeds': [42]}}


def test_fixed_folds_select_disjoint_development_pool_and_preserve_official_rows(tmp_path):
    rows = cv_rows()
    original = copy.deepcopy(rows)
    assignments = make_dev_folds(rows, 5, SPLIT_SEED)
    path = write_folds(tmp_path / 'folds.csv', assignments)
    assert read_dev_folds(path) == assignments
    assert validate_dev_folds(rows, assignments, SPLIT_SEED) == assignments
    seen = []
    for fold in range(5):
        cfg = cv_config(path, fold)
        train = select_config_partition(rows, cfg, 'train')
        valid = select_config_partition(rows, cfg, 'valid')
        assert len(train) == 20 and len(valid) == 5
        assert {row['sample_id'] for row in train}.isdisjoint(row['sample_id'] for row in valid)
        assert {row['diagnosis_index'] for row in valid} == set(range(5))
        assert all(row['official_split'] != 'test' for row in train + valid)
        assert any(row['official_split'] == 'valid' for row in train)
        seen.extend(row['sample_id'] for row in valid)
        cfg['experiment']['seeds'] = [777]
        assert select_config_partition(rows, cfg, 'valid') == valid
    assert sorted(seen) == sorted(row['sample_id'] for row in rows[:25])
    assert rows == original


def test_collector_csv_strings_are_normalized_without_mutating_rows(tmp_path):
    rows = cv_rows()
    assignments = make_dev_folds(rows, 5, SPLIT_SEED)
    strings = [{key: str(value) for key, value in row.items()} for row in rows]
    original = copy.deepcopy(strings)
    path = write_folds(tmp_path / 'folds.csv', assignments)
    assert validate_dev_folds(strings, read_dev_folds(path), SPLIT_SEED) == assignments
    selected = select_config_partition(strings, cv_config(path), 'valid')
    assert len(selected) == 5
    assert isinstance(selected[0]['diagnosis_index'], str)
    assert strings == original


@pytest.mark.parametrize('mutation, message', [
    ('missing', 'every train\\+valid'), ('duplicate', 'Duplicate'),
    ('test', 'test or unknown'), ('identity', 'identity'),
    ('out_of_range', '0..4'), ('fractional', 'integer'),
    ('boolean', 'integer'), ('unknown_field', 'exactly'),
])
def test_fold_coverage_and_identity_reject_partial_or_unknown_rows(mutation, message):
    rows = cv_rows()
    assignments = make_dev_folds(rows, 5, SPLIT_SEED)
    if mutation == 'missing':
        assignments.pop()
    elif mutation == 'duplicate':
        assignments.append(dict(assignments[0]))
    elif mutation == 'test':
        assignments.append(dict(sample_id='s25', lesion_id='l25', source_row_index=25, dev_fold=0))
    elif mutation == 'identity':
        assignments[0]['source_row_index'] = 100
    elif mutation == 'out_of_range':
        assignments[0]['dev_fold'] = 5
    elif mutation == 'fractional':
        assignments[0]['source_row_index'] = .9
    elif mutation == 'boolean':
        assignments[0]['dev_fold'] = True
    else:
        assignments[0]['extra'] = 1
    with pytest.raises(ValueError, match=message):
        validate_dev_folds(rows, assignments, SPLIT_SEED)


def test_split_seed_rejects_balanced_assignment_drift():
    rows = cv_rows()
    assignments = make_dev_folds(rows, 5, SPLIT_SEED)
    first, second = [record for record in assignments if int(record['source_row_index']) % 5 == 0][:2]
    first['dev_fold'], second['dev_fold'] = second['dev_fold'], first['dev_fold']
    validate_dev_folds(rows, assignments)
    with pytest.raises(ValueError, match='split_seed'):
        validate_dev_folds(rows, assignments, SPLIT_SEED)


def test_stratification_and_all_five_folds_are_required():
    rows = cv_rows()
    assignments = make_dev_folds(rows, 5, SPLIT_SEED)
    assignments[0]['dev_fold'] = (assignments[0]['dev_fold'] + 1) % 5
    with pytest.raises(ValueError, match='stratified'):
        validate_dev_folds(rows, assignments)
    assignments = make_dev_folds(rows, 5, SPLIT_SEED)
    for record in assignments:
        if record['dev_fold'] == 4:
            record['dev_fold'] = 3
    with pytest.raises(ValueError, match='All five'):
        validate_dev_folds(rows, assignments)
    with pytest.raises(ValueError, match='fewer than 5'):
        make_dev_folds(rows[:20], 5, SPLIT_SEED)


def test_repeated_lesion_and_image_content_cannot_cross_folds():
    rows = cv_rows()
    assignments = make_dev_folds(rows, 5, SPLIT_SEED)
    first, second = [record for record in assignments if record['source_row_index'] % 5 == 0][:2]
    a, b = first['source_row_index'], second['source_row_index']
    rows[b]['clinical_sha256'] = rows[a]['dermoscopic_sha256']
    with pytest.raises(ValueError, match='image content crosses'):
        validate_dev_folds(rows, assignments)
    rows = cv_rows()
    rows[b]['lesion_id'] = rows[a]['lesion_id']
    second['lesion_id'] = first['lesion_id']
    with pytest.raises(ValueError, match='lesion crosses'):
        validate_dev_folds(rows, assignments)


def test_fold_csv_has_fixed_schema(tmp_path):
    path = tmp_path / 'bad.csv'
    path.write_text('sample_id,lesion_id,source_row_index,dev_fold,extra\ns0,l0,0,0,x\n')
    with pytest.raises(ValueError, match='exactly'):
        read_dev_folds(path)
    path.write_text('sample_id,lesion_id,source_row_index,dev_fold\ns0,l0,0.9,0\n')
    with pytest.raises(ValueError, match='integer'):
        read_dev_folds(path)


def test_cv_blocks_test_and_unknown_partitions_while_holdout_stays_compatible(tmp_path):
    rows = cv_rows()
    cfg = cv_config(write_folds(tmp_path / 'folds.csv', make_dev_folds(rows, 5, SPLIT_SEED)))
    for partition in ('test', 'dev', 'all', 'refit'):
        with pytest.raises(ValueError, match='test remains sealed'):
            select_config_partition(rows, cfg, partition)
    assert prediction_partition(cfg, 'valid') == 'dev_cv_valid'
    with pytest.raises(ValueError, match='never predicts official test'):
        prediction_partition(cfg, 'test')
    holdout = {'dataset': {'split_protocol': 'official_holdout'}}
    for partition, count in (('train', 15), ('valid', 10), ('test', 5)):
        assert len(select_config_partition(rows, holdout, partition)) == count
        assert prediction_partition(holdout, partition) == f'official_{partition}'


@pytest.mark.parametrize('fold', [None, True, -1, 5, '0'])
def test_cv_fold_requires_integer_0_to_4(tmp_path, fold):
    rows = cv_rows()
    cfg = cv_config(write_folds(tmp_path / 'folds.csv', make_dev_folds(rows, 5, SPLIT_SEED)), fold)
    with pytest.raises(ValueError, match='cv_fold'):
        select_config_partition(rows, cfg, 'valid')
