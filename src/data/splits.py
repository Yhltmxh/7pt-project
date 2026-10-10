"""Official positional indexes and test-sealed development partitions."""

import csv
from collections import defaultdict
from pathlib import Path
import random

OFFICIAL_PARTITIONS = ("train", "valid", "test")
OFFICIAL_COUNTS = {"train": 413, "valid": 203, "test": 395}


def read_official_indexes(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if "indexes" not in (reader.fieldnames or []):
            raise ValueError(f"{path}: required column 'indexes' is missing")
        indexes = []
        for line, row in enumerate(reader, start=2):
            try:
                indexes.append(int(row["indexes"]))
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}:{line}: indexes must be integers") from error
    return indexes


def validate_official_indexes(indexes, n_rows, require_official_counts=True):
    if set(indexes) != set(OFFICIAL_PARTITIONS):
        raise ValueError("Exactly train, valid, and test positional indexes are required")
    positions = {}
    for partition in OFFICIAL_PARTITIONS:
        for index in indexes[partition]:
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < n_rows:
                raise ValueError(f"{partition}: original row index out of bounds: {index!r}")
            if index in positions:
                raise ValueError(f"Duplicate original row index {index}: {positions[index]} and {partition}")
            positions[index] = partition
    missing = sorted(set(range(n_rows)) - set(positions))
    if missing:
        raise ValueError(f"Official indexes do not cover original rows: {missing[:20]}")
    counts = {partition: len(indexes[partition]) for partition in OFFICIAL_PARTITIONS}
    if require_official_counts and counts != OFFICIAL_COUNTS:
        raise ValueError(f"Official counts changed: expected {OFFICIAL_COUNTS}, got {counts}")
    return positions


def select_partition(rows, partition):
    if partition in OFFICIAL_PARTITIONS:
        selected = {partition}
    elif partition in ("refit", "dev", "train+valid"):
        selected = {"train", "valid"}
    elif partition == "all":
        selected = set(OFFICIAL_PARTITIONS)
    else:
        raise ValueError(f"Unknown partition: {partition!r}")
    return [row for row in rows if row["official_split"] in selected]


def make_dev_folds(rows, n_folds=5, seed=101):
    """Stratify verified lesion groups inside train+valid only.

    This does not infer patient identity. Test rows are never assigned a fold.
    Repeated lesion IDs remain together and conflicting diagnoses fail.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    groups = defaultdict(list)
    for row in select_partition(rows, "dev"):
        groups[row["lesion_id"]].append(row)
    by_class = defaultdict(list)
    for lesion_id, group in groups.items():
        labels = {row["diagnosis_index"] for row in group}
        if len(labels) != 1:
            raise ValueError(f"Conflicting diagnoses for lesion {lesion_id}")
        by_class[next(iter(labels))].append((lesion_id, group))
    rng = random.Random(seed)
    result = []
    fold_sizes = [0] * n_folds
    for label in sorted(by_class):
        group_list = sorted(by_class[label])
        if len(group_list) < n_folds:
            raise ValueError(f"Diagnosis {label} has fewer than {n_folds} lesion groups")
        rng.shuffle(group_list)
        label_sizes = [0] * n_folds
        for lesion_id, group in group_list:
            fold = min(range(n_folds), key=lambda value: (label_sizes[value], fold_sizes[value], value))
            label_sizes[fold] += len(group)
            fold_sizes[fold] += len(group)
            result.extend({"sample_id": row["sample_id"], "lesion_id": lesion_id, "source_row_index": row["source_row_index"], "dev_fold": fold} for row in group)
    return sorted(result, key=lambda row: row["source_row_index"])


DEV_FOLD_FIELDS = ("sample_id", "lesion_id", "source_row_index", "dev_fold")


def _fold_integer(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"Fold {field} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Fold {field} must be an integer") from error
    if result < 0:
        raise ValueError(f"Fold {field} must be nonnegative")
    return result


def _normalize_fold_assignment(record):
    if not isinstance(record, dict) or set(record) != set(DEV_FOLD_FIELDS):
        raise ValueError(f"Fold rows require exactly {DEV_FOLD_FIELDS}")
    result = dict(record)
    for field in ("sample_id", "lesion_id"):
        if not isinstance(result[field], str) or not result[field].strip():
            raise ValueError(f"Fold {field} must be a nonempty string")
    for field in ("source_row_index", "dev_fold"):
        result[field] = _fold_integer(result[field], field)
    if result["dev_fold"] not in range(5):
        raise ValueError("Development fold must be in 0..4")
    return result


def read_dev_folds(path):
    """Read the fixed five-fold assignment file without inferring missing rows."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if len(reader.fieldnames or []) != 4 or set(reader.fieldnames or []) != set(DEV_FOLD_FIELDS):
            raise ValueError(f"Fold CSV requires exactly {DEV_FOLD_FIELDS}")
        return [_normalize_fold_assignment(record) for record in reader]


def validate_dev_folds(rows, assignments, split_seed=None):
    """Validate test-sealed coverage, stratification and exact image separation.

    Passing split_seed additionally compares every assignment with the frozen
    deterministic make_dev_folds algorithm. Original official_split values are
    never changed. Identity remains at the verified case/lesion level.
    """
    development = []
    for source in select_partition(rows, "dev"):
        row = dict(source)
        for field in ("source_row_index", "diagnosis_index"):
            row[field] = _fold_integer(row[field], field)
        development.append(row)
    expected = {row["sample_id"]: row for row in development}
    if len(expected) != len(development) or not expected:
        raise ValueError("Development pool must have unique, nonempty sample coverage")
    normalized = [_normalize_fold_assignment(record) for record in assignments]
    sample_ids = [record["sample_id"] for record in normalized]
    positions = [record["source_row_index"] for record in normalized]
    if len(set(sample_ids)) != len(sample_ids) or len(set(positions)) != len(positions):
        raise ValueError("Duplicate sample or source row in fold coverage")
    if set(sample_ids) - set(expected):
        raise ValueError("Fold assignment includes test or unknown samples")
    if set(sample_ids) != set(expected):
        raise ValueError("Fold coverage must include every train+valid sample exactly once")
    if {record["dev_fold"] for record in normalized} != set(range(5)):
        raise ValueError("All five development folds 0..4 must be represented")
    lesion_folds, lesion_labels, image_folds = {}, {}, {}
    class_counts = defaultdict(lambda: [0] * 5)
    class_group_sizes = defaultdict(lambda: defaultdict(int))
    for record in normalized:
        source = expected[record["sample_id"]]
        fold = record["dev_fold"]
        if record["source_row_index"] != source["source_row_index"] or record["lesion_id"] != source["lesion_id"]:
            raise ValueError("Fold source row or lesion identity differs from the manifest")
        label = source["diagnosis_index"]
        if label not in range(5):
            raise ValueError("Fold diagnosis must follow the fixed five-class mapping")
        lesion = source["lesion_id"]
        if lesion_folds.setdefault(lesion, fold) != fold:
            raise ValueError("A lesion crosses development folds")
        if lesion_labels.setdefault(lesion, label) != label:
            raise ValueError("Conflicting diagnoses within a fold lesion group")
        class_counts[label][fold] += 1
        class_group_sizes[label][lesion] += 1
        for modality in ("clinical", "dermoscopic"):
            digest = source.get(f"{modality}_sha256")
            if digest and image_folds.setdefault(digest, fold) != fold:
                raise ValueError("Identical image content crosses development folds")
    if set(class_counts) != set(range(5)):
        raise ValueError("Development folds require all five diagnosis classes")
    for label, counts in class_counts.items():
        largest_group = max(class_group_sizes[label].values())
        if min(counts) == 0 or max(counts) - min(counts) > largest_group:
            raise ValueError("Development folds are not stratified over the five diagnosis classes")
    normalized.sort(key=lambda record: record["source_row_index"])
    if split_seed is not None:
        split_seed = _fold_integer(split_seed, "split_seed")
        if normalized != make_dev_folds(development, n_folds=5, seed=split_seed):
            raise ValueError("Fold assignments differ from the fixed split_seed")
    return normalized


def select_config_partition(rows, cfg, partition):
    """Select a configured train/valid view, preserving the official partition."""
    data = cfg["dataset"]
    protocol = data["split_protocol"]
    if protocol == "official_holdout":
        if partition not in OFFICIAL_PARTITIONS:
            raise ValueError(f"Unsupported holdout partition {partition!r}")
        return select_partition(rows, partition)
    if protocol != "dev_cv5":
        raise ValueError(f"Unsupported split protocol {protocol!r}")
    if partition not in ("train", "valid"):
        raise ValueError("Development CV only permits train/valid; official test remains sealed")
    fold = data.get("cv_fold")
    if isinstance(fold, bool) or not isinstance(fold, int) or fold not in range(5):
        raise ValueError("dev_cv5 requires a configured cv_fold in 0..4")
    if not data.get("folds_file"):
        raise ValueError("dev_cv5 requires a fixed folds_file")
    from src.config import resolve_path
    assignments = validate_dev_folds(rows, read_dev_folds(resolve_path(data["folds_file"])),
                                     split_seed=data["split_seed"])
    held_out = {record["sample_id"] for record in assignments if record["dev_fold"] == fold}
    return [row for row in select_partition(rows, "dev")
            if (row["sample_id"] in held_out) == (partition == "valid")]


def prediction_partition(cfg, partition):
    protocol = cfg["dataset"]["split_protocol"]
    if protocol == "official_holdout" and partition in OFFICIAL_PARTITIONS:
        return f"official_{partition}"
    if protocol == "dev_cv5" and partition == "valid":
        return "dev_cv_valid"
    raise ValueError("Unsupported prediction partition; development CV never predicts official test")
