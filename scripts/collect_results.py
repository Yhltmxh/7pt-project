"""Verify holdout or five-fold development predictions and aggregate seeds."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import re
import statistics
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASS_ORDER, ConfigError, config_hash, load_config, resolve_path
from src.provenance import file_hash, object_hash, write_json

METRICS = ("diagnosis_macro_f1", "accuracy", "balanced_accuracy", "macro_ovr_auroc")
FORMAL_SEEDS = {101, 3407, 777}
CV_FOLDS = set(range(5))
PREDICTION_FIELDS = {"sample_id", "y_true", "logits", "probabilities", "class_order", "seed", "fold", "eval_partition", "checkpoint_hash"}
CV_IDENTITY_FIELDS = ("identity_hash", "config_hash", "source_hash", "manifest_hash", "label_map_hash",
                      "raw_data_hash", "folds_hash", "split_hash", "cv_fold")
CV_COMMON_FIELDS = ("parent_config_hash", "source_hash", "manifest_hash", "label_map_hash",
                    "raw_data_hash", "folds_hash", "split_hash", "protocol", "checkpoint_type")


def parent_config_hash(cfg: dict) -> str:
    """Common configuration identity across folds, retaining every other setting."""
    parent = copy.deepcopy(cfg)
    parent["dataset"].pop("cv_fold", None)
    return config_hash(parent)


def _hash(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"Missing or unknown {name}")
    return value


def _schema(value, name):
    if type(value) is not int or value != 1:
        raise ValueError(f"Unsupported {name} schema_version: {value!r}")


def _manifest_records(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _scalar(value, name):
    import numpy as np
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Prediction {name} must contain one scalar")
    return array.reshape(-1)[0].item()


def _integer_scalar(value, name):
    result = _scalar(value, name)
    if type(result) is not int:
        raise ValueError(f"Prediction {name} must be an integer scalar")
    return result


def verify_run(path: Path, expected: dict, expected_config: dict) -> dict:
    """Verify completion, identities, exact valid coverage and metric recomputation."""
    import numpy as np
    from src.evaluation.metrics import validate_predictions

    status = _json(path / "status.json")
    if status.get("status") != "complete":
        raise ValueError(f"Run status is {status.get('status')!r}, not complete")
    inventory = status.get("artifact_hashes", {})
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("Complete status has no artifact hash inventory")
    for relative, digest in inventory.items():
        artifact_path = path / relative
        try:
            artifact_path.resolve().relative_to(path.resolve())
        except ValueError as exc:
            raise ValueError("Artifact inventory path escapes run directory") from exc
        if not artifact_path.is_file() or file_hash(artifact_path) != digest:
            raise ValueError(f"Complete artifact hash mismatch: {relative}")
    cfg = load_config(path / "resolved_config.yaml")
    protocol = cfg["dataset"]["split_protocol"]
    is_cv = protocol == "dev_cv5"
    if protocol not in ("official_holdout", "dev_cv5") or expected.get("protocol") != protocol:
        raise ValueError("Run protocol disagrees with the suite plan")
    fold = cfg["dataset"].get("cv_fold") if is_cv else -1
    if is_cv and (type(fold) is not int or fold not in CV_FOLDS or type(expected.get("fold")) is not int or expected.get("fold") != fold):
        raise ValueError("CV fold disagrees with the suite plan")
    actual_cfg_hash = config_hash(cfg)
    if actual_cfg_hash != expected["config_hash"] or actual_cfg_hash != config_hash(expected_config):
        raise ValueError("Resolved configuration disagrees with the preregistered suite plan")
    if cfg["experiment"]["id"] != expected["spec_id"]:
        raise ValueError("Experiment ID disagrees with the suite plan")
    provenance = _json(path / "provenance.json")
    for field in ("identity_hash", "config_hash", "source_hash", "manifest_hash", "label_map_hash"):
        _hash(provenance.get(field), f"provenance.{field}")
    if provenance["config_hash"] != actual_cfg_hash or provenance.get("seed") != expected["seed"]:
        raise ValueError("Provenance configuration/seed mismatch")
    if is_cv:
        if type(expected["seed"]) is not int or type(provenance.get("seed")) is not int:
            raise ValueError("CV seed must be an integer")
        for field, value in (("seed", expected["seed"]), ("fold", fold), ("cv_fold", fold), ("protocol", protocol), ("identity_hash", provenance["identity_hash"])):
            if field in status and status[field] != value:
                raise ValueError(f"Complete CV status {field} mismatch")
    source = provenance.get("source", {})
    if object_hash(source.get("files", {})) != provenance["source_hash"] or source.get("sha256") != provenance["source_hash"]:
        raise ValueError("Source hash does not match the saved file hash inventory")
    if expected.get("identity") is not None:
        for field, value in expected["identity"].items():
            if provenance.get(field) != value:
                raise ValueError(f"Provenance disagrees with the suite identity: {field}")
    manifest = resolve_path(cfg["dataset"]["manifest"])
    label_map = resolve_path(cfg["dataset"]["label_map"])
    if file_hash(manifest) != provenance["manifest_hash"] or file_hash(label_map) != provenance["label_map_hash"]:
        raise ValueError("Current data manifest/label map differs from the completed run")
    records = _manifest_records(manifest)
    folds_hash, split_hash = None, None
    if is_cv:
        from src.data.splits import select_config_partition
        expected_rows = select_config_partition(records, cfg, "valid")
        expected_train = select_config_partition(records, cfg, "train")
        training_ids = provenance.get("training_sample_ids")
        if not isinstance(training_ids, list) or len(set(training_ids)) != len(training_ids) or set(training_ids) != {row["sample_id"] for row in expected_train}:
            raise ValueError("CV training sample IDs differ from the exact outer training fold")
        if type(provenance.get("cv_fold")) is not int or type(provenance.get("fold")) is not int or provenance.get("cv_fold") != fold or provenance.get("fold") != fold or provenance.get("protocol") != protocol:
            raise ValueError("CV provenance fold/protocol mismatch")
        folds_hash = _hash(provenance.get("folds_hash"), "provenance.folds_hash")
        split_hash = _hash(provenance.get("split_hash"), "provenance.split_hash")
        if file_hash(resolve_path(cfg["dataset"]["folds_file"])) != folds_hash:
            raise ValueError("Current folds file differs from the completed run")
        expected_split_hash = object_hash({"protocol": "dev_cv5", "folds_hash": folds_hash, "split_seed": cfg["dataset"]["split_seed"]})
        if split_hash != expected_split_hash:
            raise ValueError("CV split_hash does not match protocol, folds file and split seed")
        raw_files = provenance.get("raw_data_files")
        if not isinstance(raw_files, dict) or set(raw_files) != {"meta.csv", "train_indexes.csv", "valid_indexes.csv", "test_indexes.csv"}:
            raise ValueError("CV raw-data source hash inventory is missing or incomplete")
        if object_hash(raw_files) != _hash(provenance.get("raw_data_hash"), "provenance.raw_data_hash"):
            raise ValueError("CV raw-data hash inventory mismatch")
        data_root = resolve_path(cfg["dataset"]["root"])
        if any(file_hash(data_root / "meta" / name) != digest for name, digest in raw_files.items()):
            raise ValueError("Current raw metadata/indexes differ from the completed CV run")
        identity_components = {field: provenance[field] for field in ("config_hash", "source_hash", "manifest_hash", "label_map_hash", "raw_data_hash", "seed", "folds_hash", "split_hash", "cv_fold")}
        if object_hash(identity_components) != provenance["identity_hash"]:
            raise ValueError("CV identity_hash does not match its source/config/data/fold components")
    else:
        expected_rows = [row for row in records if row["official_split"] == "valid" and not row.get("exclusion_reason")]
    expected_ids = [row["sample_id"] for row in expected_rows]
    if not expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Manifest has empty or duplicate valid sample IDs")
    checkpoint_type = cfg["evaluation"]["checkpoint_type"]
    stem = f"valid_{checkpoint_type}_argmax_notta"
    regular = path / "checkpoints/final_regular.pt"
    ema = path / "checkpoints/final_ema.pt"
    if not regular.is_file() or regular.stat().st_size == 0:
        raise ValueError("Missing final regular checkpoint")
    if cfg["train"]["ema"]["enabled"] and (not ema.is_file() or ema.stat().st_size == 0):
        raise ValueError("Missing final EMA checkpoint")
    selected_checkpoint = ema if checkpoint_type == "ema" else regular
    prediction = path / "predictions" / f"{stem}.npz"
    required_inventory = {"checkpoints/final_regular.pt", f"predictions/{stem}.npz", f"metrics/{stem}.json"}
    if cfg["train"]["ema"]["enabled"]:
        required_inventory.add("checkpoints/final_ema.pt")
    if required_inventory - set(inventory):
        raise ValueError("Artifact inventory omits required checkpoint/prediction/metrics files")
    with np.load(prediction, allow_pickle=False) as artifact:
        required = set(PREDICTION_FIELDS)
        if is_cv:
            required.update(("schema_version", "lesion_id", "source_row_index", "checkpoint_type", "tta", "decision_rule", "protocol", *CV_IDENTITY_FIELDS))
        missing = required - set(artifact.files)
        if missing:
            raise ValueError(f"Prediction fields missing: {sorted(missing)}")
        if "schema_version" in artifact:
            _schema(_scalar(artifact["schema_version"], "schema_version"), "prediction")
        if is_cv:
            allowed = required
            if set(artifact.files) - allowed:
                raise ValueError(f"Unknown CV prediction fields: {sorted(set(artifact.files) - allowed)}")
            for field in CV_IDENTITY_FIELDS:
                if _scalar(artifact[field], field) != provenance[field]:
                    raise ValueError(f"CV prediction {field} differs from provenance")
            for field in ("seed", "fold", "cv_fold"):
                _integer_scalar(artifact[field], field)
            if type(_scalar(artifact["tta"], "tta")) is not bool:
                raise ValueError("Prediction tta must be a boolean scalar")
            if np.asarray(artifact["sample_id"]).shape != (len(expected_ids),) or np.asarray(artifact["probabilities"]).shape != (len(expected_ids), 5):
                raise ValueError("CV sample IDs/probabilities have incorrect row shape")
        computed = validate_predictions(artifact, expected_ids=expected_ids)
        if _scalar(artifact["seed"], "seed") != expected["seed"]:
            raise ValueError("Prediction seed mismatch")
        eval_partition = "dev_cv_valid" if is_cv else "official_valid"
        if _scalar(artifact["eval_partition"], "eval_partition") != eval_partition:
            raise ValueError("Collector only accepts valid development predictions")
        if _scalar(artifact["checkpoint_hash"], "checkpoint_hash") != file_hash(selected_checkpoint):
            raise ValueError("Prediction checkpoint hash mismatch")
        declared_fold = _scalar(artifact["fold"], "fold")
        if (is_cv and declared_fold != fold) or (not is_cv and declared_fold not in (-1, None)):
            raise ValueError("Prediction fold mismatch")
        for field, value in (("checkpoint_type", checkpoint_type), ("tta", False), ("decision_rule", "argmax"), ("protocol", protocol)):
            if field in artifact and _scalar(artifact[field], field) != value:
                raise ValueError(f"Prediction {field} mismatch")
        diagnoses = {row["sample_id"]: int(row["diagnosis_index"]) for row in expected_rows}
        if any(int(label) != diagnoses[str(sample_id)] for sample_id, label in zip(artifact["sample_id"], artifact["y_true"])):
            raise ValueError("Prediction labels differ from the audited valid manifest")
        if is_cv:
            expected_by_id = {row["sample_id"]: row for row in expected_rows}
            for field in ("lesion_id", "source_row_index"):
                values = np.asarray(artifact[field])
                if values.shape != (len(expected_ids),) or (field == "source_row_index" and not np.issubdtype(values.dtype, np.integer)):
                    raise ValueError(f"Prediction {field} must align with sample rows")
                if any(str(value) != str(expected_by_id[str(sample_id)][field]) for sample_id, value in zip(artifact["sample_id"], values)):
                    raise ValueError(f"Prediction {field} differs from the manifest")
    metrics = _json(path / "metrics" / f"{stem}.json")
    if metrics.get("class_order") != CLASS_ORDER or metrics.get("n_samples") != len(expected_ids):
        raise ValueError("Metrics class order/sample count mismatch")
    for field in METRICS:
        value, measured = metrics.get(field), computed[field]
        if measured is None:
            if value is not None:
                raise ValueError(f"{field} must be unavailable for this label support")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not math.isclose(value, measured, abs_tol=1e-10):
            raise ValueError(f"Metrics {field} differs from the verified predictions")
    if is_cv:
        for field, value in (("eval_partition", "dev_cv_valid"), ("seed", expected["seed"]), ("fold", fold), ("protocol", protocol),
                             ("checkpoint_type", checkpoint_type), ("checkpoint_hash", file_hash(selected_checkpoint)), ("tta", False), ("decision_rule", "argmax")):
            if metrics.get(field) != value:
                raise ValueError(f"CV metrics {field} mismatch")
        if "schema_version" in metrics:
            _schema(metrics["schema_version"], "metrics")
    return {"spec_id": expected["spec_id"], "parent_id": expected["parent_id"], "seed": expected["seed"],
            "fold": fold, "protocol": protocol, "eval_partition": "dev_cv_valid" if is_cv else "official_valid",
            "checkpoint_type": checkpoint_type, "status": "complete", "stage": cfg["stage"],
            "run_dir": str(path), "n_samples": len(expected_ids), "config_hash": actual_cfg_hash,
            "source_hash": provenance["source_hash"], "manifest_hash": provenance["manifest_hash"],
            "label_map_hash": provenance["label_map_hash"], "checkpoint_hash": file_hash(selected_checkpoint),
            "raw_data_hash": provenance.get("raw_data_hash"),
            "folds_hash": folds_hash, "split_hash": split_hash, "parent_config_hash": parent_config_hash(cfg),
            "prediction_path": str(prediction), "prediction_hash": file_hash(prediction), "identity_hash": provenance["identity_hash"],
            **{field: metrics[field] for field in METRICS}}


def _consistent(rows, fields):
    return bool(rows) and all(len({row[field] for row in rows}) == 1 for field in fields)


def _cv_registration(plan):
    """Registration is the full spec x seed x fold product, never observed runs."""
    seeds = plan.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("CV plan must preregister unique nonnegative integer seeds")
    configs = plan["resolved_configs"]
    if not isinstance(configs, dict) or not configs:
        raise ValueError("CV plan has no resolved configurations")
    specs = []
    expected_configs = set()
    for key, cfg in configs.items():
        spec, fold = cfg["experiment"]["id"], cfg["dataset"]["cv_fold"]
        if not isinstance(spec, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", spec):
            raise ValueError("CV experiment ID must be a simple artifact name")
        if cfg["dataset"]["split_protocol"] != "dev_cv5" or type(fold) is not int or fold not in CV_FOLDS or key != f"{spec}_fold{fold}":
            raise ValueError("CV resolved configuration key/fold/protocol mismatch")
        if cfg["experiment"]["seeds"] != seeds:
            raise ValueError("CV resolved configuration seeds differ from registration")
        if spec not in specs:
            specs.append(spec)
        expected_configs.add(key)
    if expected_configs != {f"{spec}_fold{fold}" for spec in specs for fold in CV_FOLDS}:
        raise ValueError("Each CV experiment requires all five resolved fold configurations")
    signatures = []
    for run in plan["runs"]:
        seed, fold = run.get("seed"), run.get("fold")
        spec = run.get("spec_id")
        if type(seed) is not int or type(fold) is not int or run.get("protocol") != "dev_cv5" or run.get("config_key") != f"{spec}_fold{fold}":
            raise ValueError("CV run registration requires explicit config_key, integer seed/fold and dev_cv5 protocol")
        signatures.append((spec, seed, fold))
    required = {(spec, seed, fold) for spec in specs for seed in seeds for fold in CV_FOLDS}
    if len(set(signatures)) != len(signatures):
        raise ValueError("Duplicate preregistered CV spec/seed/fold")
    if set(signatures) != required:
        raise ValueError("CV registration is missing or adds a spec/seed/fold; exactly five folds are required")
    if plan.get("run_count", len(signatures)) != len(signatures):
        raise ValueError("CV run_count disagrees with the full preregistered run set")
    return specs, sorted(seeds)


def _build_oof(suite_dir, spec, seed, rows, cfg):
    """Build sample-level OOF in source row order after exact fold verification."""
    import numpy as np
    from src.evaluation.metrics import validate_predictions

    if len(rows) != 5 or {row["fold"] for row in rows} != CV_FOLDS or any(row["spec_id"] != spec or row["seed"] != seed for row in rows):
        raise ValueError("OOF requires exactly five complete folds from one experiment and seed")
    if not _consistent(rows, CV_COMMON_FIELDS):
        raise ValueError("OOF source/config/data/protocol identity mismatch across folds")
    development = sorted((row for row in _manifest_records(resolve_path(cfg["dataset"]["manifest"]))
                          if row["official_split"] in ("train", "valid")), key=lambda row: int(row["source_row_index"]))
    expected = {row["sample_id"]: row for row in development}
    if not expected or len(expected) != len(development):
        raise ValueError("OOF development manifest must have unique cases")
    by_sample = {}
    ordered_folds = sorted(rows, key=lambda row: row["fold"])
    for row in ordered_folds:
        run = Path(row["run_dir"])
        checkpoint = run / "checkpoints" / f"final_{row['checkpoint_type']}.pt"
        if file_hash(checkpoint) != row["checkpoint_hash"]:
            raise ValueError("Checkpoint hash changed between fold verification and OOF collection")
        if file_hash(row["prediction_path"]) != row["prediction_hash"]:
            raise ValueError("Prediction hash changed between fold verification and OOF collection")
        with np.load(row["prediction_path"], allow_pickle=False) as artifact:
            if _scalar(artifact["checkpoint_hash"], "checkpoint_hash") != row["checkpoint_hash"]:
                raise ValueError("OOF prediction checkpoint hash drift")
            for index, sample in enumerate(artifact["sample_id"].astype(str)):
                if sample not in expected:
                    raise ValueError("OOF includes test or unknown cases")
                if sample in by_sample:
                    raise ValueError("A case is predicted by more than one OOF fold")
                if int(artifact["y_true"][index]) != int(expected[sample]["diagnosis_index"]):
                    raise ValueError("OOF labels differ from the development manifest")
                by_sample[sample] = (artifact["logits"][index].copy(), artifact["probabilities"][index].copy(), row["fold"], row["checkpoint_hash"])
    if set(by_sample) != set(expected):
        raise ValueError("OOF must cover every official train+valid case exactly once")
    ids = [row["sample_id"] for row in development]
    folds = np.asarray([by_sample[sample][2] for sample in ids], dtype=np.int64)
    checkpoint_hashes = np.asarray([by_sample[sample][3] for sample in ids])
    artifact = {"schema_version": np.asarray(1), "sample_id": np.asarray(ids),
                "lesion_id": np.asarray([row["lesion_id"] for row in development]),
                "source_row_index": np.asarray([int(row["source_row_index"]) for row in development], dtype=np.int64),
                "y_true": np.asarray([int(row["diagnosis_index"]) for row in development], dtype=np.int64),
                "logits": np.stack([by_sample[sample][0] for sample in ids]),
                "probabilities": np.stack([by_sample[sample][1] for sample in ids]),
                "class_order": np.asarray(CLASS_ORDER), "seed": np.asarray(seed),
                "fold": folds, "folds": folds.copy(), "eval_partition": np.asarray("dev_cv_oof"),
                "checkpoint_hash": checkpoint_hashes, "fold_checkpoint_hashes": np.asarray([row["checkpoint_hash"] for row in ordered_folds]),
                "fold_config_hashes": np.asarray([row["config_hash"] for row in ordered_folds]),
                "fold_identity_hashes": np.asarray([row["identity_hash"] for row in ordered_folds]),
                "checkpoint_type": np.asarray(rows[0]["checkpoint_type"]), "tta": np.asarray(False),
                "decision_rule": np.asarray("argmax"), "protocol": np.asarray("dev_cv5"),
                "spec_id": np.asarray(spec), "n_folds": np.asarray(5), "formal": np.asarray(False)}
    artifact.update({field: np.asarray(rows[0][field]) for field in CV_COMMON_FIELDS if field not in artifact})
    # validate_predictions recomputes pooled metrics; fold means never enter this calculation.
    metrics = validate_predictions(artifact, expected_ids=ids)
    metrics.update(schema_version=1, spec_id=spec, seed=seed, protocol="dev_cv5", eval_partition="dev_cv_oof",
                   checkpoint_type=rows[0]["checkpoint_type"], tta=False, decision_rule="argmax", formal=False,
                   n_folds=5, config_hash_definition="parent_config_hash omits only dataset.cv_fold",
                   fold_checkpoint_hashes=[row["checkpoint_hash"] for row in ordered_folds],
                   fold_config_hashes=[row["config_hash"] for row in ordered_folds],
                   fold_identity_hashes=[row["identity_hash"] for row in ordered_folds])
    metrics.update({field: rows[0][field] for field in CV_COMMON_FIELDS})
    for field in METRICS:
        values = [row[field] for row in ordered_folds]
        metrics[f"{field}_fold_mean"] = statistics.mean(values) if all(value is not None for value in values) else None
        metrics[f"{field}_fold_sample_std"] = statistics.stdev(values) if all(value is not None for value in values) else None
    stem = f"{spec}_seed{seed}_{rows[0]['checkpoint_type']}_argmax_notta"
    return artifact, metrics, suite_dir / "oof" / f"{stem}.npz", suite_dir / "oof" / f"{stem}.metrics.json"


def _check_oof_export(artifact, metrics, prediction, metrics_path):
    """Existing exports are immutable; identical re-exports preserve their bytes."""
    import numpy as np
    if prediction.exists() or metrics_path.exists():
        if not prediction.is_file() or (metrics_path.exists() and not metrics_path.is_file()):
            raise ValueError("Existing OOF predictions/metrics are incomplete; refusing partial reuse")
        with np.load(prediction, allow_pickle=False) as saved:
            if set(saved.files) != set(artifact) or any(not np.array_equal(saved[key], value) for key, value in artifact.items()):
                raise ValueError("Existing OOF predictions differ; refusing overwrite")
        # An interrupted metrics write can be repaired only after exact NPZ comparison.
        if metrics_path.exists() and _json(metrics_path) != metrics:
            raise ValueError("Existing OOF metrics differ; refusing overwrite")


def _atomic_npz(path, artifact):
    """A serialization interruption must never leave a partial final ZIP."""
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w+b", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **artifact)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _cv_collect(suite_dir, plan, rows, failures):
    specs = list(dict.fromkeys(cfg["experiment"]["id"] for cfg in plan["resolved_configs"].values()))
    registered = plan.get("seeds", [])
    seeds = sorted(registered) if isinstance(registered, list) and all(type(seed) is int for seed in registered) else []
    try:
        specs, seeds = _cv_registration(plan)
    except (ValueError, KeyError, TypeError) as exc:
        failures.append({"status": "invalid", "error": str(exc)})
    prepared = []
    if not failures:
        for spec in specs:
            complete = [row for row in rows if row["spec_id"] == spec]
            if not _consistent(complete, CV_COMMON_FIELDS):
                failures.append({"spec_id": spec, "status": "invalid", "error": "OOF identity mismatch across registered seeds/folds"})
                continue
            for seed in seeds:
                group = [row for row in complete if row["seed"] == seed]
                try:
                    cfg = plan["resolved_configs"][f"{spec}_fold0"]
                    export = _build_oof(suite_dir, spec, seed, group, cfg)
                    _check_oof_export(*export)
                    prepared.append(export)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    failures.append({"spec_id": spec, "seed": seed, "status": "invalid", "error": str(exc)})
    seed_rows = []
    if not failures:
        for artifact, metrics, prediction, metrics_path in prepared:
            prediction.parent.mkdir(parents=True, exist_ok=True)
            if not prediction.exists():
                _atomic_npz(prediction, artifact)
            if not metrics_path.exists():
                write_json(metrics_path, metrics)
            seed_rows.append({**metrics, "prediction_path": str(prediction), "metrics_path": str(metrics_path),
                              "oof_hash": file_hash(prediction), "metrics_hash": file_hash(metrics_path), "status": "complete"})
    aggregates = []
    for spec in specs:
        complete = [row for row in seed_rows if row["spec_id"] == spec]
        actual = sorted(row["seed"] for row in complete)
        all_seeds = not failures and actual == seeds
        summary = {"spec_id": spec, "eval_partition": "dev_cv_oof", "n_seeds": len(complete),
                   "seeds": json.dumps(actual), "expected_seeds": json.dumps(seeds), "complete_seed_set": all_seeds,
                   "formal": False, "reason": "development CV is not formal evidence" if all_seeds else "missing/invalid registered CV folds or identities"}
        for field in METRICS:
            values = [row[field] for row in complete]
            ready = all_seeds and all(value is not None for value in values)
            summary[f"{field}_mean"] = statistics.mean(values) if ready else None
            summary[f"{field}_sample_std"] = statistics.stdev(values) if ready and len(values) > 1 else None
        aggregates.append(summary)
    seed_fields = ["spec_id", "seed", "protocol", "eval_partition", "n_folds", "n_samples", "checkpoint_type", "formal", "status",
                   *CV_COMMON_FIELDS, *METRICS, "prediction_path", "metrics_path", "oof_hash", "metrics_hash"]
    # DictWriter requires a unique field list even when metadata appears in two groups.
    _write_csv(suite_dir / "seed_metrics.csv", seed_rows, list(dict.fromkeys(seed_fields)))
    return aggregates, len(seed_rows)


def _holdout_collect(plan, rows, failures):
    aggregates = []
    for spec_id in dict.fromkeys(run["spec_id"] for run in plan["runs"]):
        expected_seeds = sorted(run["seed"] for run in plan["runs"] if run["spec_id"] == spec_id)
        complete = [row for row in rows if row["spec_id"] == spec_id and row["status"] == "complete"]
        actual_seeds = sorted(row["seed"] for row in complete)
        all_seeds = actual_seeds == expected_seeds and len(set(actual_seeds)) == len(expected_seeds)
        identity_consistent = _consistent(complete, ("config_hash", "source_hash", "manifest_hash", "label_map_hash", "protocol", "checkpoint_type"))
        frozen = bool(complete) and all(row["stage"] == "frozen" for row in complete)
        formal = all_seeds and identity_consistent and frozen and set(expected_seeds) == FORMAL_SEEDS
        if all_seeds and not identity_consistent:
            failures.append({"spec_id": spec_id, "status": "invalid", "error": "Holdout identity/protocol mismatch across seeds"})
        reason = ("missing/invalid preregistered seeds" if not all_seeds else "identity/protocol mismatch" if not identity_consistent else
                  "development configuration is not frozen" if not frozen else "formal three-seed set is incomplete" if set(expected_seeds) != FORMAL_SEEDS else "")
        summary = {"spec_id": spec_id, "eval_partition": "official_valid", "n_seeds": len(complete), "seeds": json.dumps(actual_seeds),
                   "expected_seeds": json.dumps(expected_seeds), "complete_seed_set": all_seeds, "formal": formal, "reason": reason}
        for field in METRICS:
            values = [row[field] for row in complete]
            ready = all_seeds and identity_consistent and all(value is not None for value in values)
            summary[f"{field}_mean"] = statistics.mean(values) if ready else None
            summary[f"{field}_sample_std"] = statistics.stdev(values) if ready and len(values) > 1 else None
        aggregates.append(summary)
    return aggregates


def collect_suite(suite_dir: str | Path) -> dict:
    suite_dir = Path(suite_dir)
    plan = _json(suite_dir / "plan.json")
    rows, failures = [], []
    is_cv = any(cfg["dataset"]["split_protocol"] == "dev_cv5" for cfg in plan["resolved_configs"].values())
    for planned in plan["runs"]:
        base = {key: planned.get(key) for key in ("spec_id", "parent_id", "seed", "fold", "config_key", "protocol", "status", "run_dir")}
        if planned.get("status") != "complete" or not planned.get("run_dir"):
            base["error"] = "Run is not complete in the suite plan"
            rows.append(base)
            failures.append(base)
            continue
        try:
            key = planned.get("config_key", planned["spec_id"])
            rows.append(verify_run(Path(planned["run_dir"]), planned, plan["resolved_configs"][key]))
        except (ConfigError, OSError, ValueError, KeyError, TypeError) as exc:
            base.update(status="invalid", error=str(exc))
            rows.append(base)
            failures.append(base)
    aggregates, n_oof = _cv_collect(suite_dir, plan, rows, failures) if is_cv else (_holdout_collect(plan, rows, failures), 0)
    run_fields = ["spec_id", "parent_id", "seed", "fold", "protocol", "eval_partition", "checkpoint_type", "status", "stage",
                  "run_dir", "n_samples", "config_hash", "parent_config_hash", "source_hash", "manifest_hash", "label_map_hash",
                  "raw_data_hash", "folds_hash", "split_hash", "checkpoint_hash", *METRICS, "error"]
    aggregate_fields = ["spec_id", "eval_partition", "n_seeds", "seeds", "expected_seeds", "complete_seed_set", "formal", "reason"]
    for field in METRICS:
        aggregate_fields.extend((f"{field}_mean", f"{field}_sample_std"))
    _write_csv(suite_dir / "runs.csv", rows, run_fields)
    _write_csv(suite_dir / "aggregate.csv", aggregates, aggregate_fields)
    complete = bool(plan["runs"]) and not failures and bool(aggregates) and all(row["complete_seed_set"] for row in aggregates)
    result = {"suite_id": plan["suite_id"], "planned_runs": len(plan["runs"]),
              "verified_complete_runs": sum(row["status"] == "complete" for row in rows),
              "all_preregistered_seeds_complete": complete, "pooled_oof_exports": n_oof,
              "formal": False if is_cv else complete and all(row["formal"] for row in aggregates),
              "eval_partition": "dev_cv_oof" if is_cv else "official_valid",
              "scope": "development CV pooled OOF; official test remains sealed" if is_cv else "valid development metrics; final test evaluation is separate",
              "failures": failures}
    write_json(suite_dir / "collection_status.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-id", required=True)
    parser.add_argument("--suites-root", default="outputs/suites")
    args = parser.parse_args(argv)
    try:
        if Path(args.suite_id).name != args.suite_id or args.suite_id in (".", ".."):
            raise ValueError("suite-id must be a simple directory name")
        result = collect_suite(resolve_path(args.suites_root) / args.suite_id)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["all_preregistered_seeds_complete"] else 2
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
