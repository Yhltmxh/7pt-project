"""Five-fold collection contracts using synthetic artifacts, without training."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.collect_results import CV_IDENTITY_FIELDS, METRICS, collect_suite, parent_config_hash
from src.config import CLASS_ORDER, config_hash, load_config, save_config
from src.data.splits import DEV_FOLD_FIELDS, make_dev_folds, select_config_partition
from src.evaluation.metrics import classification_metrics, softmax
from src.provenance import file_hash, make_identity, object_hash, write_json

ROOT = Path(__file__).resolve().parents[1]
STEM = "valid_ema_argmax_notta"


def _csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _plan(path):
    return json.loads((path / "plan.json").read_text(encoding="utf-8"))


def _save_plan(path, plan):
    write_json(path / "plan.json", plan)


def _refresh_inventory(run):
    path = Path(run["run_dir"])
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    status["artifact_hashes"] = {relative: file_hash(path / relative) for relative in status["artifact_hashes"]}
    write_json(path / "status.json", status)


def _rewrite_npz(run, update):
    path = Path(run["run_dir"]) / "predictions" / f"{STEM}.npz"
    with np.load(path, allow_pickle=False) as saved:
        artifact = {key: saved[key].copy() for key in saved.files}
    update(artifact)
    np.savez_compressed(path, **artifact)
    _refresh_inventory(run)


@pytest.fixture
def cv_suite(tmp_path, monkeypatch):
    """25 development cases cover five classes; five official test cases stay sealed."""
    def create(seeds=(101,)):
        suite = tmp_path / "suite"
        suite.mkdir()
        data = tmp_path / "data"
        (data / "meta").mkdir(parents=True)
        for name in ("meta.csv", "train_indexes.csv", "valid_indexes.csv", "test_indexes.csv"):
            (data / "meta" / name).write_text(f"fixture source {name}\n", encoding="utf-8")
        records = [{"sample_id": f"case_{index:03}", "lesion_id": f"lesion_{index:03}",
                    "source_row_index": index, "official_split": "train" if index < 15 else "valid" if index < 25 else "test",
                    "diagnosis_index": index % 5, "exclusion_reason": ""} for index in range(30)]
        manifest, label_map, folds = data / "samples.csv", data / "label_map.json", data / "dev_folds.csv"
        _csv(manifest, records, list(records[0]))
        write_json(label_map, {"class_order": CLASS_ORDER})
        assignments = make_dev_folds(records, seed=20261004)
        _csv(folds, assignments, DEV_FOLD_FIELDS)
        source_files = {"src/train.py": "0" * 64, "scripts/collect_results.py": "2" * 64}
        source = {"files": source_files, "sha256": object_hash(source_files)}
        monkeypatch.setattr("src.provenance.source_snapshot", lambda: copy.deepcopy(source))
        base = load_config("configs/base.yaml")
        base["stage"] = "development"
        base["experiment"].update(id="B04", suite="cvfixture", seeds=list(seeds), output_root=str(suite / "runs"))
        base["dataset"].update(root=str(data), manifest=str(manifest), label_map=str(label_map),
                               split_protocol="dev_cv5", folds_file=str(folds), clinical_substitution_policy="retain_official")
        base["train"].update(epochs=2, warmup_epochs=1, device="cpu")
        plan = {"suite_id": "cvfixture", "seeds": list(seeds), "runs": [], "resolved_configs": {}, "run_count": 5 * len(seeds)}
        for fold in range(5):
            cfg = copy.deepcopy(base)
            cfg["dataset"]["cv_fold"] = fold
            key = f"B04_fold{fold}"
            plan["resolved_configs"][key] = cfg
            valid = list(reversed(select_config_partition(records, cfg, "valid")))
            train = select_config_partition(records, cfg, "train")
            for seed_index, seed in enumerate(seeds):
                run_dir = suite / "runs" / f"B04_seed{seed}_fold{fold}"
                save_config(cfg, run_dir / "resolved_config.yaml")
                for folder in ("checkpoints", "predictions", "metrics"):
                    (run_dir / folder).mkdir()
                regular = run_dir / "checkpoints/final_regular.pt"
                checkpoint = run_dir / "checkpoints/final_ema.pt"
                regular.write_bytes(f"synthetic regular seed={seed} fold={fold}".encode())
                checkpoint.write_bytes(f"synthetic EMA seed={seed} fold={fold}".encode())
                identity = make_identity(cfg, seed, manifest, label_map)
                provenance = {**identity, "fold": fold, "protocol": "dev_cv5",
                              "training_sample_ids": [row["sample_id"] for row in train]}
                write_json(run_dir / "provenance.json", provenance)
                labels = np.asarray([row["diagnosis_index"] for row in valid])
                guesses = np.zeros(len(labels), dtype=int) if fold <= seed_index else labels
                logits = np.eye(5, dtype=np.float32)[guesses] * 3
                artifact = {"schema_version": np.asarray(1), "sample_id": np.asarray([row["sample_id"] for row in valid]),
                            "lesion_id": np.asarray([row["lesion_id"] for row in valid]),
                            "source_row_index": np.asarray([row["source_row_index"] for row in valid]),
                            "y_true": labels, "logits": logits, "probabilities": softmax(logits),
                            "class_order": np.asarray(CLASS_ORDER), "seed": np.asarray(seed), "fold": np.asarray(fold),
                            "eval_partition": np.asarray("dev_cv_valid"), "checkpoint_hash": np.asarray(file_hash(checkpoint)),
                            "checkpoint_type": np.asarray("ema"), "tta": np.asarray(False), "decision_rule": np.asarray("argmax"),
                            "protocol": np.asarray("dev_cv5"),
                            **{field: np.asarray(identity[field]) for field in CV_IDENTITY_FIELDS}}
                prediction = run_dir / "predictions" / f"{STEM}.npz"
                np.savez_compressed(prediction, **artifact)
                metrics = classification_metrics(logits, labels)
                metrics.update(eval_partition="dev_cv_valid", seed=seed, fold=fold, protocol="dev_cv5", checkpoint_type="ema",
                               checkpoint_hash=file_hash(checkpoint), tta=False, decision_rule="argmax")
                metric_path = run_dir / "metrics" / f"{STEM}.json"
                write_json(metric_path, metrics)
                inventory = {path.relative_to(run_dir).as_posix(): file_hash(path) for path in (regular, checkpoint, prediction, metric_path)}
                write_json(run_dir / "status.json", {"status": "complete", "artifact_hashes": inventory})
                run = {"spec_id": "B04", "parent_id": "base", "seed": seed, "fold": fold, "config_key": key,
                       "protocol": "dev_cv5", "config_hash": config_hash(cfg), "status": "complete", "run_dir": str(run_dir),
                       "identity": {field: value for field, value in identity.items() if field not in ("source", "raw_data_files")}}
                plan["runs"].append(run)
        _save_plan(suite, plan)
        return suite, records, assignments
    return create


def _reidentity(plan, run, mutate):
    """Change one saved run consistently, leaving cross-fold identity to reject it."""
    path = Path(run["run_dir"])
    cfg = copy.deepcopy(plan["resolved_configs"][run["config_key"]])
    old = json.loads((path / "provenance.json").read_text(encoding="utf-8"))
    mutate(cfg, old)
    save_config(cfg, path / "resolved_config.yaml")
    plan["resolved_configs"][run["config_key"]] = cfg
    components = {field: old[field] for field in ("config_hash", "source_hash", "manifest_hash", "label_map_hash", "raw_data_hash", "seed", "folds_hash", "split_hash", "cv_fold")}
    components["config_hash"] = config_hash(cfg)
    old.update(components, identity_hash=object_hash(components))
    run.update(config_hash=components["config_hash"], identity={field: old[field] for field in (*components, "identity_hash")})
    write_json(path / "provenance.json", old)
    _rewrite_npz(run, lambda artifact: artifact.update({field: np.asarray(old[field]) for field in CV_IDENTITY_FIELDS}))


def test_cv_oof_uses_sample_pool_not_fold_mean_and_preserves_case_provenance(cv_suite):
    suite, records, assignments = cv_suite()
    result = collect_suite(suite)
    assert result["all_preregistered_seeds_complete"] and result["verified_complete_runs"] == 5
    assert result["pooled_oof_exports"] == 1 and result["eval_partition"] == "dev_cv_oof" and not result["formal"]
    path = suite / "oof/B04_seed101_ema_argmax_notta.npz"
    plan = _plan(suite)
    with np.load(path, allow_pickle=False) as oof:
        assert oof["schema_version"].item() == 1 and oof["eval_partition"].item() == "dev_cv_oof"
        assert oof["sample_id"].tolist() == [row["sample_id"] for row in records[:25]]
        assert oof["source_row_index"].tolist() == list(range(25))
        assert oof["fold"].tolist() == [row["dev_fold"] for row in assignments]
        assert np.array_equal(oof["fold"], oof["folds"])
        assert not oof["formal"].item() and len(oof["fold_config_hashes"]) == 5
        assert len(set(oof["fold_config_hashes"].tolist())) == 5
        for index, fold in enumerate(oof["fold"]):
            run = next(run for run in plan["runs"] if run["fold"] == fold)
            assert oof["checkpoint_hash"][index] == file_hash(Path(run["run_dir"]) / "checkpoints/final_ema.pt")
        pooled = classification_metrics(oof["logits"], oof["y_true"])
    metrics = json.loads(path.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    assert metrics["diagnosis_macro_f1"] == pooled["diagnosis_macro_f1"]
    assert metrics["diagnosis_macro_f1"] != pytest.approx(metrics["diagnosis_macro_f1_fold_mean"])
    seed_rows, aggregate, fold_rows = _read_csv(suite / "seed_metrics.csv"), _read_csv(suite / "aggregate.csv"), _read_csv(suite / "runs.csv")
    assert len(seed_rows) == 1 and len(fold_rows) == 5 and aggregate[0]["formal"] == "False"
    assert float(aggregate[0]["diagnosis_macro_f1_mean"]) == pooled["diagnosis_macro_f1"]
    assert aggregate[0]["diagnosis_macro_f1_sample_std"] == ""
    assert len({row["parent_config_hash"] for row in fold_rows}) == 1


def test_cv_aggregate_uses_pooled_metric_per_seed_and_sample_std(cv_suite):
    suite, _, _ = cv_suite(seeds=(101, 3407))
    assert collect_suite(suite)["all_preregistered_seeds_complete"]
    seed_rows = _read_csv(suite / "seed_metrics.csv")
    aggregate = _read_csv(suite / "aggregate.csv")[0]
    for field in METRICS:
        values = [float(row[field]) for row in seed_rows]
        assert float(aggregate[f"{field}_mean"]) == pytest.approx(np.mean(values))
        assert float(aggregate[f"{field}_sample_std"]) == pytest.approx(np.std(values, ddof=1))


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unregistered", "incomplete", "missing_config", "missing_seed", "count"])
def test_cv_partial_or_bad_registration_never_exports_pooled_predictions(cv_suite, mutation):
    suite, _, _ = cv_suite()
    plan = _plan(suite)
    if mutation == "missing":
        plan["runs"].pop()
    elif mutation == "duplicate":
        plan["runs"].append(copy.deepcopy(plan["runs"][0]))
    elif mutation == "unregistered":
        plan["runs"][0]["seed"] = 999
    elif mutation == "incomplete":
        plan["runs"][0]["status"] = "failed"
    elif mutation == "missing_config":
        plan["resolved_configs"].pop("B04_fold4")
    elif mutation == "missing_seed":
        plan["seeds"] = []
    else:
        plan["run_count"] = 4
    _save_plan(suite, plan)
    result = collect_suite(suite)
    assert not result["all_preregistered_seeds_complete"] and not result["formal"]
    assert result["pooled_oof_exports"] == 0 and not (suite / "oof").exists()
    assert _read_csv(suite / "seed_metrics.csv") == []
    assert _read_csv(suite / "aggregate.csv")[0]["diagnosis_macro_f1_mean"] == ""


@pytest.mark.parametrize("mutation", ["test_case", "repeated_case", "wrong_label", "wrong_fold", "schema", "unknown", "identity", "source_row", "float_seed", "float_fold", "probability_shape", "missing_protocol"])
def test_cv_rejects_wrong_sample_or_prediction_contract(cv_suite, mutation):
    suite, _, _ = cv_suite()
    run = _plan(suite)["runs"][0]
    def mutate(artifact):
        if mutation == "test_case":
            artifact["sample_id"][0] = "case_025"
        elif mutation == "repeated_case":
            artifact["sample_id"][0] = artifact["sample_id"][1]
        elif mutation == "wrong_label":
            artifact["y_true"][0] = (artifact["y_true"][0] + 1) % 5
        elif mutation == "wrong_fold":
            artifact["fold"] = np.asarray(1)
        elif mutation == "schema":
            artifact["schema_version"] = np.asarray(2)
        elif mutation == "unknown":
            artifact["made_up_field"] = np.asarray(1)
        elif mutation == "identity":
            artifact["raw_data_hash"] = np.asarray("f" * 64)
        elif mutation == "source_row":
            artifact["source_row_index"][0] = 999
        elif mutation == "float_seed":
            artifact["seed"] = np.asarray(101.0)
        elif mutation == "float_fold":
            artifact["fold"] = np.asarray(0.0)
        elif mutation == "probability_shape":
            artifact["probabilities"] = np.full((len(artifact["y_true"]), 1), 0.2)
        else:
            artifact.pop("protocol")
    _rewrite_npz(run, mutate)
    result = collect_suite(suite)
    assert not result["all_preregistered_seeds_complete"] and result["verified_complete_runs"] == 4
    assert not (suite / "oof").exists()


def test_cv_training_ids_must_be_exact_training_fold(cv_suite):
    suite, _, _ = cv_suite()
    run = _plan(suite)["runs"][0]
    path = Path(run["run_dir"]) / "provenance.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["training_sample_ids"][0] = "case_025"
    write_json(path, saved)
    result = collect_suite(suite)
    assert not result["all_preregistered_seeds_complete"]
    assert "exact outer training fold" in result["failures"][0]["error"]


@pytest.mark.parametrize("kind", ["config", "source"])
def test_cv_cross_fold_parent_config_and_source_identity_must_match(cv_suite, kind):
    suite, _, _ = cv_suite()
    plan = _plan(suite)
    def mutate(cfg, provenance):
        if kind == "config":
            cfg["train"]["batch_size"] += 1
        else:
            provenance["source"]["files"]["src/train.py"] = "f" * 64
            provenance["source_hash"] = object_hash(provenance["source"]["files"])
            provenance["source"]["sha256"] = provenance["source_hash"]
    _reidentity(plan, plan["runs"][0], mutate)
    _save_plan(suite, plan)
    result = collect_suite(suite)
    assert result["verified_complete_runs"] == 5  # Each fold verifies independently.
    assert not result["all_preregistered_seeds_complete"] and result["pooled_oof_exports"] == 0
    assert "identity mismatch" in result["failures"][0]["error"]


@pytest.mark.parametrize("name", ["samples.csv", "label_map.json", "dev_folds.csv", "meta/meta.csv"])
def test_cv_current_data_or_split_hash_drift_is_rejected(cv_suite, name):
    suite, _, _ = cv_suite()
    root = Path(_plan(suite)["resolved_configs"]["B04_fold0"]["dataset"]["root"])
    with (root / name).open("a", encoding="utf-8") as stream:
        stream.write("\n")
    result = collect_suite(suite)
    assert not result["all_preregistered_seeds_complete"] and result["verified_complete_runs"] == 0
    assert not (suite / "oof").exists()


def test_cv_checkpoint_hash_cannot_be_replaced_even_with_refreshed_inventory(cv_suite):
    suite, _, _ = cv_suite()
    run = _plan(suite)["runs"][0]
    (Path(run["run_dir"]) / "checkpoints/final_ema.pt").write_bytes(b"new checkpoint")
    _refresh_inventory(run)
    result = collect_suite(suite)
    assert not result["all_preregistered_seeds_complete"]
    assert "checkpoint hash mismatch" in result["failures"][0]["error"]


def test_cv_reexport_is_identical_and_changed_oof_is_never_overwritten(cv_suite):
    suite, _, _ = cv_suite()
    assert collect_suite(suite)["all_preregistered_seeds_complete"]
    prediction = suite / "oof/B04_seed101_ema_argmax_notta.npz"
    metrics = prediction.with_suffix(".metrics.json")
    previous = (prediction.read_bytes(), metrics.read_bytes())
    assert collect_suite(suite)["all_preregistered_seeds_complete"]
    assert previous == (prediction.read_bytes(), metrics.read_bytes())
    with np.load(prediction, allow_pickle=False) as saved:
        artifact = {key: saved[key].copy() for key in saved.files}
    artifact["fold"][0] = (artifact["fold"][0] + 1) % 5
    np.savez_compressed(prediction, **artifact)
    altered = prediction.read_bytes()
    result = collect_suite(suite)
    assert not result["all_preregistered_seeds_complete"] and result["pooled_oof_exports"] == 0
    assert "refusing overwrite" in result["failures"][0]["error"]
    assert prediction.read_bytes() == altered and metrics.read_bytes() == previous[1]


def test_cv_one_incomplete_seed_blocks_pooled_exports_for_every_registered_seed(cv_suite):
    suite, _, _ = cv_suite(seeds=(101, 3407))
    plan = _plan(suite)
    next(run for run in plan["runs"] if run["seed"] == 3407 and run["fold"] == 4)["status"] = "failed"
    _save_plan(suite, plan)
    result = collect_suite(suite)
    assert result["verified_complete_runs"] == 9 and result["pooled_oof_exports"] == 0
    assert not result["all_preregistered_seeds_complete"] and not (suite / "oof").exists()


def test_cv_rejects_unknown_oof_schema_without_overwriting_existing_artifact(cv_suite):
    suite, _, _ = cv_suite()
    assert collect_suite(suite)["all_preregistered_seeds_complete"]
    prediction = suite / "oof/B04_seed101_ema_argmax_notta.npz"
    with np.load(prediction, allow_pickle=False) as saved:
        artifact = {key: saved[key].copy() for key in saved.files}
    artifact["schema_version"] = np.asarray(2)
    np.savez_compressed(prediction, **artifact)
    before = prediction.read_bytes()
    result = collect_suite(suite)
    assert not result["all_preregistered_seeds_complete"] and prediction.read_bytes() == before


def test_cv_interrupted_npz_write_leaves_no_partial_final_or_temporary_file(cv_suite, monkeypatch):
    suite, _, _ = cv_suite()
    prediction = suite / "oof/B04_seed101_ema_argmax_notta.npz"
    def interrupted(stream, **artifact):
        stream.write(b"partial ZIP written before interruption")
        stream.flush()
        raise KeyboardInterrupt("Injected interruption")
    with monkeypatch.context() as patch:
        patch.setattr(np, "savez_compressed", interrupted)
        with pytest.raises(KeyboardInterrupt, match="Injected interruption"):
            collect_suite(suite)
    assert not prediction.exists()
    assert list(prediction.parent.iterdir()) == []
    assert collect_suite(suite)["all_preregistered_seeds_complete"]
    assert prediction.is_file() and not list(prediction.parent.glob("*.tmp"))


def test_cv_missing_metrics_can_be_repaired_after_exact_npz_comparison(cv_suite):
    suite, _, _ = cv_suite()
    assert collect_suite(suite)["all_preregistered_seeds_complete"]
    prediction = suite / "oof/B04_seed101_ema_argmax_notta.npz"
    metrics = prediction.with_suffix(".metrics.json")
    before_npz, before_metrics = prediction.read_bytes(), metrics.read_bytes()
    metrics.unlink()
    assert collect_suite(suite)["all_preregistered_seeds_complete"]
    assert prediction.read_bytes() == before_npz and metrics.read_bytes() == before_metrics


def test_cv_missing_metrics_does_not_allow_changed_npz_reuse(cv_suite):
    suite, _, _ = cv_suite()
    assert collect_suite(suite)["all_preregistered_seeds_complete"]
    prediction = suite / "oof/B04_seed101_ema_argmax_notta.npz"
    metrics = prediction.with_suffix(".metrics.json")
    with np.load(prediction, allow_pickle=False) as saved:
        artifact = {key: saved[key].copy() for key in saved.files}
    artifact["fold"][0] = (artifact["fold"][0] + 1) % 5
    np.savez_compressed(prediction, **artifact)
    metrics.unlink()
    changed = prediction.read_bytes()
    result = collect_suite(suite)
    assert not result["all_preregistered_seeds_complete"] and not metrics.exists()
    assert prediction.read_bytes() == changed


def test_parent_config_identity_removes_only_the_current_fold():
    cfg = load_config("configs/base.yaml")
    cfg["dataset"].update(split_protocol="dev_cv5", cv_fold=0)
    cfg2 = copy.deepcopy(cfg)
    cfg2["dataset"]["cv_fold"] = 4
    assert parent_config_hash(cfg) == parent_config_hash(cfg2)
    cfg2["dataset"]["split_seed"] += 1
    assert parent_config_hash(cfg) != parent_config_hash(cfg2)


def test_cv_collector_import_does_not_load_torch():
    result = subprocess.run([sys.executable, "-c", "import scripts.collect_results; import sys; assert 'torch' not in sys.modules"],
                            cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
