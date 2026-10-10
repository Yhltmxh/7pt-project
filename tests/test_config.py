import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.config import ConfigError, apply_overrides, config_hash, deep_merge, load_config, validate_config
from scripts.run_experiments import build_plan, expand_suite
from scripts.collect_results import collect_suite

ROOT = Path(__file__).resolve().parents[1]


def _suite(tmp_path, experiments, **changes):
    value = {
        "suite_id": "test_suite", "base_config": str(ROOT / "configs/base.yaml"),
        "protocol_config": str(ROOT / "configs/protocols/official_holdout.yaml"),
        "seeds": [101, 3407, 777], "experiments": experiments,
    }
    value.update(changes)
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    return path


def test_draft_preserves_missing_decisions_and_preflight_lists_them():
    cfg = load_config("configs/base.yaml")
    assert cfg["dataset"]["root"] is None
    assert cfg["train"]["epochs"] is None
    assert cfg["dataset"]["clinical_substitution_policy"] is None
    with pytest.raises(ConfigError) as error:
        validate_config(cfg, require_ready=True)
    for field in ("dataset.root", "train.epochs", "dataset.clinical_substitution_policy"):
        assert field in str(error.value)


def test_unknown_nested_and_dotted_fields_are_rejected():
    cfg = load_config("configs/base.yaml")
    with pytest.raises(ConfigError, match="train.typo"):
        deep_merge(cfg, {"train": {"typo": 30}})
    with pytest.raises(ConfigError, match="model.typo"):
        apply_overrides(cfg, {"model.typo": True})
    assert cfg["model"]["fusion"] == "mean"


def test_duplicate_yaml_keys_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("stage: draft\nstage: frozen\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Duplicate YAML key"):
        load_config(path)


def test_config_identity_is_order_independent_and_changes_with_protocol():
    cfg = load_config("configs/base.yaml")
    reordered = dict(reversed(list(cfg.items())))
    assert config_hash(reordered) == config_hash(cfg)
    changed = apply_overrides(cfg, {"train.batch_size": 8})
    assert config_hash(changed) != config_hash(cfg)


@pytest.mark.parametrize("value", ["dev_cv", "refit", "invented"])
def test_unimplemented_protocols_are_explicitly_rejected(value):
    with pytest.raises(ConfigError, match="not implemented"):
        load_config("configs/base.yaml", {"dataset.split_protocol": value})


@pytest.mark.parametrize("overrides", [
    {"model.fusion": "spatial_cross_attention"},
    {"model.attribute_head": True},
    {"model.metadata_tokens": True},
    {"dataset.metadata_fields": ["location"]},
    {"dataset.clinical_substitution_policy": "complete_pair"},
    {"evaluation.tta": True},
    {"model.modalities": ["clinical"]},
    {"train.epochs": 5},
    {"train.batch_size": True},
])
def test_unsupported_or_inconsistent_configs_are_rejected(overrides):
    with pytest.raises(ConfigError):
        load_config("configs/base.yaml", overrides)


def test_tiny_encoder_smoke_remains_development_only():
    overrides = {"model.clinical_encoder": "tiny_cnn", "model.dermoscopic_encoder": "tiny_cnn",
                 "model.pretrained_id": "none", "dataset.root": "/tmp/synthetic",
                 "dataset.clinical_substitution_policy": "retain_official", "train.epochs": 1,
                 "train.warmup_epochs": 0}
    cfg = load_config("configs/base.yaml", overrides)
    validate_config(cfg, require_ready=True)
    cfg["stage"] = "frozen"
    with pytest.raises(ConfigError, match="development smoke"):
        validate_config(cfg)


def test_full_suite_has_fifteen_runs_and_correct_parent_fusions():
    expanded = expand_suite("configs/experiments/baselines.yaml")
    assert build_plan(expanded)["run_count"] == 15
    configs = {spec["id"]: spec["config"] for spec in expanded["experiments"]}
    assert configs["B01"]["model"]["modalities"] == ["clinical"]
    assert configs["B02"]["model"]["modalities"] == ["dermoscopic"]
    assert [configs[key]["model"]["fusion"] for key in ("B04", "B05", "B06")] == ["mean", "concat", "gated"]
    assert not expanded["test_inference"]


def test_parent_topology_and_only_selection(tmp_path):
    path = _suite(tmp_path, [
        {"id": "B05", "parent": "B04", "overrides": {"model.fusion": "concat"}},
        {"id": "B04", "parent": "base", "overrides": {"model.fusion": "mean"}},
    ])
    expanded = expand_suite(path)
    assert [spec["id"] for spec in expanded["experiments"]] == ["B04", "B05"]
    selected = expand_suite(path, only=["B05"], seeds=[101])
    assert build_plan(selected)["run_count"] == 1
    assert selected["experiments"][0]["config"]["model"]["fusion"] == "concat"


def test_cyclic_unknown_or_duplicate_suite_entries_rejected(tmp_path):
    path = _suite(tmp_path, [
        {"id": "B04", "parent": "B05", "overrides": {}},
        {"id": "B05", "parent": "B04", "overrides": {}},
    ])
    with pytest.raises(ConfigError, match="Cyclic"):
        expand_suite(path)
    path = _suite(tmp_path, [{"id": "B04", "parent": "unknown", "overrides": {}}])
    with pytest.raises(ConfigError, match="Unknown experiment parent"):
        expand_suite(path)
    path = _suite(tmp_path, [{"id": "B04", "parent": "base", "overrides": {}}] * 2)
    with pytest.raises(ConfigError, match="Duplicate"):
        expand_suite(path)


def test_experiment_specific_budget_changes_are_rejected(tmp_path):
    path = _suite(tmp_path, [{"id": "B04", "parent": "base", "overrides": {"train.epochs": 25}}])
    with pytest.raises(ConfigError, match="Unregistered differences"):
        expand_suite(path)


def test_plan_expansion_does_not_import_torch():
    result = subprocess.run([sys.executable, "-c", "from scripts.run_experiments import expand_suite; import sys; expand_suite('configs/experiments/baselines.yaml'); assert 'torch' not in sys.modules"],
                            cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_missing_seeds_cannot_produce_a_formal_or_partial_mean(tmp_path):
    expanded = expand_suite("configs/experiments/baselines.yaml", only=["B04"])
    plan = build_plan(expanded)
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    result = collect_suite(tmp_path)
    assert not result["formal"]
    assert not result["all_preregistered_seeds_complete"]
    assert result["verified_complete_runs"] == 0
    import csv
    with (tmp_path / "aggregate.csv").open(newline="", encoding="utf-8") as stream:
        aggregate = next(csv.DictReader(stream))
    assert aggregate["diagnosis_macro_f1_mean"] == ""
    assert aggregate["complete_seed_set"] == "False"


def _complete_synthetic_run(tmp_path):
    import numpy as np
    from src.config import save_config
    from src.evaluation.metrics import classification_metrics, softmax
    from src.provenance import file_hash, object_hash

    manifest = tmp_path / "samples.csv"
    manifest.write_text("sample_id,official_split,diagnosis_index,exclusion_reason\n" +
                        "".join(f"case_{i},valid,{i},\n" for i in range(5)), encoding="utf-8")
    label_map = tmp_path / "label_map.json"
    label_map.write_text(json.dumps({"class_order": ["BCC", "NEV", "MEL", "MISC", "SK"]}), encoding="utf-8")
    expanded = expand_suite("configs/experiments/baselines.yaml", only=["B04"], seeds=[101], overrides={
        "dataset.root": str(tmp_path), "dataset.manifest": str(manifest), "dataset.label_map": str(label_map),
        "dataset.clinical_substitution_policy": "retain_official", "train.epochs": 10,
    })
    plan = build_plan(expanded)
    cfg = expanded["experiments"][0]["config"]
    run = tmp_path / "run"
    save_config(cfg, run / "resolved_config.yaml")
    (run / "checkpoints").mkdir()
    (run / "predictions").mkdir()
    (run / "metrics").mkdir()
    regular, ema = run / "checkpoints/final_regular.pt", run / "checkpoints/final_ema.pt"
    regular.write_bytes(b"synthetic-checkpoint-regular")
    ema.write_bytes(b"synthetic-checkpoint-ema")
    logits = np.eye(5, dtype=np.float32)
    pred = run / "predictions/valid_ema_argmax_notta.npz"
    np.savez_compressed(pred, sample_id=np.asarray([f"case_{i}" for i in range(5)]),
                        y_true=np.arange(5), logits=logits, probabilities=softmax(logits),
                        class_order=np.asarray(cfg["evaluation"]["class_order"]), seed=101, fold=-1,
                        eval_partition="official_valid", checkpoint_hash=file_hash(ema))
    metric_path = run / "metrics/valid_ema_argmax_notta.json"
    metric_path.write_text(json.dumps(classification_metrics(logits, np.arange(5))), encoding="utf-8")
    source = {"src/train.py": "0" * 64}
    provenance = {"identity_hash": "1" * 64, "config_hash": config_hash(cfg), "source_hash": object_hash(source),
                  "source": {"files": source, "sha256": object_hash(source)},
                  "manifest_hash": file_hash(manifest), "label_map_hash": file_hash(label_map), "seed": 101}
    (run / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    status = {"status": "complete", "artifact_hashes": {
        str(path.relative_to(run)): file_hash(path) for path in (regular, ema, pred, metric_path)}}
    (run / "status.json").write_text(json.dumps(status), encoding="utf-8")
    plan["runs"][0].update(status="complete", run_dir=str(run))
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    return tmp_path, run


def test_collector_accepts_verified_development_artifacts_without_claiming_formal(tmp_path):
    suite, _ = _complete_synthetic_run(tmp_path)
    result = collect_suite(suite)
    assert result["verified_complete_runs"] == 1
    assert result["all_preregistered_seeds_complete"]
    assert not result["formal"]


def test_collector_rejects_tampered_completed_prediction(tmp_path):
    suite, run = _complete_synthetic_run(tmp_path)
    (run / "predictions/valid_ema_argmax_notta.npz").write_bytes(b"altered after completion")
    result = collect_suite(suite)
    assert result["verified_complete_runs"] == 0
    assert result["failures"][0]["status"] == "invalid"
    assert "artifact hash mismatch" in result["failures"][0]["error"]
