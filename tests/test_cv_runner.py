"""Development five-fold planning contracts; no model is trained in these tests."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.run_experiments as runner
import src.provenance as provenance_tools
from src.config import ConfigError, config_hash, load_config, save_config, validate_config
from src.provenance import object_hash

SUITE = "configs/experiments/dev_cv_seed42.yaml"
ROOT = Path(__file__).resolve().parents[1]


def test_default_cv_plan_is_three_baselines_one_seed_five_folds():
    expanded = runner.expand_suite(SUITE)
    plan = runner.build_plan(expanded)
    assert plan["run_count"] == 15
    assert plan["seeds"] == [42]
    assert plan["evaluation_partition"] == "dev_cv_oof"
    assert not plan["test_inference"]
    assert set(plan["resolved_configs"]) == {f"{spec}_fold{fold}" for spec in ("B01", "B02", "B04") for fold in range(5)}
    assert {(run["spec_id"], run["seed"], run["fold"]) for run in plan["runs"]} == {
        (spec, 42, fold) for spec in ("B01", "B02", "B04") for fold in range(5)}
    for run in plan["runs"]:
        cfg = plan["resolved_configs"][run["config_key"]]
        assert cfg["dataset"]["cv_fold"] == run["fold"]
        assert cfg["dataset"]["split_seed"] == 20261004
        assert cfg["stage"] == "development"
        assert cfg["train"]["epochs"] == 25
        assert cfg["train"]["batch_size"] == 1
        assert cfg["train"]["gradient_accumulation"] == 16
        assert f"_seed42_fold{run['fold']}_" in run["provisional_output_dir"]
    assert len({run["config_hash"] for run in plan["runs"]}) == 15


def test_selected_baseline_seed_overrides_preserve_fixed_split_seed():
    expanded = runner.expand_suite(SUITE, only=["B04"], seeds=[101, 777])
    plan = runner.build_plan(expanded)
    assert plan["run_count"] == 10
    assert {run["seed"] for run in plan["runs"]} == {101, 777}
    assert len(plan["resolved_configs"]) == 5
    assert {cfg["dataset"]["split_seed"] for cfg in plan["resolved_configs"].values()} == {20261004}
    assert {cfg["model"]["fusion"] for cfg in plan["resolved_configs"].values()} == {"mean"}


def test_cv_template_is_dry_ready_but_requires_folds_before_execution():
    cfg = load_config("configs/protocols/dev_cv5.yaml")
    assert cfg["dataset"]["cv_fold"] is None
    assert cfg["dataset"]["folds_file"] is None
    with pytest.raises(ConfigError) as error:
        validate_config(cfg, require_ready=True)
    assert "dataset.root" in str(error.value)
    assert "dataset.folds_file" in str(error.value)
    ready = load_config("configs/protocols/dev_cv5.yaml", {"dataset.root": "/tmp/synthetic",
        "dataset.folds_file": "/tmp/synthetic/folds.csv", "dataset.clinical_substitution_policy": "retain_official"})
    # The trainer checks the concrete fold; the suite template may remain unexpanded.
    validate_config(ready, require_ready=True)


@pytest.mark.parametrize("stage", ["draft", "frozen"])
def test_cv_cannot_be_declared_frozen_final_evidence(stage):
    with pytest.raises(ConfigError, match="requires stage=development"):
        load_config("configs/protocols/dev_cv5.yaml", {"stage": stage})


@pytest.mark.parametrize("fold", [-1, 5, True, "2"])
def test_invalid_cv_fold_is_rejected(fold):
    with pytest.raises(ConfigError):
        load_config("configs/protocols/dev_cv5.yaml", {"dataset.cv_fold": fold})


def test_holdout_fold_must_remain_null():
    with pytest.raises(ConfigError, match="null for official_holdout"):
        load_config("configs/base.yaml", {"dataset.cv_fold": 0})


def test_suite_rejects_an_explicit_single_fold_override():
    with pytest.raises(ConfigError, match="expands all five folds"):
        runner.expand_suite(SUITE, overrides={"dataset.cv_fold": 2})


def test_cv_plan_expansion_does_not_import_torch():
    result = subprocess.run([sys.executable, "-c", "from scripts.run_experiments import expand_suite,build_plan; import sys; p=build_plan(expand_suite('configs/experiments/dev_cv_seed42.yaml')); assert p['run_count']==15; assert 'torch' not in sys.modules"],
        cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_old_holdout_plan_without_fold_or_config_key_remains_comparable():
    modern = runner.build_plan(runner.expand_suite("configs/experiments/baselines.yaml", only=["B04"], seeds=[101]))
    old = copy.deepcopy(modern)
    for run in old["runs"]:
        run.pop("fold")
        run.pop("config_key")
    runner._check_same_plan(old, modern)


@pytest.fixture
def cv_action_fixture(tmp_path, monkeypatch):
    root = tmp_path / "data"
    (root / "meta").mkdir(parents=True)
    for name in ("meta.csv", "train_indexes.csv", "valid_indexes.csv", "test_indexes.csv"):
        (root / "meta" / name).write_text("synthetic raw metadata\n", encoding="utf-8")
    manifest = tmp_path / "samples.csv"
    manifest.write_text("synthetic manifest", encoding="utf-8")
    labels = tmp_path / "label_map.json"
    labels.write_text("{}", encoding="utf-8")
    folds = tmp_path / "folds.csv"
    folds.write_text("synthetic folds", encoding="utf-8")
    files = {"fixture_source.py": "a" * 64}
    monkeypatch.setattr(provenance_tools, "source_snapshot", lambda: {"files": files, "sha256": object_hash(files)})
    expanded = runner.expand_suite(SUITE, only=["B04"], overrides={
        "dataset.root": str(root), "dataset.manifest": str(manifest), "dataset.label_map": str(labels),
        "dataset.folds_file": str(folds), "dataset.clinical_substitution_policy": "retain_official",
        "experiment.output_root": str(tmp_path / "runs")})
    plan = runner.build_plan(expanded)
    runner._prepare_actions(plan, plan["resolved_configs"], allow_resume=False)
    return plan, folds


def test_fold_paths_and_provenance_bind_the_same_fixed_partition_file(cv_action_fixture):
    plan, _ = cv_action_fixture
    runs = plan["runs"]
    assert len({run["run_dir"] for run in runs}) == 5
    assert len({run["identity"]["identity_hash"] for run in runs}) == 5
    assert len({run["identity"]["folds_hash"] for run in runs}) == 1
    assert len({run["identity"]["split_hash"] for run in runs}) == 1
    assert {run["identity"]["cv_fold"] for run in runs} == set(range(5))


def test_resume_refuses_fold_and_configuration_key_drift(cv_action_fixture):
    plan, _ = cv_action_fixture
    changed = copy.deepcopy(plan)
    changed["runs"][0]["fold"] = 1
    with pytest.raises(ConfigError, match="experiment/seed/protocol drift"):
        runner._check_same_plan(changed, plan)
    changed = copy.deepcopy(plan)
    changed["runs"][0]["config_key"] = "B04_fold1"
    with pytest.raises(ConfigError, match="experiment/seed/protocol drift"):
        runner._check_same_plan(changed, plan)
    with pytest.raises(ConfigError, match="fold disagrees"):
        runner._prepare_actions(changed, changed["resolved_configs"], allow_resume=True)


def test_resume_refuses_changed_fixed_folds_without_touching_existing_artifacts(cv_action_fixture):
    plan, folds = cv_action_fixture
    run = plan["runs"][0]
    cfg = plan["resolved_configs"][run["config_key"]]
    directory = Path(run["run_dir"])
    directory.mkdir(parents=True)
    save_config(cfg, directory / "resolved_config.yaml")
    identity = runner.make_identity(cfg, run["seed"], Path(cfg["dataset"]["manifest"]), Path(cfg["dataset"]["label_map"]))
    (directory / "provenance.json").write_text(json.dumps(identity), encoding="utf-8")
    (directory / "status.json").write_text(json.dumps({"status": "failed", "identity_hash": identity["identity_hash"],
        "seed": 42, "fold": 0}), encoding="utf-8")
    (directory / "checkpoints").mkdir()
    checkpoint = directory / "checkpoints/final_regular.pt"
    checkpoint.write_bytes(b"synthetic epoch marker; no trained model")
    run["status"] = "failed"
    before = {str(path): path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    folds.write_text("changed fixed fold assignment", encoding="utf-8")
    with pytest.raises(ConfigError, match="identity drift"):
        runner._prepare_actions(plan, plan["resolved_configs"], allow_resume=True)
    assert {str(path): path.read_bytes() for path in directory.rglob("*") if path.is_file()} == before


def test_main_dispatches_each_fold_config_to_the_single_training_entry(cv_action_fixture, monkeypatch):
    """Mock dispatch only: no checkpoint, learned prediction or result is produced."""
    import types
    import yaml
    plan, folds = cv_action_fixture
    temporary_root = folds.parent
    cfg = copy.deepcopy(plan["resolved_configs"]["B04_fold0"])
    cfg["dataset"]["cv_fold"] = None
    base = temporary_root / "dispatch_base.yaml"
    save_config(cfg, base)
    suite = temporary_root / "dispatch_suite.yaml"
    suite.write_text(yaml.safe_dump({"suite_id": "mock_dispatch_cv", "base_config": str(base),
        "protocol_config": None, "seeds": [42], "experiments": [
            {"id": "B04", "parent": "base", "overrides": {"model.fusion": "mean"}}]}), encoding="utf-8")
    monkeypatch.setattr(runner, "PROJECT_ROOT", temporary_root)
    fake_train = types.ModuleType("src.train")
    checked, trained, verified = [], [], []

    def preflight(run_cfg):
        checked.append(run_cfg["dataset"]["cv_fold"])
        return {"protocol": "dev_cv5", "fold": run_cfg["dataset"]["cv_fold"], "test_evaluation": False}

    def train(run_cfg, seed, output_dir=None, resume=None):
        fold = run_cfg["dataset"]["cv_fold"]
        assert seed == 42 and resume is None
        directory = Path(output_dir)
        assert f"B04_seed42_fold{fold}_" in directory.name
        assert run_cfg["experiment"]["id"] == "B04"
        identity = runner.make_identity(run_cfg, seed, Path(run_cfg["dataset"]["manifest"]),
                                        Path(run_cfg["dataset"]["label_map"]))
        directory.mkdir(parents=True)
        (directory / "status.json").write_text(json.dumps({"status": "complete", "identity_hash": identity["identity_hash"],
            "seed": seed, "fold": fold}), encoding="utf-8")
        (directory / "provenance.json").write_text(json.dumps(identity), encoding="utf-8")
        trained.append(fold)
        return directory

    def verify(directory, expected, run_cfg):
        assert expected["fold"] == run_cfg["dataset"]["cv_fold"]
        assert expected["config_key"] == f"B04_fold{expected['fold']}"
        verified.append(expected["fold"])

    fake_train.preflight, fake_train.run_training = preflight, train
    monkeypatch.setitem(sys.modules, "src.train", fake_train)
    monkeypatch.setattr(runner, "verify_run", verify)
    assert runner.main(["--suite", str(suite)]) == 0
    assert checked == trained == verified == list(range(5))
    written = json.loads((temporary_root / "outputs/suites/mock_dispatch_cv/plan.json").read_text())
    assert written["run_count"] == 5
    assert all(run["status"] == "complete" for run in written["runs"])
