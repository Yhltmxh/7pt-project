"""Mocked server recovery contracts; fixture metrics are never research evidence."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

import scripts.run_experiments as runner
import src.provenance as provenance_tools
from src.config import load_config, save_config
from src.evaluation.metrics import classification_metrics, softmax
from src.provenance import file_hash, object_hash


@pytest.fixture
def server_fixture(tmp_path, monkeypatch):
    root = tmp_path / "data"
    (root / "meta").mkdir(parents=True)
    for name in ("meta.csv", "train_indexes.csv", "valid_indexes.csv", "test_indexes.csv"):
        (root / "meta" / name).write_text("synthetic fixture\n", encoding="utf-8")
    manifest = tmp_path / "samples.csv"
    manifest.write_text("sample_id,official_split,diagnosis_index,exclusion_reason\n" +
                        "".join(f"fixture_{i},valid,{i},\n" for i in range(5)), encoding="utf-8")
    label_map = tmp_path / "label_map.json"
    label_map.write_text(json.dumps({"class_order": ["BCC", "NEV", "MEL", "MISC", "SK"]}), encoding="utf-8")
    cfg = load_config("configs/base.yaml", {
        "stage": "development", "dataset.root": str(root), "dataset.manifest": str(manifest),
        "dataset.label_map": str(label_map), "dataset.clinical_substitution_policy": "retain_official",
        "train.epochs": 2, "train.warmup_epochs": 0, "train.device": "cpu",
        "experiment.output_root": str(tmp_path / "runs"),
    })
    base = tmp_path / "base.yaml"
    save_config(cfg, base)
    suite = tmp_path / "suite.yaml"
    suite.write_text(yaml.safe_dump({"suite_id": "mock_server", "base_config": str(base),
        "protocol_config": None, "seeds": [101], "experiments": [
            {"id": "B04", "parent": "base", "overrides": {"model.fusion": "mean"}},
            {"id": "B05", "parent": "B04", "overrides": {"model.fusion": "concat"}},
        ]}), encoding="utf-8")
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    source_files = {"fixture_source.py": "a" * 64}
    monkeypatch.setattr(provenance_tools, "source_snapshot", lambda: {
        "files": source_files, "sha256": object_hash(source_files)})
    train = types.ModuleType("src.train")
    train.preflight = lambda cfg: {"device": "cpu", "test_evaluation": False}
    monkeypatch.setitem(sys.modules, "src.train", train)
    return types.SimpleNamespace(root=tmp_path, data_root=root, cfg=cfg, suite=suite, train=train,
        args=["--suite", str(suite), "--only", "B04", "--seeds", "101"],
        plan_path=tmp_path / "outputs/suites/mock_server/plan.json", source_files=source_files)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _initialize_mock_run(cfg, seed, output_dir, status):
    """Only orchestration is exercised; these are explicitly synthetic bytes."""
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = runner.make_identity(cfg, seed, Path(cfg["dataset"]["manifest"]), Path(cfg["dataset"]["label_map"]))
    save_config(cfg, run_dir / "resolved_config.yaml")
    (run_dir / "provenance.json").write_text(json.dumps(identity), encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({"status": status, "seed": seed,
        "identity_hash": identity["identity_hash"]}), encoding="utf-8")
    return run_dir, identity


def _complete_mock_run(cfg, seed, output_dir, resume=None):
    run_dir, identity = _initialize_mock_run(cfg, seed, output_dir, "complete")
    for directory in ("checkpoints", "predictions", "metrics"):
        (run_dir / directory).mkdir(exist_ok=True)
    regular = run_dir / "checkpoints/final_regular.pt"
    ema = run_dir / "checkpoints/final_ema.pt"
    regular.write_bytes(b"synthetic regular fixture; no trained model")
    ema.write_bytes(b"synthetic EMA fixture; no trained model")
    logits = np.eye(5, dtype=np.float32)
    prediction = run_dir / "predictions/valid_ema_argmax_notta.npz"
    np.savez_compressed(prediction, sample_id=np.asarray([f"fixture_{i}" for i in range(5)]),
        y_true=np.arange(5), logits=logits, probabilities=softmax(logits),
        class_order=np.asarray(cfg["evaluation"]["class_order"]), seed=seed, fold=-1,
        eval_partition="official_valid", checkpoint_hash=file_hash(ema))
    metric = run_dir / "metrics/valid_ema_argmax_notta.json"
    metric.write_text(json.dumps(classification_metrics(logits, np.arange(5))), encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({"status": "complete", "seed": seed,
        "identity_hash": identity["identity_hash"], "artifact_hashes": {
            path.relative_to(run_dir).as_posix(): file_hash(path) for path in (regular, ema, prediction, metric)}}),
        encoding="utf-8")
    return run_dir


def _failing_trainer(with_checkpoint):
    def run(cfg, seed, output_dir=None, resume=None):
        assert resume is None
        directory, _ = _initialize_mock_run(cfg, seed, output_dir, "failed")
        (directory / "preserved.txt").write_text("original failure evidence", encoding="utf-8")
        if with_checkpoint:
            (directory / "checkpoints").mkdir()
            (directory / "checkpoints/final_regular.pt").write_bytes(b"synthetic completed-epoch marker")
        raise RuntimeError("mock training interruption")
    return run


def test_failed_run_records_actual_path_and_explicit_resume_calls_epoch_checkpoint(server_fixture):
    fixture = server_fixture
    fixture.train.run_training = _failing_trainer(with_checkpoint=True)
    assert runner.main(fixture.args) == 2
    plan = _json(fixture.plan_path)
    saved = plan["runs"][0]
    assert saved["status"] == "failed"
    assert saved["run_dir"] and saved["identity"]["source_hash"]
    assert Path(saved["run_dir"]).is_dir()
    calls = []

    def resumed(cfg, seed, output_dir=None, resume=None):
        calls.append((seed, Path(output_dir), Path(resume)))
        assert Path(resume).read_bytes() == b"synthetic completed-epoch marker"
        return _complete_mock_run(cfg, seed, output_dir, resume)

    fixture.train.run_training = resumed
    assert runner.main(fixture.args + ["--resume"]) == 0
    assert calls == [(101, Path(saved["run_dir"]), Path(saved["run_dir"]) / "checkpoints/final_regular.pt")]
    assert (Path(saved["run_dir"]) / "preserved.txt").read_text() == "original failure evidence"
    assert _json(fixture.plan_path)["runs"][0]["status"] == "complete"


def test_verified_complete_run_reused_without_calling_training(server_fixture):
    fixture = server_fixture
    fixture.train.run_training = _complete_mock_run
    assert runner.main(fixture.args) == 0
    path = Path(_json(fixture.plan_path)["runs"][0]["run_dir"])
    run_before = _snapshot(path)

    def forbidden(*args, **kwargs):
        pytest.fail("A verified complete run must not be retrained")

    fixture.train.run_training = forbidden
    assert runner.main(fixture.args + ["--resume"]) == 0
    assert _snapshot(path) == run_before
    assert _json(fixture.plan_path)["runs"][0]["next_action"] == "reuse_complete"


def test_completed_artifact_tampering_is_rejected_before_any_writes(server_fixture, capsys):
    fixture = server_fixture
    fixture.train.run_training = _complete_mock_run
    assert runner.main(fixture.args) == 0
    path = Path(_json(fixture.plan_path)["runs"][0]["run_dir"])
    (path / "predictions/valid_ema_argmax_notta.npz").write_bytes(b"tampered")
    before = _snapshot(fixture.root)
    assert runner.main(fixture.args + ["--resume"]) == 2
    assert _snapshot(fixture.root) == before
    assert "hash mismatch" in capsys.readouterr().err


def test_failed_run_without_checkpoint_preserves_entire_job(server_fixture, capsys):
    fixture = server_fixture
    fixture.train.run_training = _failing_trainer(with_checkpoint=False)
    assert runner.main(fixture.args) == 2
    before = _snapshot(fixture.root)
    assert runner.main(fixture.args + ["--resume"]) == 2
    assert _snapshot(fixture.root) == before
    assert "no final_regular.pt epoch checkpoint" in capsys.readouterr().err


def test_attempted_suite_without_explicit_resume_is_rejected(server_fixture):
    fixture = server_fixture
    fixture.train.run_training = _complete_mock_run
    assert runner.main(fixture.args) == 0
    before = _snapshot(fixture.root)
    assert runner.main(fixture.args) == 2
    assert _snapshot(fixture.root) == before


@pytest.mark.parametrize("change", [["--set", "train.batch_size=8"], ["--seeds", "777"], ["--only", "B05"]])
def test_plan_configuration_seed_and_selection_drift_rejected_before_writes(server_fixture, change, capsys):
    fixture = server_fixture
    fixture.train.run_training = _complete_mock_run
    assert runner.main(fixture.args) == 0
    before = _snapshot(fixture.root)
    fixture.train.preflight = lambda cfg: pytest.fail("Plan drift must be rejected before ML preflight")
    assert runner.main(fixture.args + change + ["--resume"]) == 2
    assert _snapshot(fixture.root) == before
    assert "drift" in capsys.readouterr().err


@pytest.mark.parametrize("kind", ["source", "raw_data"])
def test_source_or_raw_data_identity_drift_rejected_before_writes(server_fixture, kind, capsys):
    fixture = server_fixture
    fixture.train.run_training = _complete_mock_run
    assert runner.main(fixture.args) == 0
    if kind == "source":
        fixture.source_files["fixture_source.py"] = "b" * 64
    else:
        (fixture.data_root / "meta/meta.csv").write_text("changed raw data", encoding="utf-8")
    before = _snapshot(fixture.root)
    assert runner.main(fixture.args + ["--resume"]) == 2
    assert _snapshot(fixture.root) == before
    assert "identity drift" in capsys.readouterr().err


def test_resumed_dry_run_preserves_saved_plan_and_run_artifacts(server_fixture):
    fixture = server_fixture
    fixture.train.run_training = _failing_trainer(with_checkpoint=True)
    assert runner.main(fixture.args) == 2
    before = _snapshot(fixture.root)
    assert runner.main(fixture.args + ["--resume", "--dry-run"]) == 0
    assert _snapshot(fixture.root) == before


def test_crash_after_complete_export_recovers_via_verified_reuse(server_fixture):
    fixture = server_fixture

    def completed_then_crashed(cfg, seed, output_dir=None, resume=None):
        _complete_mock_run(cfg, seed, output_dir)
        raise RuntimeError("crashed after completed artifacts before suite status")

    fixture.train.run_training = completed_then_crashed
    assert runner.main(fixture.args) == 2
    assert _json(fixture.plan_path)["runs"][0]["status"] == "failed"
    fixture.train.run_training = lambda *args, **kwargs: pytest.fail("Complete artifacts should be recovered without training")
    assert runner.main(fixture.args + ["--resume"]) == 0


@pytest.mark.parametrize("kind", ["source", "manifest"])
def test_planned_preflight_identity_can_be_rebuilt_before_any_run_attempt(server_fixture, kind):
    fixture = server_fixture
    fixture.train.run_training = lambda *args, **kwargs: pytest.fail("Preflight/dry run must never start training")
    assert runner.main(fixture.args + ["--preflight-only"]) == 0
    previous = _json(fixture.plan_path)["runs"][0]
    assert previous["status"] == "planned"
    assert not Path(previous["run_dir"]).exists()
    if kind == "source":
        fixture.source_files["fixture_source.py"] = "c" * 64
    else:
        manifest = Path(fixture.cfg["dataset"]["manifest"])
        manifest.write_text(manifest.read_text() + "\n", encoding="utf-8")
    assert runner.main(fixture.args + ["--resume", "--dry-run"]) == 0
    assert runner.main(fixture.args + ["--resume", "--preflight-only"]) == 0
    current = _json(fixture.plan_path)["runs"][0]
    assert current["status"] == "planned"
    assert current["identity"]["identity_hash"] != previous["identity"]["identity_hash"]
    assert not Path(current["run_dir"]).exists()
