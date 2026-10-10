"""Server orchestration checks; real training coverage remains in smoke tests."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import train_server as launcher
from src.config import ConfigError
from src.provenance import file_hash


def _args(tmp_path, *extra):
    return launcher.parser().parse_args([
        "--data-root", str(tmp_path / "data with spaces"), "--job-dir", str(tmp_path / "job"),
        "--only", "B04", "--seeds", "101", *extra])


def _files(directory):
    return {str(path.relative_to(directory)): path.read_bytes() for path in directory.rglob("*") if path.is_file()}


def test_dry_run_never_imports_ml_or_reads_dataset(tmp_path):
    code = """
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torchvision', 'PIL', 'sklearn'}:
        raise AssertionError('dry run imported ML: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from scripts.train_server import main
raise SystemExit(main(sys.argv[1:]))
"""
    result = subprocess.run([sys.executable, "-c", code,
        "--data-root", str(tmp_path / "missing raw data"), "--job-dir", str(tmp_path / "dry"),
        "--only", "B04", "--seeds", "101", "--dry-run"], cwd=launcher.PROJECT_ROOT,
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    job = json.loads((tmp_path / "dry/job.json").read_text())
    assert not (tmp_path / "dry/data").exists()
    assert not (tmp_path / "dry/runs").exists()
    assert job["formal"] is False and job["test_inference"] is False
    assert job["settings"]["batch_size"] == 1
    plan = json.loads((Path(job["suite_dir"]) / "plan.json").read_text())
    assert plan["run_count"] == 1 and plan["runs"][0]["status"] == "planned"
    assert plan["resolved_configs"]["B04"]["dataset"]["root"].endswith("missing raw data")


def test_explicit_data_root_and_invalid_arguments_fail_before_job_creation(tmp_path, monkeypatch):
    monkeypatch.delenv("DERM7PT_ROOT", raising=False)
    with pytest.raises(ConfigError, match="data-root"):
        launcher._resolve_job(launcher.parser().parse_args(["--dry-run"]))
    for extra in (["--epochs", "2"], ["--seeds", "101", "101"], ["--only", "B04", "B04"]):
        with pytest.raises(ConfigError):
            launcher._resolve_job(_args(tmp_path, *extra))
        assert not (tmp_path / "job").exists()


def test_resume_loads_saved_parameters_and_rejects_drift_without_writes(tmp_path, monkeypatch):
    directory, job = launcher._resolve_job(_args(tmp_path, "--epochs", "30", "--num-workers", "0"))
    # Environment data root affects only a new job, never a saved job.
    monkeypatch.setenv("DERM7PT_ROOT", "/unrelated/data")
    restored_dir, restored = launcher._resolve_job(launcher.parser().parse_args(["--resume", str(directory)]))
    assert restored_dir == directory and restored == job
    assert restored["settings"]["epochs"] == 30
    before = _files(directory)
    with pytest.raises(ConfigError, match="setting drift"):
        launcher._resolve_job(launcher.parser().parse_args(["--resume", str(directory), "--epochs", "31"]))
    assert _files(directory) == before
    monkeypatch.setattr(launcher, "_source", lambda: {"changed": True})
    with pytest.raises(ConfigError, match="hash drift"):
        launcher._resolve_job(launcher.parser().parse_args(["--resume", str(directory)]))
    assert _files(directory) == before


def test_changed_suite_and_existing_job_are_never_overwritten(tmp_path):
    args = _args(tmp_path)
    directory, _ = launcher._resolve_job(args)
    with pytest.raises(ConfigError, match="already exists"):
        launcher._resolve_job(args)
    (directory / "suite.yaml").write_text("changed\n")
    before = _files(directory)
    with pytest.raises(ConfigError, match="suite.yaml changed"):
        launcher._resolve_job(launcher.parser().parse_args(["--resume", str(directory)]))
    assert _files(directory) == before


def test_job_lock_excludes_concurrent_launcher_and_releases(tmp_path):
    with launcher._lock(tmp_path):
        with pytest.raises(ConfigError, match="already running"):
            with launcher._lock(tmp_path):
                pytest.fail("second launcher acquired the job")
    with launcher._lock(tmp_path):
        pass


def test_storage_checks_entire_unfinished_suite(tmp_path, monkeypatch):
    usage = type("Usage", (), {"free": 12 * launcher.GIB})()
    monkeypatch.setattr(launcher.shutil, "disk_usage", lambda _: usage)
    job = {"job_dir": str(tmp_path)}
    with pytest.raises(RuntimeError, match="Insufficient"):
        launcher._storage(job, {"runs": [{"status": "planned"}] * 15})
    result = launcher._storage(job, {"runs": [{"status": "complete"}] * 14 + [{"status": "planned"}]})
    assert result["required_free_bytes"] == 6 * launcher.GIB


def test_saved_manifest_is_not_regenerated(tmp_path, monkeypatch):
    from src.data import manifest
    root = tmp_path / "raw"
    root.mkdir()
    directory = tmp_path / "job/data"
    directory.mkdir(parents=True)
    path = directory / "samples.csv"
    path.write_text("original manifest")
    audit = {"manifest_sha256": file_hash(path), "passed": True, "formal_ready": False,
             "sample_count": 1011, "status": "needs_review"}
    (directory / "dataset_audit.json").write_text(json.dumps(audit))
    monkeypatch.setattr(manifest, "prepare_dataset", lambda *a, **kw: pytest.fail("recovery regenerated the manifest"))
    job = {"job_dir": str(tmp_path / "job"), "settings": {"data_root": str(root)}}
    assert launcher._prepare_data(job) == audit
    path.write_text("drift")
    before = _files(directory)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        launcher._prepare_data(job)
    assert _files(directory) == before


def test_pretrained_cache_missing_corrupt_and_valid(tmp_path, monkeypatch):
    import torch
    from torchvision.models import get_weight
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path / "nested/cache"))
    configs = {"B04": {"model": {"pretrained_id": "ConvNeXt_Base_Weights.IMAGENET1K_V1"}}}
    report = launcher._weights(configs, offline=False, download=False)
    assert report[0]["available"] is False
    with pytest.raises(RuntimeError, match="Offline.*missing"):
        launcher._weights(configs, offline=True, download=False)
    cached = Path(report[0]["cache_path"])
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"damaged checkpoint")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        launcher._weights(configs, offline=True, download=False)
    from src import provenance
    prefix = Path(get_weight(configs["B04"]["model"]["pretrained_id"]).url).name.rsplit("-", 1)[-1].split(".")[0]
    monkeypatch.setattr(provenance, "file_hash", lambda _: prefix + "0" * (64 - len(prefix)))
    verified = launcher._weights(configs, offline=True, download=False)
    assert verified[0]["available"] is True
    assert len(verified[0]["sha256"]) == 64


def test_pretrained_download_checks_separate_cache_disk_before_network(tmp_path, monkeypatch):
    import torch
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path / "cache"))
    monkeypatch.setattr(launcher.shutil, "disk_usage", lambda _: type("Usage", (), {"free": 1})())
    monkeypatch.setattr(torch.hub, "download_url_to_file", lambda *a, **kw: pytest.fail("download started on full cache disk"))
    configs = {"B04": {"model": {"pretrained_id": "ConvNeXt_Base_Weights.IMAGENET1K_V1"}}}
    with pytest.raises(RuntimeError, match="cache disk space"):
        launcher._weights(configs, offline=False, download=True)
    assert not (tmp_path / "cache").exists()


def test_cpu_environment_probe_preserves_rng():
    import torch
    torch.set_num_threads(2)
    before = torch.get_rng_state().clone()
    report = launcher._environment({"device": "cpu"})
    assert report["optimizer_update_passed"]
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("failure", [False, True])
def test_launch_pipeline_collects_success_and_propagates_failure(tmp_path, monkeypatch, failure):
    from scripts import run_experiments, collect_results
    args = _args(tmp_path)
    directory, job = launcher._resolve_job(args)
    real_runner = run_experiments.main
    # Save an actual validated plan, then mock only checks/training orchestration.
    assert real_runner(launcher._runner_args(job, "--dry-run")) == 0
    calls = []

    def mock_runner(argv):
        calls.append(argv)
        print("runner invoked")
        return 2 if failure and "--dry-run" not in argv and "--preflight-only" not in argv else 0

    monkeypatch.setattr(run_experiments, "main", mock_runner)
    monkeypatch.setattr(launcher, "_environment", lambda _: {"verified": "mock"})
    monkeypatch.setattr(launcher, "_storage", lambda *a: {})
    monkeypatch.setattr(launcher, "_prepare_data", lambda _: {"passed": True, "formal_ready": False})
    monkeypatch.setattr(launcher, "_weights", lambda *a, **kw: [])
    collected = []

    def mock_collect(path):
        collected.append(path)
        return {"all_preregistered_seeds_complete": True, "formal": False}

    monkeypatch.setattr(collect_results, "collect_suite", mock_collect)
    if failure:
        with pytest.raises(RuntimeError, match="suite failed"):
            launcher._run(args, directory, job)
        assert not collected
        assert not (directory / "summary.json").exists()
    else:
        assert launcher._run(args, directory, job) == 0
        assert collected == [Path(job["suite_dir"])]
        assert json.loads((directory / "summary.json").read_text())["formal"] is False
    assert ["--dry-run" in argv for argv in calls] == [True, False, False]
    assert "runner invoked" in (directory / "train.log").read_text()
    assert json.loads((directory / "status.json").read_text())["status"] == ("failed" if failure else "complete")


def test_shell_preserves_python_exit_code_and_handles_spaces(tmp_path):
    fake_python = tmp_path / "python with spaces"
    fake_python.write_text("#!/bin/bash\nexit 17\n")
    fake_python.chmod(0o755)
    result = subprocess.run(["bash", str(launcher.PROJECT_ROOT / "scripts/train_server.sh"),
                             "--python", str(fake_python), "--data-root", "/data/with spaces"], cwd=tmp_path)
    assert result.returncode == 17
    invalid = subprocess.run(["bash", str(launcher.PROJECT_ROOT / "scripts/train_server.sh"), "--python"],
                             capture_output=True, text=True)
    assert invalid.returncode == 2 and "requires" in invalid.stderr


def test_synthetic_server_epoch_recovery_and_verified_collection(tmp_path, monkeypatch):
    """Exercise the real runner/trainer/collector with synthetic data and tiny encoders."""
    import torch
    from scripts import run_experiments as runner
    from scripts.smoke_test import create_fixture
    from src import train
    from src.config import apply_overrides
    from src.data import manifest

    torch.set_num_threads(2)
    root = tmp_path / "raw synthetic data"
    create_fixture(root)
    original_expand, original_prepare, original_train = runner.expand_suite, manifest.prepare_dataset, train.run_training

    def tiny_expand(*a, **kw):
        result = original_expand(*a, **kw)
        for spec in result["experiments"]:
            spec["config"] = apply_overrides(spec["config"], {
                "model.clinical_encoder": "tiny_cnn", "model.dermoscopic_encoder": "tiny_cnn",
                "model.pretrained_id": None, "preprocess.image_size": 32})
        return result

    def synthetic_prepare(*a, **kw):
        kw["require_official_counts"] = False
        return original_prepare(*a, **kw)

    def interrupt_after_epoch(cfg, seed, output_dir=None, resume=None):
        assert resume is None
        original_train(cfg, seed, output_dir=output_dir, stop_after_epoch=1)
        raise RuntimeError("synthetic interruption after checkpoint")

    monkeypatch.setattr(runner, "expand_suite", tiny_expand)
    monkeypatch.setattr(manifest, "prepare_dataset", synthetic_prepare)
    monkeypatch.setattr(train, "run_training", interrupt_after_epoch)
    job_dir = tmp_path / "server synthetic job"
    args = ["--data-root", str(root), "--job-dir", str(job_dir), "--only", "B04", "--seeds", "101",
            "--epochs", "2", "--warmup-epochs", "0", "--device", "cpu", "--num-workers", "0",
            "--batch-size", "3", "--gradient-accumulation", "3"]
    assert launcher.main(args) == 2
    job = json.loads((job_dir / "job.json").read_text())
    suite_dir = Path(job["suite_dir"])
    interrupted_plan = json.loads((suite_dir / "plan.json").read_text())
    run = Path(interrupted_plan["runs"][0]["run_dir"])
    assert interrupted_plan["runs"][0]["status"] == "failed"
    assert (run / "checkpoints/final_regular.pt").is_file()
    assert not (run / "predictions/valid_ema_argmax_notta.npz").exists()
    monkeypatch.setattr(train, "run_training", original_train)
    assert launcher.main(["--resume", str(job_dir)]) == 0
    completed = json.loads((job_dir / "summary.json").read_text())
    assert completed["run_count"] == 1 and completed["formal"] is False
    assert completed["collection"]["verified_complete_runs"] == 1
    assert (suite_dir / "aggregate.csv").is_file()
    assert not list(run.rglob("test*.npz"))
    regular_hash = file_hash(run / "checkpoints/final_regular.pt")
    monkeypatch.setattr(train, "run_training", lambda *a, **kw: pytest.fail("verified complete run was retrained"))
    assert launcher.main(["--resume", str(job_dir)]) == 0
    assert file_hash(run / "checkpoints/final_regular.pt") == regular_hash
