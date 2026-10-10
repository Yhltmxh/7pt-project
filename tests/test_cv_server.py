"""Seed42 server settings, immutable fold manifests, and real synthetic CV smoke."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts import train_server as launcher
from scripts.smoke_test import create_fixture, run_cv_smoke
from src.provenance import file_hash, make_identity


def test_cv_server_dry_run_defaults_seed42_three_baselines_five_folds(tmp_path):
    code = """
import builtins, sys
original = builtins.__import__
def guarded(name, *a, **kw):
    if name.split('.')[0] in {'torch', 'torchvision', 'PIL', 'sklearn'}:
        raise AssertionError('CV dry run imported ML: ' + name)
    return original(name, *a, **kw)
builtins.__import__ = guarded
from scripts.train_server import main
raise SystemExit(main(sys.argv[1:]))
"""
    directory = tmp_path / "CV job with spaces"
    result = subprocess.run([sys.executable, "-c", code, "--protocol", "dev_cv5",
        "--data-root", str(tmp_path / "missing data"), "--job-dir", str(directory), "--dry-run"],
        cwd=launcher.PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    job = json.loads((directory / "job.json").read_text())
    assert job["settings"]["seeds"] == [42]
    assert job["settings"]["split_seed"] == 20261004
    assert job["settings"]["only"] == ["B01", "B02", "B04"]
    assert job["n_folds"] == 5 and job["formal"] is False
    plan = json.loads((Path(job["suite_dir"]) / "plan.json").read_text())
    assert plan["run_count"] == 15 and plan["evaluation_partition"] == "dev_cv_oof"
    assert {(r["spec_id"], r["seed"], r["fold"]) for r in plan["runs"]} == {
        (spec, 42, fold) for spec in ("B01", "B02", "B04") for fold in range(5)}
    assert len(plan["resolved_configs"]) == 15
    assert not (directory / "data").exists() and not (directory / "runs").exists()


def test_server_fold_manifest_is_fixed_and_drift_is_refused(tmp_path):
    root = tmp_path / "synthetic"
    prepared = create_fixture(root, n_cases=30, train_count=15, valid_count=10)
    args = launcher.parser().parse_args(["--protocol", "dev_cv5", "--seeds", "42",
        "--data-root", str(root), "--job-dir", str(tmp_path / "job"), "--device", "cpu"])
    directory, job = launcher._resolve_job(args)
    shutil.copytree(prepared, directory / "data")
    original_manifest = (directory / "data/samples.csv").read_bytes()
    first = launcher._prepare_folds(job)
    assert first["development_count"] == 25 and first["test_assigned"] is False
    assert [f["valid_count"] for f in first["folds"]] == [5] * 5
    hashes = {p.name: file_hash(p) for p in (directory / "data").iterdir() if p.is_file()}
    assert launcher._prepare_folds(job) == first
    assert hashes == {p.name: file_hash(p) for p in (directory / "data").iterdir() if p.is_file()}
    assert (directory / "data/samples.csv").read_bytes() == original_manifest
    from scripts.run_experiments import expand_suite
    configs = expand_suite(directory / "suite.yaml", only=["B04"], overrides=job["overrides"])["experiments"]
    a, b = [make_identity(s["config"], 42, directory / "data/samples.csv", directory / "data/label_map.json") for s in configs[:2]]
    assert a["identity_hash"] != b["identity_hash"]
    assert a["split_hash"] == b["split_hash"] and a["folds_hash"] == b["folds_hash"]
    job["settings"]["split_seed"] = 12345
    before = {p.name: p.read_bytes() for p in (directory / "data").iterdir() if p.is_file()}
    with pytest.raises((ValueError, RuntimeError), match="[Ss]eed|[Ff]old|assignment"):
        launcher._prepare_folds(job)
    assert before == {p.name: p.read_bytes() for p in (directory / "data").iterdir() if p.is_file()}


def test_seed42_real_synthetic_five_fold_pipeline(tmp_path):
    report = run_cv_smoke(tmp_path, device="cpu")
    assert report["passed"] and report["synthetic_only"] and not report["formal"]
    assert report["run_count"] == 15 and report["training_seed"] == 42


def test_cv_server_real_synthetic_recovery_and_pooled_collection(tmp_path, monkeypatch):
    """Use the real server, trainer and collector with a five-fold tiny fixture."""
    import torch
    import numpy as np
    from scripts import run_experiments as runner
    from src import train
    from src.config import apply_overrides
    from src.data import manifest

    torch.set_num_threads(2)
    root = tmp_path / "raw CV data"
    create_fixture(root, n_cases=30, train_count=15, valid_count=10)
    raw_hashes = {str(p.relative_to(root)): file_hash(p) for p in (root / "meta").iterdir()}
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
        assert seed == 42 and cfg["dataset"]["cv_fold"] == 0 and resume is None
        original_train(cfg, seed, output_dir=output_dir, stop_after_epoch=1)
        raise RuntimeError("CV server interruption after checkpoint")

    monkeypatch.setattr(runner, "expand_suite", tiny_expand)
    monkeypatch.setattr(manifest, "prepare_dataset", synthetic_prepare)
    monkeypatch.setattr(train, "run_training", interrupt_after_epoch)
    directory = tmp_path / "server CV job"
    args = ["--protocol", "dev_cv5", "--data-root", str(root), "--job-dir", str(directory),
            "--only", "B04", "--seeds", "42", "--epochs", "2", "--warmup-epochs", "0",
            "--device", "cpu", "--num-workers", "0", "--batch-size", "3", "--gradient-accumulation", "3"]
    assert launcher.main(args) == 2
    job = json.loads((directory / "job.json").read_text())
    suite_dir = Path(job["suite_dir"])
    saved_folds = (directory / "data/folds.csv").read_bytes()
    first_plan = json.loads((suite_dir / "plan.json").read_text())
    assert first_plan["run_count"] == 5 and first_plan["runs"][0]["status"] == "failed"
    monkeypatch.setattr(train, "run_training", original_train)
    assert launcher.main(["--resume", str(directory)]) == 0
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["run_count"] == 5 and summary["protocol"] == "dev_cv5" and not summary["formal"]
    assert summary["collection"]["verified_complete_runs"] == 5
    assert Path(summary["seed_metrics_csv"]).is_file()
    pooled = Path(summary["oof_directory"]) / "B04_seed42_ema_argmax_notta.npz"
    with np.load(pooled, allow_pickle=False) as archive:
        assert len(archive["sample_id"]) == 25 and set(archive["fold"].tolist()) == set(range(5))
    assert (directory / "data/folds.csv").read_bytes() == saved_folds
    assert raw_hashes == {str(p.relative_to(root)): file_hash(p) for p in (root / "meta").iterdir()}
    assert not list((directory / "runs").rglob("test*.npz"))
    complete_plan = json.loads((suite_dir / "plan.json").read_text())
    hashes = {r["run_dir"]: file_hash(Path(r["run_dir"]) / "checkpoints/final_regular.pt")
              for r in complete_plan["runs"]}
    monkeypatch.setattr(train, "run_training", lambda *a, **kw: pytest.fail("complete CV fold was retrained"))
    assert launcher.main(["--resume", str(directory)]) == 0
    assert hashes == {r["run_dir"]: file_hash(Path(r["run_dir"]) / "checkpoints/final_regular.pt")
                      for r in complete_plan["runs"]}
