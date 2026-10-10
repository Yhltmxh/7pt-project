"""Installer boundaries without creating real Conda environments or downloading."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux/WSL installer")


@pytest.fixture
def setup(tmp_path):
    project = tmp_path / "project with spaces"
    (project / "scripts").mkdir(parents=True)
    shutil.copyfile(ROOT / "scripts/setup_conda.sh", project / "scripts/setup_conda.sh")
    shutil.copyfile(ROOT / "requirements.txt", project / "requirements.txt")
    base = tmp_path / "Conda base"
    (base / "bin").mkdir(parents=True)
    (base / "bin/python").symlink_to(sys.executable)
    executable = tmp_path / "mock conda"
    executable.write_text("""#!/usr/bin/env python3
import json, os, pathlib, subprocess, sys
args = sys.argv[1:]
base = pathlib.Path(os.environ['SETUP_TEST_BASE'])
with pathlib.Path(os.environ['SETUP_TEST_CALLS']).open('a') as f:
    f.write(json.dumps(args) + '\\n')
if args == ['info', '--base']:
    print(base)
elif args == ['env', 'list', '--json']:
    print(json.dumps({'envs': [str(base / 'envs/dlenv')] if (base / 'exists').exists() else []}))
elif args[0] == 'create':
    (base / 'exists').touch()
elif args[0] == 'run':
    command = args[4:]
    assert command[0] == 'python', command
    if command[1:4] == ['-m', 'pip', 'install']:
        if '--constraint' in command:
            constraints = pathlib.Path(command[command.index('--constraint') + 1]).read_text()
            assert 'torch==2.8.0+cpu' in constraints and 'torchvision==0.23.0+cpu' in constraints
    elif command[1] == '-c' and os.environ.get('SETUP_TEST_BAD_PYTHON'):
        print('Python 3.12 required', file=sys.stderr)
        sys.exit(2)
    else:
        sys.exit(subprocess.call([os.environ['SETUP_TEST_PYTHON'], *command[1:]]))
else:
    raise AssertionError(args)
""", encoding="utf-8")
    executable.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    env = {**os.environ, "CONDA_EXE": str(executable), "SETUP_TEST_BASE": str(base),
           "SETUP_TEST_CALLS": str(calls), "SETUP_TEST_PYTHON": sys.executable}

    def run(*args):
        result = subprocess.run(["bash", str(project / "scripts/setup_conda.sh"), *args],
                                env=env, capture_output=True, text=True)
        recorded = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
        return result, recorded
    return project, base, env, run


def test_default_dlenv_dry_run_needs_no_conda(setup):
    project, _, env, run = setup
    env["CONDA_EXE"] = "/missing/conda"
    result, calls = run("--dry-run")
    assert result.returncode == 0 and not calls and not (project / "outputs").exists()
    assert "Environment: dlenv" in result.stdout
    assert "Python 3.13.9" in result.stdout
    assert "torch==2.9.0+cu130" in result.stdout and "torchvision==0.24.0+cu130" in result.stdout


@pytest.mark.parametrize("args", [("--name", "base"), ("--name", "root"),
    ("--variant", "cu129"), ("--name", "bad/name"), ("--name",), ("--unknown",)])
def test_rejects_invalid_settings_before_environment_mutation(setup, args):
    _, _, _, run = setup
    result, calls = run(*args)
    assert result.returncode == 2 and not calls


def test_existing_dlenv_is_preserved_without_explicit_reuse(setup):
    project, base, _, run = setup
    (base / "exists").touch()
    result, calls = run()
    assert result.returncode == 2 and "exists" in result.stderr
    assert calls == [["info", "--base"], ["env", "list", "--json"]]
    assert not (project / "outputs").exists()


def test_reuse_requires_existing_environment(setup):
    project, _, _, run = setup
    result, calls = run("--reuse")
    assert result.returncode == 2 and "does not exist" in result.stderr
    assert not any(c[0] == "create" for c in calls) and not (project / "outputs").exists()


def test_reuse_wrong_python_stops_before_pip(setup):
    _, base, env, run = setup
    (base / "exists").touch()
    env["SETUP_TEST_BAD_PYTHON"] = "1"
    result, calls = run("--reuse")
    assert result.returncode == 2
    assert not any("pip" in c for c in calls)


@pytest.mark.parametrize("reuse", [False, True])
def test_mock_conda_routing_and_actual_cpu_dependency_probe(setup, reuse):
    import torch
    import torchvision
    if (torch.__version__, torchvision.__version__) != ("2.8.0+cpu", "0.23.0+cpu"):
        pytest.skip("Real CPU probe uses the project's validated CPU wheel pair")
    project, base, _, run = setup
    if reuse:
        (base / "exists").touch()
    result, calls = run("--variant", "cpu", *( ["--reuse"] if reuse else []))
    assert result.returncode == 0, result.stderr
    assert any(c[0] == "create" for c in calls) != reuse
    reports = list((project / "outputs").glob("conda_setup.*/environment.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["probe_passed"] and report["selected_device"] == "cpu"
    assert report["torch"] == "2.8.0+cpu" and report["torchvision"] == "0.23.0+cpu"
    assert len(report["requirements_sha256"]) == len(report["setup_script_sha256"]) == 64
    assert reports[0].with_name("requirements.freeze.txt").is_file()
    assert "conda activate dlenv" in result.stdout and "--device cpu" in result.stdout
