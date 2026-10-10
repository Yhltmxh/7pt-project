"""One-command Linux server launcher; training remains in src.train."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import sys
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULTS = {
    "only": ["B01", "B02", "B04", "B05", "B06"], "seeds": [101, 3407, 777],
    "epochs": 25, "warmup_epochs": 5, "batch_size": 1,
    "gradient_accumulation": 16, "num_workers": 4, "device": "cuda", "offline": False,
    "protocol": "official_holdout", "split_seed": 20261004,
}
CV_DEFAULTS = {**DEFAULTS, "protocol": "dev_cv5", "only": ["B01", "B02", "B04"], "seeds": [42]}
SETTING_KEYS = set(DEFAULTS) | {"data_root"}
GIB = 1024 ** 3


def parser():
    p = argparse.ArgumentParser(description=__doc__, epilog=(
        "Holdout defaults: 5 baselines x 3 seeds. --protocol dev_cv5 defaults: "
        "B01/B02/B04 x 5 folds, seed 42. Both use 25 development epochs, CUDA, "
        "batch 1 / accumulation 16. See docs/server_training.md."))
    p.add_argument("--data-root", help="Derm7pt directory containing meta/ and images/; or DERM7PT_ROOT")
    p.add_argument("--job-dir", type=Path, help="New job directory; default outputs/server/<unique_id>")
    p.add_argument("--resume", type=Path, metavar="JOB_DIR", help="Resume the saved job with unchanged settings")
    p.add_argument("--protocol", choices=("official_holdout", "dev_cv5"),
                   help="CV defaults: B01/B02/B04 x five fixed folds, training seed 42")
    p.add_argument("--split-seed", type=int, help="Fixed CV partition seed, independent of training seeds; default 20261004")
    p.add_argument("--only", nargs="+", choices=DEFAULTS["only"])
    p.add_argument("--seeds", nargs="+", type=int)
    for name in ("epochs", "warmup_epochs", "batch_size", "gradient_accumulation", "num_workers"):
        p.add_argument("--" + name.replace("_", "-"), type=int)
    p.add_argument("--device", choices=("cuda", "cpu", "auto"))
    p.add_argument("--offline", action="store_true", default=None,
                   help="Require existing verified pretrained weights; never download")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Save/inspect plan without ML imports or data access")
    mode.add_argument("--preflight-only", action="store_true", help="Check environment/data/storage, without training")
    return p


def _absolute(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _source():
    from src.provenance import file_hash, source_snapshot
    return {"python_and_configs": source_snapshot(),
            "shell_sha256": file_hash(PROJECT_ROOT / "scripts/train_server.sh")}


def _overrides(settings, job_dir):
    return {
        "stage": "development", "dataset.root": settings["data_root"],
        "dataset.manifest": str(job_dir / "data/samples.csv"),
        "dataset.label_map": str(job_dir / "data/label_map.json"),
        "dataset.clinical_substitution_policy": "retain_official",
        "dataset.split_protocol": settings["protocol"], "dataset.split_seed": settings["split_seed"],
        "dataset.cv_fold": None,
        "dataset.folds_file": str(job_dir / "data/folds.csv") if settings["protocol"] == "dev_cv5" else None,
        "experiment.output_root": str(job_dir / "runs"),
        **{"train." + key: settings[key] for key in (
            "epochs", "warmup_epochs", "batch_size", "gradient_accumulation", "num_workers", "device")},
    }


def _resolve_job(args):
    """Reject resume drift before changing any saved metadata, data or log."""
    from src.config import ConfigError, read_yaml
    from src.provenance import file_hash, object_hash, utc_now, write_json
    from scripts.run_experiments import expand_suite
    import yaml

    if args.resume:
        if args.job_dir:
            raise ConfigError("--resume and --job-dir cannot be combined")
        job_dir = _absolute(args.resume)
        job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
        payload = {key: value for key, value in job.items() if key != "job_hash"}
        if job.get("job_hash") != object_hash(payload) or job.get("schema_version") != 1:
            raise ConfigError("Saved job metadata is incompatible or changed")
        settings = job["settings"]
        if set(settings) != SETTING_KEYS or job.get("job_dir") != str(job_dir):
            raise ConfigError("Saved job settings/directory are incompatible")
        for key in SETTING_KEYS:
            requested = getattr(args, key)
            if requested is not None:
                requested = str(_absolute(requested)) if key == "data_root" else requested
                if requested != settings[key]:
                    raise ConfigError(f"Resume setting drift in {key}; saved job preserved")
        if job["source"] != _source():
            raise ConfigError("Resume source/config/shell hash drift; saved job preserved")
        if file_hash(job_dir / "suite.yaml") != job["suite_sha256"]:
            raise ConfigError("Saved suite.yaml changed; saved job preserved")
        if _overrides(settings, job_dir) != job["overrides"]:
            raise ConfigError("Saved shared configuration is incompatible")
        expand_suite(job_dir / "suite.yaml", only=settings["only"], overrides=job["overrides"])
        return job_dir, job

    root = args.data_root or os.environ.get("DERM7PT_ROOT")
    if not root:
        raise ConfigError("Set --data-root /path/to/derm7pt or DERM7PT_ROOT")
    defaults = CV_DEFAULTS if args.protocol == "dev_cv5" else DEFAULTS
    settings = {key: getattr(args, key) if getattr(args, key) is not None else value
                for key, value in defaults.items()}
    settings["data_root"] = str(_absolute(root))
    if len(set(settings["only"])) != len(settings["only"]):
        raise ConfigError("--only must contain unique baseline IDs")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    job_dir = _absolute(args.job_dir or f"outputs/server/{stamp}_{uuid.uuid4().hex[:8]}")
    if job_dir.exists():
        raise ConfigError(f"Job directory already exists: {job_dir}; use --resume or a new directory")
    suite = read_yaml(PROJECT_ROOT / "configs/experiments/baselines.yaml")
    if settings["protocol"] == "dev_cv5":
        suite["protocol_config"] = "configs/protocols/dev_cv5.yaml"
    suite["suite_id"] = "server_" + uuid.uuid4().hex
    suite["base_config"] = str(_absolute(suite["base_config"]))
    suite["protocol_config"] = str(_absolute(suite["protocol_config"]))
    suite["seeds"] = settings["seeds"]
    overrides = _overrides(settings, job_dir)
    # Validate all CLI/config values using the existing strict schema before writing.
    from src.config import apply_overrides, deep_merge, load_config, validate_config
    validate_config(apply_overrides(deep_merge(load_config(suite["base_config"]),
                    read_yaml(suite["protocol_config"])), overrides), require_ready=True)
    if not settings["seeds"] or len(set(settings["seeds"])) != len(settings["seeds"]) or any(s < 0 for s in settings["seeds"]):
        raise ConfigError("--seeds must contain unique nonnegative integers")
    job_dir.mkdir(parents=True)
    suite_path = job_dir / "suite.yaml"
    suite_path.write_text(yaml.safe_dump(suite, sort_keys=False), encoding="utf-8")
    try:
        expand_suite(suite_path, only=settings["only"], overrides=overrides)
        job = {"schema_version": 1, "created_at": utc_now(), "job_dir": str(job_dir),
               "suite_id": suite["suite_id"], "suite_dir": str(PROJECT_ROOT / "outputs/suites" / suite["suite_id"]),
               "settings": settings, "overrides": overrides, "source": _source(),
               "suite_sha256": file_hash(suite_path), "stage": "development", "formal": False,
               "normalization": "unique_path_case_preserving_raw_references", "test_inference": False}
        job["protocol"] = settings["protocol"]
        job["n_folds"] = 5 if settings["protocol"] == "dev_cv5" else 1
        job["job_hash"] = object_hash(job)
        write_json(job_dir / "job.json", job)
    except BaseException:
        # No training or unrelated files exist in this freshly created directory.
        suite_path.unlink()
        job_dir.rmdir()
        raise
    return job_dir, job


@contextlib.contextmanager
def _lock(job_dir):
    import fcntl
    from src.config import ConfigError
    with (job_dir / ".launch.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ConfigError(f"Job is already running: {job_dir}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class _Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, value):
        self.console.write(value)
        self.log.write(value)
        self.flush()
        return len(value)

    def flush(self):
        self.console.flush()
        self.log.flush()


def _environment(settings):
    """Import matching ML packages and exercise the actual selected device."""
    import torch
    import torchvision
    import numpy  # noqa: F401
    import PIL  # noqa: F401
    import sklearn  # noqa: F401
    from src.provenance import environment_info
    device = settings["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment; install a matching CUDA PyTorch build")
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device == "cuda" else []):
        layer = torch.nn.Conv2d(3, 4, 3).to(device)
        optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda", init_scale=1024.)
        x = torch.randn(2, 3, 16, 16, device=device)
        with torch.autocast(device_type=device, enabled=device == "cuda"):
            loss = layer(x).square().mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all() for p in layer.parameters()):
            raise RuntimeError("Device forward/backward probe produced nonfinite values")
        before = layer.weight.detach().clone()
        scaler.step(optimizer)
        scaler.update()
        if torch.equal(before, layer.weight.detach()):
            raise RuntimeError("Device/AMP probe did not update the optimizer")
        if device == "cuda":
            torch.cuda.synchronize()
    report = {**environment_info(), "selected_device": device, "torch_cuda_build": torch.version.cuda,
              "torchvision": torchvision.__version__, "device_forward_backward_passed": True,
              "optimizer_update_passed": True, "probe_scope": "small Conv2d; real baseline memory capacity untested",
              "gpu": torch.cuda.get_device_name() if device == "cuda" else None}
    if Path("/dev/shm").is_dir():
        report["shared_memory_free_bytes"] = shutil.disk_usage("/dev/shm").free
    print("Environment:", json.dumps(report, ensure_ascii=False))
    return report


def _storage(job, plan):
    pending = sum(run["status"] != "complete" for run in plan["runs"])
    required = pending * 6 * GIB
    free = shutil.disk_usage(job["job_dir"]).free
    report = {"pending_runs": pending, "conservative_bytes_per_run": 6 * GIB,
              "required_free_bytes": required, "free_bytes": free}
    print(f"Storage: {pending} unfinished runs; conservative free-space budget {required / GIB:.0f} GiB; available {free / GIB:.1f} GiB")
    if free < required:
        raise RuntimeError("Insufficient output disk space for the selected suite; choose fewer runs or a larger disk")
    return report


def _prepare_data(job):
    from src.data.manifest import prepare_dataset
    from src.provenance import file_hash
    root, directory = Path(job["settings"]["data_root"]), Path(job["job_dir"]) / "data"
    if not root.is_dir():
        raise RuntimeError(f"Dataset directory does not exist: {root}")
    if directory.exists():
        # Recovery never regenerates data; preflight verifies these against live raw files.
        audit = json.loads((directory / "dataset_audit.json").read_text(encoding="utf-8"))
        if audit["manifest_sha256"] != file_hash(directory / "samples.csv"):
            raise RuntimeError("Saved manifest/audit hash mismatch; original data preserved")
    else:
        audit = prepare_dataset(root, directory, decode_images=True,
                                require_official_counts=True, normalize_path_case=True)
    print(f"Data audit: samples={audit['sample_count']}, passed={audit['passed']}, formal_ready={audit['formal_ready']}, status={audit['status']}")
    if not audit["passed"]:
        raise RuntimeError(f"Data audit failed: {audit['errors']}")
    if not audit["formal_ready"]:
        print("Development only: unresolved audit evidence remains; see data/dataset_audit.json. No formal/test evaluation.")
    return audit


def _prepare_folds(job):
    """Freeze only train+valid assignments; retain every official split column."""
    if job["settings"]["protocol"] != "dev_cv5":
        return None
    import csv
    from collections import Counter
    from src.config import CLASS_ORDER
    from src.data.manifest import read_manifest
    from src.data.splits import make_dev_folds, read_dev_folds, validate_dev_folds
    from src.provenance import file_hash, object_hash, write_json
    directory = Path(job["job_dir"]) / "data"
    manifest = directory / "samples.csv"
    path, meta_path = directory / "folds.csv", directory / "folds_meta.json"
    rows = read_manifest(manifest)
    split_seed = job["settings"]["split_seed"]
    assignments = make_dev_folds(rows, n_folds=5, seed=split_seed)
    assignments = validate_dev_folds(rows, assignments, split_seed=split_seed)
    lookup = {row["sample_id"]: row for row in rows}
    summaries = []
    for fold in range(5):
        valid = [a for a in assignments if a["dev_fold"] == fold]
        counts = Counter(lookup[a["sample_id"]]["diagnosis_name"] for a in valid)
        summaries.append({"fold": fold, "train_count": len(assignments) - len(valid),
                          "valid_count": len(valid), "valid_class_counts": {c: counts[c] for c in CLASS_ORDER}})
    if path.exists() or meta_path.exists():
        if not path.is_file() or not meta_path.is_file():
            raise RuntimeError("Saved folds are incomplete; preserve this job and create a new job directory")
        saved = validate_dev_folds(rows, read_dev_folds(path), split_seed=split_seed)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if saved != assignments or meta.get("folds_sha256") != file_hash(path) or \
                meta.get("manifest_sha256") != file_hash(manifest) or meta.get("split_seed") != split_seed:
            raise RuntimeError("Saved fold assignment/manifest/seed hash drift; original files preserved")
        if meta.get("assignments_hash") != object_hash(assignments):
            raise RuntimeError("Saved fold assignment inventory hash mismatch")
    else:
        temporary = path.with_suffix(".csv.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("sample_id", "lesion_id", "source_row_index", "dev_fold"))
                writer.writeheader()
                writer.writerows(assignments)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        meta = {"schema_version": 1, "protocol": "dev_cv5", "n_folds": 5, "split_seed": split_seed,
                "development_count": len(assignments), "folds_sha256": file_hash(path),
                "manifest_sha256": file_hash(manifest), "assignments_hash": object_hash(assignments),
                "patient_identity_verified": False, "test_assigned": False, "formal": False,
                "folds": summaries}
        write_json(meta_path, meta)
    print(f"Fixed five folds: {len(assignments)} development cases; split_seed={split_seed}; training seeds={job['settings']['seeds']}")
    return meta


def _weights(configs, *, offline, download):
    import torch
    from torchvision.models import get_weight
    from src.provenance import file_hash
    reports = []
    identifiers = sorted({cfg["model"]["pretrained_id"] for cfg in configs.values()
                          if cfg["model"]["pretrained_id"] not in (None, "none")})
    for identifier in identifiers:
        weights = get_weight(identifier)
        filename = Path(urlparse(weights.url).path).name
        path = Path(torch.hub.get_dir()) / "checkpoints" / filename
        expected_prefix = filename.rsplit("-", 1)[-1].split(".")[0]
        cache_disk = path.parent
        while not cache_disk.exists():
            cache_disk = cache_disk.parent
        cache_free = shutil.disk_usage(cache_disk).free
        cache_required = int(weights.meta["num_params"]) * 4 + 64 * 1024 ** 2
        if not path.is_file() and download and not offline:
            if cache_free < cache_required:
                raise RuntimeError(f"Insufficient pretrained cache disk space: {cache_disk}; require about {cache_required / GIB:.2f} GiB")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".download-" + uuid.uuid4().hex)
            print(f"Downloading pretrained weights: {weights.url}")
            try:
                torch.hub.download_url_to_file(weights.url, str(temporary), hash_prefix=expected_prefix)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        sha = file_hash(path) if path.is_file() else None
        if sha is not None and not sha.startswith(expected_prefix):
            raise RuntimeError(f"Pretrained cache checksum mismatch: {path}; replace the damaged cache explicitly")
        if sha is None and offline:
            raise RuntimeError(f"Offline pretrained weights missing: {path}")
        report = {"id": identifier, "url": weights.url, "cache_path": str(path), "sha256": sha,
                  "available": sha is not None, "download_budget_bytes": cache_required,
                  "cache_disk_free_bytes": shutil.disk_usage(cache_disk).free}
        reports.append(report)
        print("Pretrained weights:", json.dumps(report))
    return reports


def _runner_args(job, mode=None):
    argv = ["--suite", str(Path(job["job_dir"]) / "suite.yaml"), "--resume", "--only", *job["settings"]["only"]]
    for key, value in job["overrides"].items():
        argv.extend(["--set", key + "=" + json.dumps(value, ensure_ascii=False)])
    if mode:
        argv.append(mode)
    return argv


def _invoke_runner(job, mode=None):
    from scripts import run_experiments as runner
    if mode is None:
        return runner.main(_runner_args(job))
    # The complete resolved plan is saved on disk; avoid printing it repeatedly.
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        result = runner.main(_runner_args(job, mode))
    if result:
        print(captured.getvalue(), end="")
    return result


def _run(args, job_dir, job):
    from scripts.collect_results import collect_suite
    from src.provenance import utc_now, write_json
    # Validate existing suite/run identities before opening the log or changing status.
    if _invoke_runner(job, "--dry-run"):
        raise RuntimeError("Experiment plan/resume validation failed; original job preserved")
    plan = json.loads((Path(job["suite_dir"]) / "plan.json").read_text(encoding="utf-8"))
    print(f"Job: {job_dir}\nLog: {job_dir / 'train.log'}\nPlan: {job['suite_dir']}/plan.json\nRuns: {plan['run_count']}; {job['settings']['protocol']}, development, test sealed")
    if args.dry_run:
        print("Dry run: no ML import, data decoding, pretrained download or training performed.")
        print(f"Run this saved plan: bash scripts/train_server.sh --resume {job_dir}")
        return 0
    with (job_dir / "train.log").open("a", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(_Tee(sys.stdout, log)), contextlib.redirect_stderr(_Tee(sys.stderr, log)):
            status = {"started_at": utc_now(), "formal": False, "test_inference": False}
            try:
                write_json(job_dir / "status.json", {**status, "status": "checking"})
                environment = _environment(job["settings"])
                storage = _storage(job, plan)
                audit = _prepare_data(job)
                folds = _prepare_folds(job)
                # Check all actual configurations/data before any weight download or training.
                if _invoke_runner(job, "--preflight-only"):
                    raise RuntimeError("Dataset/configuration preflight failed")
                weights = _weights(plan["resolved_configs"], offline=job["settings"]["offline"], download=not args.preflight_only)
                write_json(job_dir / "preflight.json", {"environment": environment, "storage": storage,
                           "audit_passed": audit["passed"], "formal_ready": audit["formal_ready"],
                           "weights": weights, "folds": folds, "checked_at": utc_now()})
                if args.preflight_only:
                    write_json(job_dir / "status.json", {**status, "status": "preflight_passed", "finished_at": utc_now()})
                    print("Preflight passed; training has not started. Uncached weights download at training start.")
                    return 0
                write_json(job_dir / "status.json", {**status, "status": "training"})
                if _invoke_runner(job):
                    raise RuntimeError("Training suite failed; review train.log and resume from the saved job")
                collection = collect_suite(Path(job["suite_dir"]))
                if not collection["all_preregistered_seeds_complete"]:
                    raise RuntimeError("Result collection is incomplete")
                write_json(job_dir / "summary.json", {"suite_dir": job["suite_dir"], "job_dir": str(job_dir),
                    "runs_csv": str(Path(job["suite_dir"]) / "runs.csv"),
                    "aggregate_csv": str(Path(job["suite_dir"]) / "aggregate.csv"),
                    "run_count": plan["run_count"], "protocol": job["settings"]["protocol"],
                    "seed_metrics_csv": str(Path(job["suite_dir"]) / "seed_metrics.csv") if folds else None,
                    "oof_directory": str(Path(job["suite_dir"]) / "oof") if folds else None,
                    "formal": False, "test_inference": False, "collection": collection})
                write_json(job_dir / "status.json", {**status, "status": "complete", "finished_at": utc_now()})
                print(f"Completed development suite. Results: {job['suite_dir']}\nSummary: {job_dir / 'summary.json'}")
                return 0
            except BaseException as exc:
                write_json(job_dir / "status.json", {**status, "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                           "finished_at": utc_now(), "error": f"{type(exc).__name__}: {exc}"})
                print(f"Job failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                raise


def main(argv=None):
    args = parser().parse_args(argv)
    if sys.version_info < (3, 10):
        print("error: Python >= 3.10 is required", file=sys.stderr)
        return 2
    if os.name != "posix":
        print("error: This launcher requires a Linux/WSL server", file=sys.stderr)
        return 2
    print(f"Python interpreter: {sys.executable}", flush=True)
    old_term = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        raise KeyboardInterrupt("SIGTERM received; recover from the last complete epoch")

    signal.signal(signal.SIGTERM, terminate)
    try:
        job_dir, job = _resolve_job(args)
        with _lock(job_dir):
            return _run(args, job_dir, job)
    except KeyboardInterrupt:
        print("Interrupted. Resume the saved job from its last full epoch checkpoint.", file=sys.stderr)
        return 130
    except ModuleNotFoundError as exc:
        print(f"error: {exc}\nPython interpreter: {sys.executable}", file=sys.stderr)
        print("Activate your project environment (conda activate dlenv), then select it with "
              '--python "$(command -v python)".', file=sys.stderr)
        if exc.name == "yaml":
            command = shlex.join([sys.executable, "-m", "pip", "install", "PyYAML>=6,<7"])
            print(f"The yaml module is provided by PyYAML. Install into this interpreter:\n{command}",
                  file=sys.stderr)
        print("If setup did not finish, rerun bash scripts/setup_conda.sh --reuse "
              "in the intended compatible Conda environment.", file=sys.stderr)
        return 2
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        signal.signal(signal.SIGTERM, old_term)


if __name__ == "__main__":
    raise SystemExit(main())
