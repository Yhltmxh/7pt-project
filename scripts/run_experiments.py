"""Expand the implemented baseline suite without touching the sealed test split."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Direct script execution and python -m scripts.run_experiments both work.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from src.config import (ConfigError, apply_overrides, config_hash, deep_merge, load_config,
                        read_yaml, resolve_path, save_config, validate_config)
from src.provenance import make_identity, run_name
from scripts.collect_results import verify_run

SUPPORTED_IDS = {"B01", "B02", "B04", "B05", "B06"}
ALLOWED_MODEL_DIFFERENCES = {"model.modalities", "model.fusion"}
SUITE_KEYS = {"suite_id", "base_config", "protocol_config", "seeds", "experiments"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _flatten_keys(mapping: dict, prefix: str = "") -> set[str]:
    result = set()
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten_keys(value, path))
        else:
            result.add(path)
    return result


def _validate_seeds(seeds: Any) -> list[int]:
    if not isinstance(seeds, list) or not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ConfigError("Suite seeds must be a nonempty list of nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ConfigError("Suite seeds must be unique")
    return seeds


def expand_suite(path: str | Path, *, only: list[str] | None = None,
                 seeds: list[int] | None = None, overrides: dict | None = None) -> dict:
    """Resolve inheritance in topological order, then select the requested runs."""
    suite = read_yaml(resolve_path(path))
    unknown = set(suite) - SUITE_KEYS
    missing = SUITE_KEYS - set(suite)
    if unknown or missing:
        raise ConfigError(f"Suite keys: unknown={sorted(unknown)}, missing={sorted(missing)}")
    if not isinstance(suite["suite_id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", suite["suite_id"]):
        raise ConfigError("suite_id must be a safe filename identifier")
    chosen_seeds = _validate_seeds(seeds if seeds is not None else suite["seeds"])
    base = load_config(suite["base_config"])
    if suite["protocol_config"] is not None:
        base = deep_merge(base, read_yaml(resolve_path(suite["protocol_config"])))
    base = apply_overrides(base, overrides)
    base["experiment"]["suite"] = suite["suite_id"]
    base["experiment"]["seeds"] = chosen_seeds
    validate_config(base)
    if not isinstance(suite["experiments"], list) or not suite["experiments"]:
        raise ConfigError("Suite experiments must be a nonempty list")
    specifications = {}
    for spec in suite["experiments"]:
        if not isinstance(spec, dict) or set(spec) != {"id", "parent", "overrides"}:
            raise ConfigError("Each experiment requires exactly id, parent and overrides")
        experiment_id = spec["id"]
        if experiment_id not in SUPPORTED_IDS:
            raise ConfigError(f"Experiment {experiment_id!r} is not implemented; supported IDs: {sorted(SUPPORTED_IDS)}")
        if experiment_id in specifications:
            raise ConfigError(f"Duplicate experiment ID: {experiment_id}")
        if not isinstance(spec["overrides"], dict):
            raise ConfigError(f"{experiment_id}.overrides must be a mapping")
        unexpected = _flatten_keys(spec["overrides"]) - ALLOWED_MODEL_DIFFERENCES
        if unexpected:
            raise ConfigError(f"Unregistered differences in {experiment_id}: {sorted(unexpected)}; use --set for shared protocol changes")
        specifications[experiment_id] = spec
    selected = set(only) if only is not None else set(specifications)
    unknown_selected = selected - set(specifications)
    if unknown_selected or not selected:
        raise ConfigError(f"Unknown or empty --only selection: {sorted(unknown_selected)}")
    resolved, visiting, order = {}, set(), []

    def visit(experiment_id: str) -> dict:
        if experiment_id in resolved:
            return resolved[experiment_id]
        if experiment_id in visiting:
            raise ConfigError(f"Cyclic experiment inheritance at {experiment_id}")
        if experiment_id not in specifications:
            raise ConfigError(f"Unknown experiment parent: {experiment_id}")
        visiting.add(experiment_id)
        spec = specifications[experiment_id]
        parent = spec["parent"]
        if not isinstance(parent, str):
            raise ConfigError(f"{experiment_id}.parent must be a string")
        cfg = apply_overrides(base if parent == "base" else visit(parent), spec["overrides"])
        cfg["experiment"]["id"] = experiment_id
        validate_config(cfg)
        visiting.remove(experiment_id)
        resolved[experiment_id] = cfg
        order.append(experiment_id)
        return cfg

    # Validate every registered entry, even one excluded by --only.
    for experiment_id in specifications:
        visit(experiment_id)
    cv = base["dataset"]["split_protocol"] == "dev_cv5"
    if cv and base["dataset"]["cv_fold"] is not None:
        raise ConfigError("A dev_cv5 suite requires cv_fold=null; the runner expands all five folds")
    experiments = []
    for experiment_id in order:
        if experiment_id not in selected:
            continue
        for fold in range(5) if cv else (-1,):
            cfg = copy.deepcopy(resolved[experiment_id])
            if cv:
                cfg["dataset"]["cv_fold"] = fold
            validate_config(cfg)
            experiments.append({"id": experiment_id, "parent": specifications[experiment_id]["parent"],
                                "fold": fold, "config_key": f"{experiment_id}_fold{fold}" if cv else experiment_id,
                                "config": cfg})
    return {"suite_id": suite["suite_id"], "seeds": chosen_seeds, "experiments": experiments,
            "training_partition": "official_train+valid_minus_heldout_fold" if cv else "train",
            "development_evaluation_partition": "dev_cv_oof" if cv else "valid",
            "test_inference": False}


def build_plan(expanded: dict) -> dict:
    plans = []
    for spec in expanded["experiments"]:
        cfg = spec["config"]
        cfg_hash = config_hash(cfg)
        for seed in expanded["seeds"]:
            provisional = hashlib.sha256(f"{cfg_hash}:{seed}".encode()).hexdigest()[:12]
            plans.append({"spec_id": spec["id"], "parent_id": spec["parent"], "seed": seed,
                          "fold": spec["fold"], "config_key": spec["config_key"],
                          "protocol": cfg["dataset"]["split_protocol"], "config_hash": cfg_hash,
                          "status": "planned", "run_dir": None,
                          "provisional_output_dir": str(resolve_path(cfg["experiment"]["output_root"]) /
                              run_name(cfg, seed, provisional))})
    return {"suite_id": expanded["suite_id"], "created_at": _now(), "seeds": expanded["seeds"],
            "run_count": len(plans), "training_partition": expanded["training_partition"],
            "evaluation_partition": expanded["development_evaluation_partition"],
            "test_inference": False, "reuse_supported": True, "resume_requires_explicit_flag": True,
            "note": "Provisional paths use config+seed only. Execution records full data/source identity. Existing runs require --resume and verified artifacts/checkpoints.",
            "resolved_configs": {item["config_key"]: item["config"] for item in expanded["experiments"]},
            "runs": plans}


def _parse_overrides(values: list[str]) -> dict:
    result = {}
    for value in values:
        key, separator, literal = value.partition("=")
        if not separator or not key:
            raise ConfigError(f"Expected --set KEY=YAML_VALUE, got {value!r}")
        if key in result:
            raise ConfigError(f"Duplicate --set key: {key}")
        result[key] = yaml.safe_load(literal)
    return result


IDENTITY_FIELDS = ("identity_hash", "config_hash", "source_hash", "manifest_hash", "label_map_hash", "raw_data_hash", "seed")
CV_IDENTITY_FIELDS = ("folds_hash", "split_hash", "cv_fold")


def _config_key(run: dict) -> str:
    fold = run.get("fold", -1)
    return run.get("config_key", run["spec_id"] if fold == -1 else f"{run['spec_id']}_fold{fold}")


def _identity_fields(identity: dict) -> tuple:
    return IDENTITY_FIELDS + CV_IDENTITY_FIELDS if identity.get("cv_fold") is not None else IDENTITY_FIELDS


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigError(f"Expected a JSON object: {path}")
    return value


def _attempted(plan: dict) -> bool:
    return any(run.get("status", "planned") != "planned" or
               (run.get("run_dir") and Path(run["run_dir"]).exists())
               for run in plan.get("runs", []))


def _check_same_plan(previous: dict, requested: dict) -> None:
    for field in ("suite_id", "seeds", "run_count", "resolved_configs"):
        if previous.get(field) != requested.get(field):
            raise ConfigError(f"Resume plan/configuration drift in {field}; original plan preserved")
    fields = ("spec_id", "parent_id", "seed", "protocol", "config_hash")
    def signature(run):
        return (*[run.get(key) for key in fields], run.get("fold", -1), _config_key(run))
    old_runs = [signature(run) for run in previous.get("runs", [])]
    new_runs = [signature(run) for run in requested["runs"]]
    if old_runs != new_runs:
        raise ConfigError("Resume experiment/seed/protocol drift; original plan preserved")


def _check_identity(saved: dict, current: dict, location: str) -> None:
    for field in _identity_fields(current):
        if saved.get(field) != current[field]:
            raise ConfigError(f"Resume identity drift in {field} ({location}); original artifacts preserved")


def _prepare_actions(plan: dict, configs: dict, allow_resume: bool) -> list[dict]:
    """Inspect every existing run before changing any suite or run artifact.

    Checkpoint payload/environment validation is additionally performed by the
    single training entry point before it changes the original run directory.
    """
    actions = []
    for run in plan["runs"]:
        cfg = configs[_config_key(run)]
        cv = cfg["dataset"]["split_protocol"] == "dev_cv5"
        expected_fold = cfg["dataset"]["cv_fold"] if cv else -1
        if run.get("fold", -1) != expected_fold:
            raise ConfigError("Run fold disagrees with its resolved configuration")
        if run["spec_id"] != cfg["experiment"]["id"]:
            raise ConfigError("Run spec_id disagrees with its resolved configuration")
        identity = make_identity(cfg, run["seed"], resolve_path(cfg["dataset"]["manifest"]),
                                 resolve_path(cfg["dataset"]["label_map"]))
        compact = {field: identity[field] for field in _identity_fields(identity)}
        canonical = resolve_path(cfg["experiment"]["output_root"]) / run_name(cfg, run["seed"], identity["identity_hash"])
        if run.get("identity") is not None:
            _check_identity(run["identity"], compact, f"plan {run['spec_id']} seed {run['seed']}")
        recorded = Path(run["run_dir"]) if run.get("run_dir") else canonical
        if recorded.resolve() != canonical.resolve():
            raise ConfigError(f"Resume run directory/identity drift: {recorded}; original artifacts preserved")
        action = "train"
        checkpoint = None
        if recorded.exists():
            if not allow_resume:
                raise ConfigError(f"Run directory already exists: {recorded}; use --resume for verified recovery")
            status = _read_json(recorded / "status.json")
            if status.get("status") not in ("complete", "failed", "training", "interrupted", "trained", "predicting"):
                raise ConfigError(f"Unsupported saved run status: {status.get('status')!r}")
            if run.get("status") == "complete" and status.get("status") != "complete":
                raise ConfigError("Suite declares complete but saved run status is incomplete; refusing partial reuse")
            if status.get("status") != "complete":
                checkpoint = recorded / "checkpoints/final_regular.pt"
                if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                    raise ConfigError(f"Cannot resume {recorded}: no final_regular.pt epoch checkpoint; keep this directory and start a new job identity")
            saved_cfg = load_config(recorded / "resolved_config.yaml")
            if saved_cfg != cfg:
                raise ConfigError(f"Resume saved configuration drift: {recorded}; original artifacts preserved")
            provenance = _read_json(recorded / "provenance.json")
            _check_identity(provenance, compact, str(recorded))
            if status.get("identity_hash") != compact["identity_hash"] or status.get("seed") != run["seed"]:
                raise ConfigError(f"Saved status identity/seed mismatch: {recorded}")
            if cv and status.get("fold") != expected_fold:
                raise ConfigError(f"Saved CV status fold mismatch: {recorded}")
            if status.get("status") == "complete":
                verify_run(recorded, run, cfg)
                action = "reuse_complete"
            else:
                action = "resume"
        elif run.get("status", "planned") != "planned":
            raise ConfigError(f"Attempted run directory is missing: {recorded}; refusing to restart silently")
        run.update(run_dir=str(canonical), identity=compact, next_action=action,
                   resume_checkpoint=str(checkpoint) if checkpoint is not None else None)
        actions.append({"action": action, "run_dir": canonical, "resume": checkpoint})
    plan["reuse_supported"] = True
    plan["resume_requires_explicit_flag"] = True
    return actions


def _save_plan_configs(suite_dir: Path, plan: dict) -> None:
    for spec_id, cfg in plan["resolved_configs"].items():
        save_config(cfg, suite_dir / "configs" / f"{spec_id}.yaml")
    _write_json(suite_dir / "plan.json", plan)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="configs/experiments/baselines.yaml")
    parser.add_argument("--only", nargs="+", help="Select registered IDs, e.g. B04 B05")
    parser.add_argument("--seeds", nargs="+", type=int, help="Override the common seed list")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Shared dotted config override")
    parser.add_argument("--resume", action="store_true", help="Verify/reuse complete runs and resume matching epoch checkpoints; preserve incomplete runs without checkpoints")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Write/print the resolved plan; does not import torch")
    mode.add_argument("--preflight-only", action="store_true", help="Verify prerequisites without training")
    args = parser.parse_args(argv)
    try:
        expanded = expand_suite(args.suite, only=args.only, seeds=args.seeds, overrides=_parse_overrides(args.set))
        requested = build_plan(expanded)
        suite_dir = PROJECT_ROOT / "outputs/suites" / expanded["suite_id"]
        plan_path = suite_dir / "plan.json"
        attempted = False
        plan = requested
        if plan_path.exists():
            previous = _read_json(plan_path)
            attempted = _attempted(previous)
            if attempted:
                if not args.resume:
                    raise ConfigError("This suite already has attempted runs. Use --resume for verified recovery or choose a new suite_id.")
                _check_same_plan(previous, requested)
                plan = copy.deepcopy(previous)
        configs = plan["resolved_configs"]
        if args.dry_run:
            if attempted:
                for cfg in configs.values():
                    validate_config(cfg, require_ready=True)
                _prepare_actions(plan, configs, allow_resume=True)
                # A resumed dry run must not rewrite the recorded attempt history.
            else:
                _save_plan_configs(suite_dir, plan)
            print(json.dumps(plan, indent=2, ensure_ascii=False))
            return 0
        # Refuse unresolved budgets and plan drift before importing the ML stack.
        for cfg in configs.values():
            validate_config(cfg, require_ready=True)
        from src.train import preflight, run_training
        checks = [{"config_key": key, "spec_id": cfg["experiment"]["id"],
                   "fold": cfg["dataset"]["cv_fold"] if cfg["dataset"]["split_protocol"] == "dev_cv5" else -1,
                   "result": preflight(cfg)} for key, cfg in configs.items()]
        actions = _prepare_actions(plan, configs, allow_resume=args.resume)
        # All identities and existing artifacts have passed before these writes.
        if not attempted:
            _save_plan_configs(suite_dir, plan)
        _write_json(suite_dir / "preflight.json", {"checked_at": _now(), "experiments": checks})
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        if args.preflight_only:
            print("Preflight passed; no training or test inference was run.")
            return 0
        _write_json(suite_dir / "status.json", {"status": "training", "started_at": _now(), "resume": args.resume})
        for run, action in zip(plan["runs"], actions):
            cfg = configs[_config_key(run)]
            latest = make_identity(cfg, run["seed"], resolve_path(cfg["dataset"]["manifest"]),
                                   resolve_path(cfg["dataset"]["label_map"]))
            _check_identity(run["identity"], latest, f"before {run['spec_id']} seed {run['seed']}")
            if action["action"] == "reuse_complete":
                run.update(status="complete", reused_at=_now())
                _write_json(plan_path, plan)
                print(f"Verified complete run reused: {run['run_dir']}", flush=True)
                continue
            run["status"] = "training"
            run.setdefault("started_at", _now())
            run["last_started_at"] = _now()
            # Persist the actual path before training so failures remain resumable.
            _write_json(plan_path, plan)
            try:
                result = run_training(copy.deepcopy(configs[_config_key(run)]), run["seed"],
                                      output_dir=action["run_dir"], resume=action["resume"])
                if Path(result).resolve() != action["run_dir"].resolve():
                    raise RuntimeError("Training returned an unexpected run directory")
                status = _read_json(Path(result) / "status.json")
                if status.get("status") != "complete":
                    raise RuntimeError(f"Training returned a run without complete status: {result}")
                _check_identity(_read_json(Path(result) / "provenance.json"), run["identity"], str(result))
                verify_run(Path(result), run, configs[_config_key(run)])
                run.update(status="complete", finished_at=_now())
                run.pop("error", None)
                _write_json(plan_path, plan)
            except BaseException as exc:
                run.update(status="failed", finished_at=_now(), error=f"{type(exc).__name__}: {exc}")
                _write_json(plan_path, plan)
                _write_json(suite_dir / "status.json", {"status": "failed", "finished_at": _now(),
                    "spec_id": run["spec_id"], "fold": run.get("fold", -1), "config_key": _config_key(run),
                    "seed": run["seed"], "error": run["error"]})
                raise
        _write_json(suite_dir / "status.json", {"status": "complete", "finished_at": _now(),
            "run_count": plan["run_count"], "reused_complete_runs": sum(item["action"] == "reuse_complete" for item in actions),
            "scope": "training and development fold predictions only" if plan["evaluation_partition"] == "dev_cv_oof"
                     else "training and valid prediction only"})
        return 0
    except (ConfigError, OSError, RuntimeError, ValueError, ImportError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
