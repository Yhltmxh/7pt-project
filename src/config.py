"""Strict, torch-independent configuration loading for the first baseline release."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_ORDER = ["BCC", "NEV", "MEL", "MISC", "SK"]


class ConfigError(ValueError):
    """A configuration does not satisfy the supported execution contract."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ConfigError("YAML mapping keys must be strings")
        if key in result:
            raise ConfigError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def read_yaml(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.load(stream, Loader=_UniqueKeyLoader)
    if not isinstance(value, dict):
        raise ConfigError(f"Expected a YAML mapping: {path}")
    return value


def resolve_path(value: str | Path, root: str | Path | None = None) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(root or PROJECT_ROOT) / path


def deep_merge(base: dict, overrides: dict, *, allow_new_keys: bool = False, _prefix: str = "") -> dict:
    """Return a new recursively merged dictionary; reject unknown fields by default."""
    if not isinstance(base, dict) or not isinstance(overrides, dict):
        raise ConfigError("deep_merge requires two mappings")
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        name = f"{_prefix}.{key}" if _prefix else str(key)
        if key not in base:
            if not allow_new_keys:
                raise ConfigError(f"Unknown configuration key: {name}")
            result[key] = copy.deepcopy(value)
        elif isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ConfigError(f"Expected mapping at {name}")
            result[key] = deep_merge(base[key], value, allow_new_keys=allow_new_keys, _prefix=name)
        else:
            if isinstance(value, dict):
                raise ConfigError(f"Unexpected mapping at {name}")
            result[key] = copy.deepcopy(value)
    return result


def apply_overrides(cfg: dict, overrides: dict | None) -> dict:
    """Apply nested or dotted overrides without allowing schema expansion."""
    result = copy.deepcopy(cfg)
    if not overrides:
        return result
    for key, value in overrides.items():
        if not isinstance(key, str):
            raise ConfigError("Override keys must be strings")
        parts = key.split(".")
        current = result
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                raise ConfigError(f"Unknown configuration key: {key}")
            current = current[part]
        leaf = parts[-1]
        if leaf not in current:
            raise ConfigError(f"Unknown configuration key: {key}")
        if isinstance(current[leaf], dict):
            if not isinstance(value, dict):
                raise ConfigError(f"Expected mapping at {key}")
            current[leaf] = deep_merge(current[leaf], value, _prefix=key)
        elif isinstance(value, dict):
            raise ConfigError(f"Unexpected mapping at {key}")
        else:
            current[leaf] = copy.deepcopy(value)
    return result


def _default_config() -> dict:
    return read_yaml(PROJECT_ROOT / "configs/base.yaml")


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    """Merge a full or partial config into the documented schema and validate it."""
    cfg = deep_merge(_default_config(), read_yaml(resolve_path(path)))
    cfg = apply_overrides(cfg, overrides)
    validate_config(cfg)
    return cfg


def config_hash(cfg: dict) -> str:
    content = json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def save_config(cfg: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _one_of(value: Any, allowed: tuple | list, name: str) -> None:
    if value not in allowed:
        raise ConfigError(f"{name} must be one of {list(allowed)}; got {value!r}")


def _number(value: Any, name: str, *, positive: bool = False, integer: bool = False,
            nullable: bool = False, minimum: float | None = None, maximum: float | None = None) -> None:
    if value is None and nullable:
        return
    expected = int if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, expected) or not math.isfinite(value):
        raise ConfigError(f"{name} must be a finite {'integer' if integer else 'number'}")
    if positive and value <= 0:
        raise ConfigError(f"{name} must be positive")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{name} must be at most {maximum}")


def _boolean(value: Any, name: str) -> None:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be a boolean")


def _require_fields(schema: dict, value: dict, prefix: str = "") -> None:
    if not isinstance(value, dict):
        raise ConfigError(f"Expected mapping at {prefix or 'configuration'}")
    missing = set(schema) - set(value)
    if missing:
        raise ConfigError(f"Missing fields at {prefix or 'configuration'}: {sorted(missing)}")
    for key, default in schema.items():
        if isinstance(default, dict):
            _require_fields(default, value[key], f"{prefix}.{key}" if prefix else key)


def validate_config(cfg: dict, require_ready: bool = False) -> None:
    """Validate supported behavior; readiness also requires a real budget and root.

    Drafts preserve missing data/budget decisions. dev_cv5 is a development-only
    fixed five-fold protocol; final evaluation still uses official_holdout.
    """
    defaults = _default_config()
    deep_merge(defaults, cfg)  # Unknown nested fields are always rejected.
    _require_fields(defaults, cfg)
    if type(cfg["schema_version"]) is not int or cfg["schema_version"] != 1:
        raise ConfigError("Only schema_version=1 is supported")
    _one_of(cfg["stage"], ("draft", "development", "frozen"), "stage")
    ready = require_ready or cfg["stage"] == "frozen"
    exp, data, pre, model, loss, train, evaluation = (cfg[key] for key in
        ("experiment", "dataset", "preprocess", "model", "loss", "train", "evaluation"))
    for field in ("id", "suite", "output_root"):
        if not isinstance(exp[field], str) or not exp[field].strip():
            raise ConfigError(f"experiment.{field} must be a nonempty string")
    if not isinstance(exp["seeds"], list) or not exp["seeds"]:
        raise ConfigError("experiment.seeds must contain unique integer seeds")
    for seed in exp["seeds"]:
        _number(seed, "experiment.seeds", integer=True, minimum=0)
    if len(set(exp["seeds"])) != len(exp["seeds"]):
        raise ConfigError("experiment.seeds must contain unique integer seeds")
    _one_of(data["name"], ("derm7pt",), "dataset.name")
    _one_of(data["split_protocol"], ("official_holdout", "dev_cv5"),
            "dataset.split_protocol (other CV protocols/refit are not implemented)")
    cv = data["split_protocol"] == "dev_cv5"
    if cv and cfg["stage"] != "development":
        raise ConfigError("dev_cv5 requires stage=development; CV is not final frozen evidence")
    if data["num_classes"] != 5:
        raise ConfigError("dataset.num_classes must be 5")
    if not cv:
        if data["folds_file"] is not None:
            raise ConfigError("folds_file is unsupported for official_holdout")
        if data["cv_fold"] is not None:
            raise ConfigError("cv_fold must be null for official_holdout")
    else:
        if data["folds_file"] is not None and (not isinstance(data["folds_file"], str) or not data["folds_file"].strip()):
            raise ConfigError("dataset.folds_file must be a nonempty path string or null")
        if data["cv_fold"] is not None:
            _number(data["cv_fold"], "dataset.cv_fold", integer=True, minimum=0, maximum=4)
    _number(data["split_seed"], "dataset.split_seed", integer=True, minimum=0)
    _one_of(data["clinical_substitution_policy"], (None, "retain_official"),
            "dataset.clinical_substitution_policy")
    if ready:
        required = []
        if not isinstance(data["root"], str) or not data["root"].strip():
            required.append("dataset.root")
        if data["clinical_substitution_policy"] is None:
            required.append("dataset.clinical_substitution_policy")
        if train["epochs"] is None:
            required.append("train.epochs")
        if cv and data["folds_file"] is None:
            required.append("dataset.folds_file")
        if required:
            raise ConfigError("Required before training/preflight: " + ", ".join(required))
    elif data["root"] is not None and not isinstance(data["root"], str):
        raise ConfigError("dataset.root must be a path string or null")
    for field in ("manifest", "label_map"):
        if not isinstance(data[field], str) or not data[field].strip():
            raise ConfigError(f"dataset.{field} must be a nonempty path string")
    if data["metadata_fields"] != []:
        raise ConfigError("Metadata input is not implemented in the image baseline release")
    _one_of(data["unknown_category"], ("explicit_unknown",), "dataset.unknown_category")
    _number(pre["image_size"], "preprocess.image_size", positive=True, integer=True)
    _one_of(pre["normalization"], ("imagenet",), "preprocess.normalization")
    _one_of(pre["geometry_coupling"], ("shared", "independent"), "preprocess.geometry_coupling")
    _one_of(pre["appearance_coupling"], ("none", "shared"), "preprocess.appearance_coupling")
    _one_of(pre["train_augmentation"], ("flips", "none", "isic2025"), "preprocess.train_augmentation")
    if pre["train_augmentation"] == "isic2025":
        if pre["geometry_coupling"] != "shared" or pre["appearance_coupling"] != "shared":
            raise ConfigError("isic2025 augmentation requires shared geometry and appearance")
    elif pre["appearance_coupling"] != "none":
        raise ConfigError("flips/none augmentation requires appearance_coupling=none")
    aug = pre["isic2025"]
    for field in ("horizontal_flip_probability", "vertical_flip_probability", "affine_probability",
                  "crop_probability", "brightness_contrast_probability", "hsv_probability", "blur_noise_probability"):
        _number(aug[field], f"preprocess.isic2025.{field}", minimum=0, maximum=1)
    for field, upper in (("translate_percent", 1), ("rotate", 180), ("brightness_limit", 1),
                         ("contrast_limit", 1), ("hue_shift_limit", 180),
                         ("saturation_shift_limit", 255), ("value_shift_limit", 255)):
        _number(aug[field], f"preprocess.isic2025.{field}", minimum=0, maximum=upper)
    for field, upper in (("scale", None), ("crop_scale", 1), ("crop_ratio", None), ("noise_std_range", 1)):
        values = aug[field]
        if not isinstance(values, list) or len(values) != 2:
            raise ConfigError(f"preprocess.isic2025.{field} must be a two-number range")
        for value in values:
            _number(value, f"preprocess.isic2025.{field}", positive=field != "noise_std_range", minimum=0, maximum=upper)
        if values[0] > values[1]:
            raise ConfigError(f"preprocess.isic2025.{field} must be ordered")
    _number(aug["blur_limit"], "preprocess.isic2025.blur_limit", integer=True, minimum=3)
    if aug["blur_limit"] % 2 != 1:
        raise ConfigError("preprocess.isic2025.blur_limit must be odd")
    _one_of(pre["inference_augmentation"], ("none",), "preprocess.inference_augmentation")
    modalities = model["modalities"]
    if modalities not in (["clinical"], ["dermoscopic"], ["clinical", "dermoscopic"]):
        raise ConfigError("model.modalities must be clinical, dermoscopic, or both in that order")
    _one_of(model["fusion"], ("single", "mean", "concat", "gated"), "model.fusion")
    if (len(modalities) == 1) != (model["fusion"] == "single"):
        raise ConfigError("Single-modality models require fusion=single; dual models require a fusion")
    for field in ("clinical_encoder", "dermoscopic_encoder"):
        _one_of(model[field], ("convnext_base", "tiny_cnn"), f"model.{field}")
        if model[field] == "tiny_cnn" and cfg["stage"] == "frozen":
            raise ConfigError("tiny_cnn is a development smoke encoder, not a frozen baseline")
    _one_of(model["pretrained_id"], (None, "none", "ConvNeXt_Base_Weights.IMAGENET1K_V1"), "model.pretrained_id")
    if any(model[f"{modality}_encoder"] == "tiny_cnn" for modality in modalities) and model["pretrained_id"] not in (None, "none"):
        raise ConfigError("tiny_cnn requires pretrained_id null/none")
    _boolean(model["independent_encoders"], "model.independent_encoders")
    if not model["independent_encoders"]:
        raise ConfigError("The baseline requires independent encoders")
    if model["feature_dim"] is not None:
        _number(model["feature_dim"], "model.feature_dim", positive=True, integer=True)
    if model["cross_attention_direction"] is not None:
        raise ConfigError("Cross attention is not implemented")
    for field in ("metadata_attention", "metadata_late_fusion", "metadata_tokens", "attribute_head"):
        _boolean(model[field], f"model.{field}")
        if model[field]:
            raise ConfigError(f"model.{field} is not implemented in the baseline release")
    _one_of(model["refinement"], ("none",), "model.refinement")
    _number(model["layer_scale_init"], "model.layer_scale_init", minimum=0)
    if model["mechanism"] != {"name": "none", "parameters": {}}:
        raise ConfigError("Mechanisms are not implemented; mechanism must remain disabled")
    _one_of(loss["diagnosis"], ("cross_entropy",), "loss.diagnosis")
    _one_of(loss["class_weights_source"], ("train_partition_only",), "loss.class_weights_source")
    if loss["focal_gamma"] is not None or loss["cb_beta"] is not None:
        raise ConfigError("Focal parameters are unsupported by the cross entropy baseline")
    _one_of(loss["attribute_type"], ("categorical_cross_entropy",), "loss.attribute_type")
    _one_of(loss["attribute_reduction"], ("mean_valid_tasks",), "loss.attribute_reduction")
    if loss["attribute_lambda"] != 0 or loss["mechanism_lambda"] != 0:
        raise ConfigError("Auxiliary/mechanism losses must be disabled")
    _number(train["epochs"], "train.epochs", positive=True, integer=True, nullable=not ready)
    _one_of(train["selection_protocol"], ("fixed",), "train.selection_protocol")
    _one_of(train["checkpoint_epoch"], ("final",), "train.checkpoint_epoch")
    for field in ("batch_size", "gradient_accumulation"):
        _number(train[field], f"train.{field}", positive=True, integer=True)
    _number(train["num_workers"], "train.num_workers", integer=True, minimum=0)
    _number(train["prefetch_factor"], "train.prefetch_factor", integer=True, minimum=1)
    for field in ("pin_memory", "persistent_workers", "progress_bar"):
        _boolean(train[field], f"train.{field}")
    if train["persistent_workers"]:
        raise ConfigError("persistent_workers must remain false for reproducible epoch-boundary recovery")
    _one_of(train["optimizer"], ("adamw",), "train.optimizer")
    for field in ("backbone_learning_rate", "head_learning_rate", "min_learning_rate", "max_grad_norm"):
        _number(train[field], f"train.{field}", positive=True)
    _number(train["weight_decay"], "train.weight_decay", minimum=0)
    _number(train["warmup_epochs"], "train.warmup_epochs", integer=True, minimum=0)
    if train["epochs"] is not None and train["warmup_epochs"] >= train["epochs"]:
        raise ConfigError("train.warmup_epochs must be smaller than train.epochs")
    if train["min_learning_rate"] > min(train["backbone_learning_rate"], train["head_learning_rate"]):
        raise ConfigError("train.min_learning_rate cannot exceed the initial learning rates")
    _one_of(train["scheduler"], ("warmup_cosine",), "train.scheduler")
    _one_of(train["device"], ("auto", "cpu", "cuda"), "train.device")
    _boolean(train["mixed_precision"], "train.mixed_precision")
    _boolean(train["ema"]["enabled"], "train.ema.enabled")
    _number(train["ema"]["decay"], "train.ema.decay", positive=True, maximum=1)
    if train["ema"]["decay"] >= 1:
        raise ConfigError("train.ema.decay must be smaller than 1")
    _one_of(evaluation["main_metric"], ("diagnosis_macro_f1",), "evaluation.main_metric")
    _one_of(evaluation["decision_rule"], ("argmax",), "evaluation.decision_rule")
    _one_of(evaluation["checkpoint_type"], ("ema", "regular"), "evaluation.checkpoint_type")
    if evaluation["checkpoint_type"] == "ema" and not train["ema"]["enabled"]:
        raise ConfigError("EMA evaluation requires train.ema.enabled")
    for field in ("tta", "aggregate_seeds"):
        _boolean(evaluation[field], f"evaluation.{field}")
        if evaluation[field]:
            raise ConfigError(f"evaluation.{field} is not implemented in the baseline release")
    if evaluation["class_order"] != CLASS_ORDER:
        raise ConfigError(f"evaluation.class_order must equal {CLASS_ORDER}")
    _number(evaluation["bootstrap_iterations"], "evaluation.bootstrap_iterations", positive=True, integer=True)
    _number(evaluation["bootstrap_seed"], "evaluation.bootstrap_seed", integer=True, minimum=0)
    _one_of(evaluation["bootstrap_unit"], ("case",), "evaluation.bootstrap_unit (patient/lesion identity is unverified)")
    for key, value in cfg["provenance"].items():
        _boolean(value, f"provenance.{key}")
    if cfg["provenance"]["legacy_checkpoint_allowed"]:
        raise ConfigError("Legacy checkpoints are unsupported")
    for key in ("require_source_hash", "require_data_hash", "require_resolved_config"):
        if not cfg["provenance"][key]:
            raise ConfigError(f"provenance.{key} must remain enabled")
