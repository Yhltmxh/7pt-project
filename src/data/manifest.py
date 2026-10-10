"""Build and audit a provenance-preserving Derm7pt sample manifest."""

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath

from .labels import ATTRIBUTE_CLASSES, DIAGNOSIS_CLASSES, encode_attribute, encode_diagnosis, label_contract
from .splits import OFFICIAL_COUNTS, OFFICIAL_PARTITIONS, read_official_indexes, validate_official_indexes

MODALITIES = ("clinical", "dermoscopic")
_BOOL_FIELDS = ("has_clinical", "has_dermoscopic") + tuple(f"attr_mask_{name}" for name in ATTRIBUTE_CLASSES)
_INT_FIELDS = ("source_row_index", "diagnosis_index") + tuple(f"attr_{name}" for name in ATTRIBUTE_CLASSES)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_relative_path(value):
    if value is None or not str(value).strip():
        return ""
    raw = str(value).strip().replace("\\", "/")
    path = PurePosixPath(raw)
    if path.is_absolute() or ":" in raw or ".." in path.parts:
        raise ValueError(f"Image path must stay relative to dataset root: {value!r}")
    if not path.parts or path.parts[0] != "images":
        path = PurePosixPath("images") / path
    return path.as_posix()


def resolve_image_path(root, value):
    root = Path(root).resolve()
    relative = image_relative_path(value)
    if not relative:
        return None
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Image path escapes dataset root: {value!r}")
    return candidate


def _check_path_spelling(root, relative, directory_cache):
    """Find case differences without repairing paths on case-insensitive mounts."""
    directory = Path(root)
    differences = []
    for part in PurePosixPath(relative).parts:
        key = str(directory)
        if key not in directory_cache:
            directory_cache[key] = {entry.name for entry in directory.iterdir()} if directory.is_dir() else set()
        names = directory_cache[key]
        if part not in names:
            matches = sorted(name for name in names if name.casefold() == part.casefold())
            if matches:
                differences.append({"requested": part, "on_disk": matches})
                directory /= matches[0]
                continue
        directory /= part
    return differences


def normalize_image_path_case(root, value, directory_cache=None):
    """Resolve spelling only when every path segment has one casefold match.

    This function never changes files. Ambiguous names are an error even on
    filesystems where one spelling currently happens to resolve.
    """
    relative = image_relative_path(value)
    if not relative:
        return ""
    directory = Path(root)
    directory_cache = {} if directory_cache is None else directory_cache
    resolved_parts = []
    for part in PurePosixPath(relative).parts:
        if not directory.is_dir():
            raise ValueError(f"Cannot normalize image path; directory missing: {directory}")
        key = str(directory)
        if key not in directory_cache:
            directory_cache[key] = tuple(entry.name for entry in directory.iterdir())
        matches = sorted(name for name in directory_cache[key] if name.casefold() == part.casefold())
        if len(matches) != 1:
            raise ValueError(f"Image path case normalization requires a unique match for {part!r}: {matches}")
        resolved_parts.append(matches[0])
        directory /= matches[0]
    resolved = PurePosixPath(*resolved_parts).as_posix()
    candidate = resolve_image_path(root, resolved)
    if candidate is None or not candidate.is_file():
        raise ValueError(f"Normalized image path is not a file: {resolved}")
    return resolved


def _parse_bool(value, field):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in ("true", "1"):
        return True
    if normalized in ("false", "0"):
        return False
    raise ValueError(f"Invalid boolean in {field}: {value!r}")


def _parse_integer(value, field):
    if isinstance(value, (bool, float)):
        raise ValueError(f"Invalid integer in {field}: {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid integer in {field}: {value!r}") from error


def validate_manifest_rows(rows):
    required = {"sample_id", "lesion_id", "source_row_index", "official_split", "clinical_path", "dermoscopic_path", "source_clinical_path", "source_dermoscopic_path", "has_clinical", "has_dermoscopic", "diagnosis_raw", "diagnosis_index", "diagnosis_name"}
    required.update(f"attr_{name}" for name in ATTRIBUTE_CLASSES)
    required.update(f"attr_mask_{name}" for name in ATTRIBUTE_CLASSES)
    seen_ids, seen_positions = set(), set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Every manifest sample must be a dictionary")
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Manifest fields missing: {sorted(missing)}")
        for field in _INT_FIELDS:
            if isinstance(row[field], bool) or not isinstance(row[field], int):
                raise ValueError(f"Manifest {field} must be an integer")
        for field in _BOOL_FIELDS:
            if not isinstance(row[field], bool):
                raise ValueError(f"Manifest {field} must be a boolean")
        if not row["sample_id"] or not row["lesion_id"]:
            raise ValueError("sample_id and lesion_id must be nonempty")
        if row["sample_id"] in seen_ids or row["source_row_index"] in seen_positions:
            raise ValueError(f"Duplicate sample or original row: {row['sample_id']}")
        seen_ids.add(row["sample_id"])
        seen_positions.add(row["source_row_index"])
        if row["source_row_index"] < 0 or row["official_split"] not in OFFICIAL_PARTITIONS:
            raise ValueError(f"Invalid source position or official partition: {row['sample_id']}")
        label = encode_diagnosis(row["diagnosis_raw"])
        if row["diagnosis_index"] != label or row["diagnosis_name"] != DIAGNOSIS_CLASSES[label]:
            raise ValueError(f"Diagnosis mapping mismatch: {row['sample_id']}")
        for modality in MODALITIES:
            path = row[f"{modality}_path"]
            if path and image_relative_path(path) != path:
                raise ValueError(f"Manifest path must start with images/: {path!r}")
            if row[f"has_{modality}"] and not path:
                raise ValueError(f"Available {modality} has no path: {row['sample_id']}")
        for name, classes in ATTRIBUTE_CLASSES.items():
            value, valid = row[f"attr_{name}"], row[f"attr_mask_{name}"]
            if (valid and not 0 <= value < len(classes)) or (not valid and value != -1):
                raise ValueError(f"Invalid attribute target/mask: {row['sample_id']} {name}")
    return rows


def read_manifest(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        if path.suffix.lower() == ".json":
            rows = json.load(stream)
        else:
            rows = list(csv.DictReader(stream))
    if not isinstance(rows, list):
        raise ValueError("Manifest must contain a list of sample rows")
    for row in rows:
        for field in _INT_FIELDS:
            if field in row:
                row[field] = _parse_integer(row[field], field)
        for field in _BOOL_FIELDS:
            if field in row:
                row[field] = _parse_bool(row[field], field)
        for field in ("patient_id", "dev_fold", "exclusion_reason"):
            if row.get(field) == "":
                row[field] = None
        if row.get("dev_fold") is not None:
            row["dev_fold"] = _parse_integer(row["dev_fold"], "dev_fold")
        if row.get("is_clinical_substituted") not in (None, "", "unknown"):
            row["is_clinical_substituted"] = _parse_bool(row["is_clinical_substituted"], "is_clinical_substituted")
        else:
            row["is_clinical_substituted"] = "unknown"
    return validate_manifest_rows(rows)


def write_manifest(rows, path):
    validate_manifest_rows(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        if not rows:
            raise ValueError("Cannot write an empty manifest")
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def build_manifest(root, require_official_counts=True, normalize_path_case=False):
    """Preserve original CSV row order before associating official positions."""
    root = Path(root)
    metadata_path = root / "meta" / "meta.csv"
    with metadata_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = {"case_num", "diagnosis", "clinic", "derm"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Official meta.csv fields missing: {sorted(missing)}")
        metadata = list(reader)
    index_paths = {name: root / "meta" / f"{name}_indexes.csv" for name in OFFICIAL_PARTITIONS}
    indexes = {name: read_official_indexes(path) for name, path in index_paths.items()}
    positions = validate_official_indexes(indexes, len(metadata), require_official_counts)
    case_nums = [row["case_num"].strip() for row in metadata]
    if any(not case_num for case_num in case_nums) or len(set(case_nums)) != len(case_nums):
        raise ValueError("Official case_num must be present and unique; cannot infer a lesion ID")
    rows = []
    directory_cache = {}
    for index, source in enumerate(metadata):
        diagnosis = encode_diagnosis(source["diagnosis"])
        case_num = source["case_num"].strip()
        row = {
            "sample_id": f"derm7pt_case_{case_num}", "source_row_index": index,
            "case_num": case_num, "case_id": source.get("case_id", ""),
            "patient_id": None, "lesion_id": f"derm7pt_case_{case_num}",
            "official_split": positions[index],
            "source_clinical_path": source["clinic"],
            "source_dermoscopic_path": source["derm"],
            "clinical_path": image_relative_path(source["clinic"]),
            "dermoscopic_path": image_relative_path(source["derm"]),
            "has_clinical": False, "has_dermoscopic": False,
            "is_clinical_substituted": "unknown",
            "clinical_sha256": "", "dermoscopic_sha256": "",
            "diagnosis_raw": source["diagnosis"], "diagnosis_name": DIAGNOSIS_CLASSES[diagnosis], "diagnosis_index": diagnosis,
            "sex": source.get("sex", ""), "location": source.get("location", ""), "elevation": source.get("elevation", ""),
            "dev_fold": None, "exclusion_reason": None,
        }
        for modality in MODALITIES:
            if normalize_path_case:
                row[f"{modality}_path"] = normalize_image_path_case(root, row[f"source_{modality}_path"], directory_cache)
            path = resolve_image_path(root, row[f"{modality}_path"])
            row[f"has_{modality}"] = path is not None and path.is_file()
        for name in ATTRIBUTE_CLASSES:
            value, valid = encode_attribute(name, source.get(name))
            row[f"attr_{name}"] = value
            row[f"attr_mask_{name}"] = valid
        rows.append(row)
    provenance = {"meta_sha256": sha256_file(metadata_path), "index_sha256": {name: sha256_file(path) for name, path in index_paths.items()}, "source_rows": len(metadata), "index_semantics": "zero-based positions in original meta.csv; no sorting or filtering"}
    return validate_manifest_rows(rows), provenance


def validate_manifest_against_source(rows, root, require_official_counts=True):
    """Verify every manifest row against original metadata and official indexes.

    Reviewed substitution fields can be added independently. Source positions,
    labels, relative paths, and original identifiers cannot be changed silently.
    Returns original meta/split hashes for preflight and audit provenance.
    """
    validate_manifest_rows(rows)
    expected, provenance = build_manifest(root, require_official_counts)
    if len(rows) != len(expected):
        raise ValueError(f"Manifest has {len(rows)} rows but original meta.csv has {len(expected)}")
    expected_by_position = {row["source_row_index"]: row for row in expected}
    fields = ("sample_id", "lesion_id", "case_num", "case_id", "official_split", "source_clinical_path", "source_dermoscopic_path", "diagnosis_raw", "diagnosis_index", "diagnosis_name", "sex", "location", "elevation")
    fields += tuple(f"attr_{name}" for name in ATTRIBUTE_CLASSES)
    fields += tuple(f"attr_mask_{name}" for name in ATTRIBUTE_CLASSES)
    for row in rows:
        position = row["source_row_index"]
        source = expected_by_position.get(position)
        if source is None:
            raise ValueError(f"Manifest source position is not in original meta.csv: {position}")
        for field in fields:
            if row.get(field) != source.get(field):
                raise ValueError(f"Manifest differs from original metadata/indexes at row {position}, field {field}")
        for modality in MODALITIES:
            original = source[f"{modality}_path"]
            if row[f"{modality}_path"] != original:
                normalized = normalize_image_path_case(root, source[f"source_{modality}_path"])
                if normalized != row[f"{modality}_path"]:
                    raise ValueError(f"Manifest image path differs from original and unique case normalization: row {position} {modality}")
    return provenance


def audit_manifest(rows, root, decode_images=True, require_official_counts=True):
    """Audit all partitions; hashes identify exact repeats, not substitutions."""
    validate_manifest_rows(rows)
    root = Path(root)
    image_module = None
    if decode_images:
        try:
            from PIL import Image
            image_module = Image
        except ImportError as error:
            raise RuntimeError("Image decoding requires Pillow. Install project dependencies or explicitly use --skip-decode for a partial audit.") from error
    errors, warnings, issues = [], [], []
    provenance = None
    try:
        provenance = validate_manifest_against_source(rows, root, require_official_counts)
    except (ValueError, OSError) as error:
        errors.append(f"Original source verification failed: {error}")
    indexes = {name: [row["source_row_index"] for row in rows if row["official_split"] == name] for name in OFFICIAL_PARTITIONS}
    try:
        validate_official_indexes(indexes, len(rows), require_official_counts)
        split_integrity = True
    except ValueError as error:
        split_integrity = False
        errors.append(str(error))
    hashes, paths = defaultdict(list), defaultdict(list)
    cache, directory_cache = {}, {}
    decoded, missing = 0, 0
    for row in rows:
        for modality in MODALITIES:
            relative = row[f"{modality}_path"]
            path = resolve_image_path(root, relative)
            reference = {"sample_id": row["sample_id"], "source_row_index": row["source_row_index"], "partition": row["official_split"], "modality": modality, "path": relative}
            spelling = _check_path_spelling(root, relative, directory_cache) if relative else []
            if spelling:
                errors.append(f"Image path case differs from disk: {row['sample_id']} {modality} {relative}")
                issues.append(dict(reference, issue="path_case_mismatch", detail=spelling))
            exists = path is not None and path.is_file()
            if exists != row[f"has_{modality}"]:
                errors.append(f"Availability changed: {row['sample_id']} {modality}")
            row[f"has_{modality}"] = exists
            if not exists:
                missing += 1
                issues.append(dict(reference, issue="missing_image"))
                errors.append(f"Missing image: {row['sample_id']} {modality} {relative}")
                continue
            key = str(path)
            if key not in cache:
                digest = sha256_file(path)
                decode_error = None
                if decode_images:
                    try:
                        with image_module.open(path) as image:
                            image.verify()
                        with image_module.open(path) as image:
                            image.convert("RGB").load()
                    except Exception as error:
                        decode_error = f"{type(error).__name__}: {error}"
                cache[key] = (digest, decode_error)
            digest, decode_error = cache[key]
            recorded = row.get(f"{modality}_sha256", "")
            if recorded and recorded != digest:
                errors.append(f"Image hash changed: {row['sample_id']} {modality}")
            row[f"{modality}_sha256"] = digest
            paths[key].append(reference)
            hashes[digest].append(reference)
            if decode_error:
                errors.append(f"Cannot decode image: {row['sample_id']} {modality}")
                issues.append(dict(reference, issue="decode_failed", detail=decode_error))
            elif decode_images:
                decoded += 1
    exact_groups = [references for references in hashes.values() if len(references) > 1]
    cross_partition = [references for references in exact_groups if len({item["partition"] for item in references}) > 1]
    cross_paths = [references for references in paths.values() if len({item["partition"] for item in references}) > 1]
    if cross_partition:
        errors.append(f"Exact image hashes occur across partitions: {len(cross_partition)} groups; resolve before formal training")
    if not decode_images:
        warnings.append("Image decoding was explicitly skipped; this audit is incomplete")
    warnings.append("Clinical substitution status is unknown unless independently documented; identical images do not prove substitution")
    warnings.append("Patient identity is not inferred from case_num or case_id; independence is lesion/case level only")
    warnings.append("Near-duplicate image search has not been run; only exact paths and SHA256 are audited")
    path_normalization_events = [
        {"sample_id": row["sample_id"], "source_row_index": row["source_row_index"], "modality": modality,
         "raw": row[f"source_{modality}_path"], "source_relative": image_relative_path(row[f"source_{modality}_path"]),
         "resolved": row[f"{modality}_path"], "sha256": row[f"{modality}_sha256"]}
        for row in rows for modality in MODALITIES
        if image_relative_path(row[f"source_{modality}_path"]) != row[f"{modality}_path"]
    ]
    summaries = {}
    for partition in OFFICIAL_PARTITIONS:
        subset = [row for row in rows if row["official_split"] == partition]
        summaries[partition] = {
            "count": len(subset),
            "diagnosis_counts": dict(Counter(row["diagnosis_name"] for row in subset)),
            "attributes": {name: {"counts": dict(Counter(str(row[f"attr_{name}"]) for row in subset if row[f"attr_mask_{name}"])), "missing": sum(not row[f"attr_mask_{name}"] for row in subset)} for name in ATTRIBUTE_CLASSES},
            "metadata_values": {name: dict(Counter(row.get(name) or "<missing>" for row in subset)) for name in ("sex", "location", "elevation")},
        }
    report = {
        "schema_version": 1, "root": str(root.resolve()), "sample_count": len(rows), "official_expected_counts": OFFICIAL_COUNTS,
        "provenance": provenance, "source_verified": provenance is not None,
        "split_integrity": split_integrity, "partitions": summaries,
        "image_decoding": {"performed": decode_images, "decoded_references": decoded, "missing_references": missing, "unique_files_hashed": len(cache)},
        "cross_partition_same_path": cross_paths, "cross_partition_same_hash": cross_partition,
        "exact_duplicate_groups": exact_groups, "image_issues": issues,
        "path_normalization_events": path_normalization_events,
        "clinical_substitution": {"unknown": sum(row.get("is_clinical_substituted") == "unknown" for row in rows), "true": sum(row.get("is_clinical_substituted") is True for row in rows), "false": sum(row.get("is_clinical_substituted") is False for row in rows)},
        "identity_evidence": "Official cases represent unique lesions; patient identity has not been verified",
        "near_duplicate_audit": "not_performed", "errors": errors, "warnings": warnings,
        "passed": not errors and decode_images,
    }
    substitutions_unknown = report["clinical_substitution"]["unknown"] > 0
    report["formal_ready"] = report["passed"] and not substitutions_unknown
    report["status"] = "passed" if report["formal_ready"] else ("needs_review" if errors or substitutions_unknown else "partial")
    return report


def prepare_dataset(root, output_dir, decode_images=True, require_official_counts=True, normalize_path_case=False):
    rows, provenance = build_manifest(root, require_official_counts, normalize_path_case)
    audit = audit_manifest(rows, root, decode_images, require_official_counts)
    audit["provenance"] = provenance
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    samples = output / "samples.csv"
    write_manifest(rows, samples)
    audit["manifest_sha256"] = sha256_file(samples)
    metadata_vocab = {
        "schema_version": 1, "input_enabled": False, "fit_partition": "train",
        "unknown_token": "<unknown>", "missing_token": "<missing>",
        "fields": {}, "excluded_fields": ["elevation"],
        "manifest_sha256": audit["manifest_sha256"],
    }
    training_rows = [row for row in rows if row["official_split"] == "train"]
    for field in ("sex", "location"):
        counts = Counter(str(row.get(field) or "<missing>") for row in training_rows)
        vocabulary = {"<unknown>": 0, "<missing>": 1}
        for token in sorted(set(counts) - set(vocabulary)):
            vocabulary[token] = len(vocabulary)
        metadata_vocab["fields"][field] = {"vocabulary": vocabulary, "training_counts": dict(counts)}
    for name, value in (("label_map.json", label_contract()), ("metadata_vocab.json", metadata_vocab), ("dataset_audit.json", audit)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit
