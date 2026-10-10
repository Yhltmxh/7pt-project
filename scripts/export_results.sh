#!/usr/bin/env bash
# Export saved artifacts without importing or changing the training implementation.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/.."
"${DERM7PT_PYTHON:-python3}" - "$@" <<'PY'
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
from datetime import datetime, timezone

parser = argparse.ArgumentParser(description="Export completed results with project-relative archive paths.")
parser.add_argument("--job-dir", type=Path, help="Relative job directory; default: latest completed outputs/server job")
parser.add_argument("--include-checkpoints", action="store_true", help="Include model weights (large archive)")
args = parser.parse_args()
project = Path.cwd().resolve()

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def inside(path):
    path = path.resolve()
    path.relative_to(project)
    return path

try:
    if args.job_dir:
        if args.job_dir.is_absolute():
            raise ValueError("--job-dir must be relative to the project")
        job = inside(project / args.job_dir)
    else:
        completed = [p.parent for p in (project / "outputs/server").glob("*/job.json")
                     if (p.parent / "status.json").is_file()
                     and read(p.parent / "status.json").get("status") == "complete"]
        if not completed:
            raise ValueError("No completed server job found; use --job-dir outputs/server/<job>")
        job = inside(max(completed, key=lambda p: (p / "status.json").stat().st_mtime_ns))
    metadata = read(job / "job.json")
    suite = Path(metadata["suite_dir"])
    suite = inside(suite if suite.is_absolute() else project / suite)
    if job == suite or job in suite.parents or suite in job.parents:
        raise ValueError("Job and suite directories must be separate")
    for directory in (job, suite):
        if read(directory / "status.json").get("status") != "complete":
            raise ValueError(f"Incomplete job/suite: {directory.relative_to(project)}")
    collection = read(suite / "collection_status.json")
    if collection.get("all_preregistered_seeds_complete") is not True:
        raise ValueError("Result collection is incomplete; preserve and inspect the original job")
    plan = read(suite / "plan.json")
    if not plan["runs"] or any(r.get("status") != "complete" for r in plan["runs"]):
        raise ValueError("Suite contains incomplete runs")
    for run in plan["runs"]:
        run_dir = Path(run["run_dir"])
        run_dir = inside(run_dir if run_dir.is_absolute() else project / run_dir)
        run_dir.relative_to(job / "runs")
        if read(run_dir / "status.json").get("status") != "complete":
            raise ValueError(f"Incomplete run: {run_dir.relative_to(project)}")
    inventory = {}
    for directory in (job, suite):
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Symlink requires inspection: {path.relative_to(project)}")
            if not path.is_file():
                continue
            relative = path.relative_to(project)
            if not args.include_checkpoints and (
                "checkpoints" in relative.parts or path.suffix in {".pt", ".pth", ".ckpt", ".safetensors"}
            ):
                continue
            with path.open("rb") as stream:
                digest = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                sha = digest.hexdigest()
            inventory[relative.as_posix()] = {"bytes": path.stat().st_size, "sha256": sha}
    export_dir = project / "outputs/exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = export_dir / f"derm7pt_{job.name}_{stamp}.tar.gz"
    report = {"job_dir": job.relative_to(project).as_posix(),
              "suite_dir": suite.relative_to(project).as_posix(),
              "include_checkpoints": args.include_checkpoints,
              "scope": "Saved analysis artifacts; checkpoint/data/source identity not revalidated",
              "files": inventory}
    try:
        with tarfile.open(archive, "x:gz") as package:
            for name in inventory:
                package.add(project / name, arcname=name, recursive=False)
            content = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode()
            info = tarfile.TarInfo("export_manifest.json")
            info.size = len(content)
            package.addfile(info, io.BytesIO(content))
    except BaseException:
        archive.unlink(missing_ok=True)
        raise
    print(f"Job: {job.relative_to(project)}")
    print(f"Export: {archive.relative_to(project)}")
    print(f"Files: {len(inventory)}; checkpoint weights included: {args.include_checkpoints}")
except (OSError, ValueError, KeyError, TypeError) as exc:
    parser.exit(2, f"error: {exc}\n")
PY
