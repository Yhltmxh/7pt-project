"""Recheck manifest structure, image content, and exact cross-split repeats."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.manifest import audit_manifest, read_manifest, sha256_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument("--allow-nonofficial-counts", action="store_true")
    args = parser.parse_args(argv)
    report = audit_manifest(read_manifest(args.manifest), args.root, not args.skip_decode, not args.allow_nonofficial_counts)
    report["manifest_sha256"] = sha256_file(args.manifest)
    output = args.output or args.manifest.parent / "dataset_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": report["status"], "passed": report["passed"], "formal_ready": report["formal_ready"], "errors": report["errors"], "decoding_performed": report["image_decoding"]["performed"]}, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
