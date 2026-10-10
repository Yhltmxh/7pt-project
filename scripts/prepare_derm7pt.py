"""Prepare official Derm7pt manifests without copying or modifying raw data."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.manifest import prepare_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--root", type=Path, help="Explicit Derm7pt root containing meta/ and images/")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-decode", action="store_true", help="Write a partial audit; never marks the audit passed")
    parser.add_argument("--allow-nonofficial-counts", action="store_true", help="For fixtures or explicitly changed data only")
    parser.add_argument("--normalize-path-case", action="store_true", help="Resolve uniquely matching case differences, preserving raw source references and recording repairs")
    args = parser.parse_args(argv)
    config = {}
    if args.config:
        from src.config import load_config
        config = load_config(args.config)
    data = config.get("dataset", {})
    root = args.root or data.get("root")
    if not root:
        parser.error("Set --root or dataset.root; no dataset root is guessed")
    output = args.output_dir or (Path(data["manifest"]).parent if data.get("manifest") else Path("data/manifests"))
    audit = prepare_dataset(root, output, not args.skip_decode, not args.allow_nonofficial_counts, normalize_path_case=args.normalize_path_case)
    print(json.dumps({"output_dir": str(output), "samples": audit["sample_count"], "status": audit["status"], "passed": audit["passed"], "formal_ready": audit["formal_ready"], "errors": audit["errors"], "decoded": audit["image_decoding"]["performed"]}, ensure_ascii=False, indent=2))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
