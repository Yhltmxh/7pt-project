#!/usr/bin/env bash
# One foreground process: Python owns logging, signals, locking and recovery.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${DERM7PT_PYTHON:-python3}"
args=()
while (($#)); do
    if [[ "$1" == "--python" ]]; then
        if (($# < 2)); then
            printf '%s\n' 'error: --python requires an interpreter path' >&2
            exit 2
        fi
        python_bin="$2"
        shift 2
    else
        args+=("$1")
        shift
    fi
done
if ! command -v -- "$python_bin" >/dev/null 2>&1; then
    printf 'error: Python interpreter not found: %s\n' "$python_bin" >&2
    exit 2
fi
cd -- "$script_dir/.."
exec "$python_bin" -u "$script_dir/train_server.py" "${args[@]}"
