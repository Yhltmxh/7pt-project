#!/usr/bin/env bash
# Conda owns Python; the matched PyTorch wheels come from the official index.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
env_name=dlenv
variant=cu130
reuse=false
dry_run=false
conda_bin="${CONDA_EXE:-conda}"

usage() {
    cat <<'HELP'
Usage: bash scripts/setup_conda.sh [options]
  --name NAME       Conda environment name (default: dlenv)
  --variant BUILD   cu130 (default), cu126, cu128, or cpu
  --reuse           Explicitly install into an existing compatible Python environment
  --dry-run         Print installation plan without Conda or downloads
  -h, --help        Show help

GPU: Python 3.13.9, torch 2.9.0 / torchvision 0.24.0 (matches local dlenv).
CPU: Python 3.12, torch 2.8.0 / torchvision 0.23.0 (validated CPU stack).
Installs requirements.txt and checks imports, dependencies and device operations.
Requires an existing Conda installation; does not install NVIDIA drivers.
HELP
}
die() { printf 'error: %s\n' "$*" >&2; exit 2; }
while (($#)); do
    case "$1" in
        --name|--variant)
            (($# >= 2)) || die "$1 requires a value"
            if [[ "$1" == --name ]]; then env_name="$2"; else variant="$2"; fi
            shift 2 ;;
        --reuse) reuse=true; shift ;;
        --dry-run) dry_run=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done
[[ "$env_name" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die 'Invalid environment name'
[[ "$env_name" != base && "$env_name" != root ]] || die 'Choose a dedicated project environment'
case "$variant" in
    cu126|cu128|cu130) python_spec=3.13.9; python_minor=3.13; torch_version=2.9.0; vision_version=0.24.0 ;;
    cpu) python_spec=3.12; python_minor=3.12; torch_version=2.8.0; vision_version=0.23.0 ;;
    *) die "Unsupported variant: $variant" ;;
esac
[[ "$(uname -s)" == Linux ]] || die 'This script targets Linux/WSL servers'
cd -- "$project_dir"
index_url="https://download.pytorch.org/whl/$variant"
runner=("$conda_bin" run --no-capture-output --name "$env_name")
if $dry_run; then
    printf 'Environment: %s; Python %s; PyTorch %s / torchvision %s (%s)\n' "$env_name" "$python_spec" "$torch_version" "$vision_version" "$variant"
    if ! $reuse; then
        printf '%q ' "$conda_bin" create --yes --name "$env_name" "python=$python_spec" pip
        printf '\n'
    fi
    printf '%q ' "${runner[@]}" python -m pip install "torch==$torch_version+$variant" "torchvision==$vision_version+$variant" --index-url "$index_url"
    printf '\n'
    printf '%q ' "${runner[@]}" python -m pip install -r "$project_dir/requirements.txt" --constraint '<generated torch constraints>'
    printf '\n'
    printf '%q ' "${runner[@]}" python -m pip check
    printf '\nChecks: Python/package versions, imports, native torchvision NMS, CPU/CUDA forward/backward.\n'
    exit 0
fi
command -v -- "$conda_bin" >/dev/null 2>&1 || die 'Conda not found; initialize Miniconda/Anaconda or set CONDA_EXE'
conda_base="$("$conda_bin" info --base)"
base_python="$conda_base/bin/python"
[[ -x "$base_python" ]] || die "Conda base Python not found: $base_python"
exists="$("$conda_bin" env list --json | "$base_python" -c '
import json, pathlib, sys
print("yes" if any(pathlib.Path(p).name == sys.argv[1] for p in json.load(sys.stdin)["envs"]) else "no")
' "$env_name")"
if [[ "$exists" == yes ]]; then
    $reuse || die "Environment '$env_name' exists; choose --name NEW_NAME or explicitly --reuse"
else
    $reuse && die "Environment '$env_name' does not exist; omit --reuse to create it"
    "$conda_bin" create --yes --name "$env_name" "python=$python_spec" pip
fi
"${runner[@]}" python -c 'import sys; assert ".".join(map(str, sys.version_info[:2])) == sys.argv[1], "Python " + sys.argv[1] + " required; choose a new environment"' "$python_minor"
mkdir -p -- "$project_dir/outputs"
record_dir="$(mktemp -d "$project_dir/outputs/conda_setup.XXXXXX")"
constraints="$record_dir/torch_constraints.txt"
printf 'torch==%s+%s\ntorchvision==%s+%s\n' "$torch_version" "$variant" "$vision_version" "$variant" > "$constraints"
"${runner[@]}" python -m pip install "torch==$torch_version+$variant" "torchvision==$vision_version+$variant" --index-url "$index_url"
# Constraints keep the selected wheel pair when the generic requirements resolve.
"${runner[@]}" python -m pip install -r "$project_dir/requirements.txt" --constraint "$constraints"
"${runner[@]}" python -m pip check
"${runner[@]}" python - "$variant" "$record_dir" "$project_dir" "$torch_version" "$vision_version" "$env_name" <<'PY'
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy
import PIL
import pytest
import sklearn
import torch
import torchvision
import yaml

variant, record_dir, project, torch_version, vision_version, env_name = sys.argv[1:]
assert torch.__version__ == f"{torch_version}+{variant}", torch.__version__
assert torchvision.__version__ == f"{vision_version}+{variant}", torchvision.__version__
if variant == "cpu":
    assert torch.version.cuda is None, "Expected CPU-only PyTorch"
    device = torch.device("cpu")
else:
    expected_cuda = {"cu126": "12.6", "cu128": "12.8", "cu130": "13.0"}[variant]
    assert torch.version.cuda == expected_cuda, torch.version.cuda
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: check NVIDIA driver, GPU visibility and selected wheel build")
    device = torch.device("cuda")
layer = torch.nn.Conv2d(3, 4, 3).to(device)
loss = layer(torch.randn(2, 3, 16, 16, device=device)).square().mean()
loss.backward()
assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in layer.parameters())
boxes = torch.tensor([[0., 0., 10., 10.], [20., 20., 30., 30.]], device=device)
assert torchvision.ops.nms(boxes, torch.tensor([.9, .8], device=device), .5).tolist() == [0, 1]
if device.type == "cuda":
    torch.cuda.synchronize()
packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}
directory = Path(record_dir)
report = {"environment_name": env_name, "python": sys.version, "executable": sys.executable, "variant": variant,
          "torch": torch.__version__, "torchvision": torchvision.__version__, "torch_cuda_build": torch.version.cuda,
          "selected_device": str(device), "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
          "probe_passed": True, "probe_scope": "small Conv2d backward and native NMS; full baseline capacity untested",
          "packages": dict(sorted(packages.items())),
          "requirements_sha256": hashlib.sha256((Path(project) / "requirements.txt").read_bytes()).hexdigest(),
          "setup_script_sha256": hashlib.sha256((Path(project) / "scripts/setup_conda.sh").read_bytes()).hexdigest()}
(directory / "environment.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(directory / "requirements.freeze.txt").write_text("".join(f"{k}=={v}\n" for k, v in sorted(packages.items())), encoding="utf-8")
print(json.dumps({k:report[k] for k in ("python", "torch", "torchvision", "selected_device", "gpu", "probe_passed")}, ensure_ascii=False))
PY
printf '\nInstalled successfully. Environment report: %s/environment.json\n' "$record_dir"
printf 'Activate: conda activate %s\n' "$env_name"
printf 'Train: bash scripts/train_server.sh --data-root /data/derm7pt --protocol dev_cv5 --seeds 42'
if [[ "$variant" == cpu ]]; then printf ' --device cpu'; fi
printf '\n'
