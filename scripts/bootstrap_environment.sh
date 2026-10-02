#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python3 -c 'import torch; assert torch.cuda.is_available(), "A working system CUDA PyTorch is required"'
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --disable-pip-version-check --no-cache-dir \
    --constraint constraints-runtime.txt --editable '.[ml,dev]'
.venv/bin/python scripts/check_environment.py --require-gpus 1 --check-ml \
    --output reports/environment.json