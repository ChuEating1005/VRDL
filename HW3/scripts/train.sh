#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd /project3/chueating/VRDL/HW3
uv run python -m src.main --mode train --output-dir outputs/maskrcnn_r50 "$@"
