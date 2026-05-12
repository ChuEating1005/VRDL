#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd /project3/chueating/VRDL/HW3
source configs/_ablation_common.sh

uv run python -m src.main \
  --mode train \
  --output-dir outputs/r101_fpn \
  --run-name r101-fpn-ep30 \
  --backbone resnet101 \
  "${ABLATION_COMMON_ARGS[@]}" \
  "$@"
