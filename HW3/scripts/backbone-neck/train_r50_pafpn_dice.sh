#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd /project3/chueating/VRDL/HW3
source configs/_ablation_common.sh

uv run python -m src.main \
  --mode train \
  --output-dir outputs/r50_pafpn_dice \
  --run-name r50-pafpn-dice-ep30 \
  --backbone resnet50 \
  --use-pafpn \
  "${ABLATION_COMMON_ARGS[@]}" \
  --mask-loss dice \
  "$@"
