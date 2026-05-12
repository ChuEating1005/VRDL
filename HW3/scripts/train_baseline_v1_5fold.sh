#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd /project3/chueating/VRDL/HW3
source configs/baseline_v1_ep20.sh

FOLDS="${FOLDS:-0 1 2 3 4}"

for fold in ${FOLDS}; do
  uv run python -m src.main \
    --mode train \
    --output-dir "outputs/r50_baseline_v1_ep20_fold${fold}" \
    --run-name "r50-baseline-v1-ep20-fold${fold}" \
    --fold "${fold}" \
    --num-folds 5 \
    "${BASELINE_V1_EP20_ARGS[@]}" \
    "$@"
done
