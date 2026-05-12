#!/usr/bin/env bash
set -euo pipefail

cd /project3/chueating/VRDL/HW3
source configs/baseline_v2_adamw_cosine.sh

./scripts/train.sh \
  --output-dir outputs/r50_baseline_v2 \
  --run-name r50-baseline-v2-adamw-cosine-ep20 \
  "${BASELINE_V2_ARGS[@]}" \
  "$@"
