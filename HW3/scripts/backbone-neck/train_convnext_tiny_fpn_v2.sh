#!/usr/bin/env bash
set -euo pipefail

cd /project3/chueating/VRDL/HW3
source configs/baseline_v2_adamw_cosine.sh

./scripts/train.sh \
  --output-dir outputs/convnext_tiny_fpn_v2 \
  --run-name convnext-tiny-fpn-ep20-v2 \
  "${BASELINE_V2_ARGS[@]}" \
  --backbone convnext_tiny \
  "$@"
