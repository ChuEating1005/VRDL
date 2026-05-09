#!/usr/bin/env bash
set -euo pipefail

cd /project3/chueating/VRDL/HW3
uv run python -m src.main --mode infer --checkpoint outputs/maskrcnn_r50/best.pth --submission outputs/test-results.json "$@"
