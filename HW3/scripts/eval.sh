#!/usr/bin/env bash
set -euo pipefail

cd /project3/chueating/VRDL/HW3
uv run python -m src.main --mode eval --checkpoint outputs/maskrcnn_r50/best.pth "$@"
