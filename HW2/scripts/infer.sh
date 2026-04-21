#!/usr/bin/env bash
set -euo pipefail

cd /project3/chueating/VRDL/HW2
python -m src.main --test --checkpoint outputs/dino_r50_4scale/checkpoint.pth "$@"
