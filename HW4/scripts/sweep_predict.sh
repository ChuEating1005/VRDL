#!/usr/bin/env bash
# Post-training sweep: for each finished variant, run predict --tta 8, zip
# pred.npz, and print best-of-history. Skips runs whose best.pt does not exist.
set -euo pipefail
cd "$(dirname "$0")/.."

source .venv/bin/activate

for tag in "$@"; do
  ckpt="outputs/${tag}/best.pt"
  if [[ ! -f "$ckpt" ]]; then
    echo "[skip] $tag: $ckpt missing"
    continue
  fi
  echo "============================================================"
  echo "[$tag] best.pt found"
  python - <<PY
import json
h = json.load(open("outputs/${tag}/history.json"))
def best(key): return max(((r.get(key, -1), r["epoch"]) for r in h), default=(-1,-1))
on = best("val_psnr"); em = best("val_psnr_ema")
print(f"  online best: psnr={on[0]:.4f} ep={on[1]}")
print(f"  EMA    best: psnr={em[0]:.4f} ep={em[1]}")
print(f"  saved best   ep={h[-1]['epoch']} last loss={h[-1]['loss']:.4f}")
PY
  echo "[$tag] predicting TTA=8 ..."
  python -m src.main predict --ckpt "$ckpt" \
    --out "outputs/${tag}/pred_tta8.npz" --tta 8
  ( cd "outputs/${tag}" && cp pred_tta8.npz pred.npz && \
    zip -j "submission_${tag}_tta8.zip" pred.npz && rm pred.npz )
  ls -la "outputs/${tag}/submission_${tag}_tta8.zip"
done
echo "DONE"
