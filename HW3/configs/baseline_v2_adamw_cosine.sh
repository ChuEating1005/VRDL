#!/usr/bin/env bash
# Baseline-v2: senior-repo-inspired optimizer/schedule + light deformation aug.
# Keeps baseline-v1 architecture/memory/TTA knobs; only training recipe changes.
BASELINE_V2_ARGS=(
  --epochs 20
  --batch-size 2
  --lr 0.0003
  --weight-decay 0.0005
  --optimizer adamw
  --scheduler cosine
  --backbone resnet50
  --trainable-backbone-layers 3
  --mask-loss bce
  --small-anchors
  --detections-per-img 1000
  --box-score-thresh 0.05
  --model-min-size 512
  --model-max-size 1024
  --rpn-pre-nms-top-n-train 1000
  --rpn-post-nms-top-n-train 500
  --rpn-pre-nms-top-n-test 1000
  --rpn-post-nms-top-n-test 500
  --box-batch-size-per-image 256
  --strong-aug
  --min-scale 0.6
  --max-scale 1.6
  --max-size 1024
  --deformation-prob 0.2
  --copy-paste-prob 0.0
  --repeat-threshold 0.0
  --score-threshold 0.05
  --mask-threshold 0.5
  --tta-scales 1.0
)
