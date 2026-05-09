#!/usr/bin/env bash
# Baseline config: Mask R-CNN R50-FPN with small anchors and strong augmentations.
EXTRA_ARGS=(
  --backbone resnet50
  --small-anchors
  --strong-aug
  --detections-per-img 1000
  --repeat-threshold 1.0
)
