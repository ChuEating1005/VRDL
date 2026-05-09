# HW3 - Instance Segmentation

## Introduction

This project solves the VRDL 2026 Spring HW3 colored medical image instance segmentation task. The dataset contains 209 training images and 101 test images in TIFF format. Each training sample provides `image.tif` plus class-specific masks (`class1.tif` ... `class4.tif`), where each non-zero unique pixel value denotes one object instance.

Key techniques implemented for ablation:

- **Baseline**: TorchVision Mask R-CNN with ResNet-50-FPN and ImageNet-pretrained weights.
- **Backbone ablation**: `resnet50`, `resnet101`, and `convnext_tiny`.
- **FPN ablation**: optional PANet-style bottom-up path augmentation (`--use-pafpn`).
- **Mask loss ablation**: standard BCE or BCE + Dice (`--mask-loss {bce,dice}`).
- **Small-object setup**: small anchors and high `--detections-per-img` for dense cell images.
- **Rare-class/data ablation**: repeat sampling and rare-class Copy-Paste augmentation.
- **Inference ablation**: horizontal flip TTA and configurable mask/score thresholds.

## Environment Setup

```bash
cd /project3/chueating/VRDL/HW3
uv sync
source .venv/bin/activate
```

Dataset should be under `data/`:

```text
data/
├── train/{image_id}/image.tif
├── train/{image_id}/class1.tif ... class4.tif
├── test_release/*.tif
└── test_image_name_to_ids.json
```

## Usage

### Smoke test

```bash
uv run python -m src.main \
  --mode train \
  --epochs 1 \
  --debug-samples 1 \
  --batch-size 1 \
  --num-workers 0 \
  --max-size 512 \
  --output-dir outputs/smoke
```

### Baseline training

```bash
./scripts/train.sh \
  --epochs 20 \
  --batch-size 2 \
  --backbone resnet50 \
  --small-anchors \
  --detections-per-img 1000 \
  --repeat-threshold 1.0 \
  --run-name r50-baseline
```

### Ablation examples

```bash
# Stronger backbone
./scripts/train.sh --output-dir outputs/r101 --backbone resnet101

# ConvNeXt-Tiny backbone
./scripts/train.sh --output-dir outputs/convnext_tiny --backbone convnext_tiny

# PANet-style FPN modification
./scripts/train.sh --output-dir outputs/r50_pafpn --use-pafpn

# BCE + Dice mask loss
./scripts/train.sh --output-dir outputs/r50_dice --mask-loss dice

# Rare-class Copy-Paste augmentation
./scripts/train.sh --output-dir outputs/r50_copypaste --copy-paste-prob 0.3

# TTA at inference
./scripts/infer.sh --checkpoint outputs/r50_dice/best.pth --tta-hflip
```


### Weights & Biases logging

Training logs to the `vrdl-hw3` project by default. Set a custom run name with:

```bash
./scripts/train.sh --run-name r50-pafpn-dice
```

Disable logging for smoke tests or offline debugging:

```bash
./scripts/train.sh --no-wandb
```

### Evaluation

```bash
./scripts/eval.sh --checkpoint outputs/maskrcnn_r50/best.pth
```

### Inference / submission

```bash
./scripts/infer.sh \
  --checkpoint outputs/maskrcnn_r50/best.pth \
  --submission outputs/test-results.json \
  --score-threshold 0.05 \
  --mask-threshold 0.5
```

The generated JSON uses COCO instance segmentation format with RLE masks and records of:

```json
{"image_id": 1, "category_id": 1, "segmentation": {"size": [H, W], "counts": "..."}, "score": 0.9, "bbox": [x, y, w, h]}
```

## Performance Snapshot

Current code has passed an end-to-end smoke training/evaluation run on one sample. Full AP50 results should be filled after training ablations.
