# HW1 - Image Classification

## Introduction

This project tackles a 100-class image classification task from the NYCU Visual Recognition using Deep Learning (Spring 2026) course. The goal is to classify RGB images into one of 100 object categories using a ResNet-based backbone with fewer than 100M parameters.

Key techniques employed:

- **Backbone**: ResNet family (ResNet-18/34/50/101/152, ResNeXt-101, ResNeSt-200) with ImageNet-pretrained weights.
- **Data Augmentation**: RandomResizedCrop, TrivialAugmentWide, RandomErasing, CutMix, and MixUp.
- **Training**: AdamW optimizer with label smoothing (0.1) and multiple LR scheduler options (Cosine Annealing, OneCycleLR, ReduceLROnPlateau, Warm Restarts).
- **Inference**: Test-Time Augmentation (TTA) with multi-scale crops and horizontal flips.
- **Logging**: Weights & Biases for experiment tracking.

## Environment Setup

**Prerequisites**: Python 3.11+, CUDA-capable GPU recommended.

1. Install [uv](https://docs.astral.sh/uv/) (recommended package manager):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone the repository and install dependencies:

   ```bash
   git clone <REPO_URL>
   cd HW1
   uv sync
   ```

   This installs all dependencies defined in `pyproject.toml`, including:

   | Package | Version |
   |---|---|
   | PyTorch | >= 2.11.0 (CUDA 12.6) |
   | TorchVision | >= 0.26.0 |
   | timm | >= 1.0.26 |
   | wandb | >= 0.25.0 |
   | pandas | >= 3.0.0 |
   | tqdm | >= 4.67.0 |
   | matplotlib | >= 3.10.0 |

3. Prepare the dataset under `data/`:

   ```
   data/
   ├── train/{class_id}/*.jpg   (20,724 images, 100 classes)
   ├── val/{class_id}/*.jpg     (300 images, 100 classes)
   └── test/*.jpg               (2,344 images)
   ```

4. (Optional) Log in to Weights & Biases:

   ```bash
   wandb login
   ```

## Usage

### Training

```bash
uv run python train.py \
    --data_dir data \
    --arch resnet50 \
    --epochs 80 \
    --batch_size 64 \
    --lr 1e-4 \
    --img_size 384 \
    --scheduler cosine
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--arch` | `resnet50` | Model architecture (`resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`, `resnext101`, `resnest200`) |
| `--epochs` | `80` | Number of training epochs |
| `--batch_size` | `64` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--img_size` | `384` | Input image resolution |
| `--scheduler` | `cosine` | LR scheduler (`cosine`, `plateau`, `onecycle`, `warm_restarts`) |
| `--dropout` | `0.0` | Dropout rate before the final FC layer |
| `--resume` | `None` | Path to checkpoint to resume training |

Checkpoints are saved to `checkpoints/` (`best.pth` and `last.pth`).

### Inference

```bash
uv run python predict.py \
    --checkpoint checkpoints/best.pth \
    --arch resnet50 \
    --img_size 384 \
    --output prediction.csv
```

Enable Test-Time Augmentation for better accuracy:

```bash
uv run python predict.py \
    --checkpoint checkpoints/best.pth \
    --arch resnet50 \
    --img_size 384 \
    --tta \
    --output prediction.csv
```

The output `prediction.csv` can be directly submitted to CodaBench (zip the file with the name `prediction.csv` inside).

## Performance Snapshot

| Model | Params | Image Size | Scheduler | Val Acc |
|---|---|---|---|---|
| ResNet-50 | 23.9M | 384 | Cosine | — |

> **Note**: Fill in the Val Acc column with your actual best validation accuracy after training.
