"""
Model module for HW1 Image Classification.

Constraint: Must use ResNet backbone, <100M parameters.
Pretrained weights (ImageNet) are allowed.
"""

import torch
import torch.nn as nn
from torchvision import models
import timm


NUM_CLASSES = 100


def build_model(
    arch: str = "resnet50",
    dropout: float = 0.0,
) -> nn.Module:
    """Build a ResNet-based classification model.
    Args:
        arch: ResNet architecture name (resnet18/34/50/101/152, resnext101, resnest200).
        dropout: Dropout rate before the final FC layer.
    Returns:
        A nn.Module ready for training.
    """

    # --- ResNeSt via timm ---
    if arch == "resnest200":
        model = timm.create_model("resnest200e", pretrained=True)  # ~68M params
        in_features = model.fc.in_features
        if dropout > 0.0:
            model.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, NUM_CLASSES),
            )
        else:
            model.fc = nn.Linear(in_features, NUM_CLASSES)
        return model

    # --- Torchvision models ---
    if arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)  # 11.7M params
    elif arch == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1
        model = models.resnet34(weights=weights)  # 21.8M params
    elif arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)  # 25.6M params
    elif arch == "resnet101":
        weights = models.ResNet101_Weights.IMAGENET1K_V2
        model = models.resnet101(weights=weights)  # 44.5M params
    elif arch == "resnet152":
        weights = models.ResNet152_Weights.IMAGENET1K_V2
        model = models.resnet152(weights=weights)  # 60.2M params
    elif arch == "resnext101":
        weights = models.ResNeXt101_64X4D_Weights.IMAGENET1K_V1
        model = models.resnext101_64x4d(weights=weights)  # ~83M params
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    # Replace the fully-connected layer
    in_features = model.fc.in_features
    if dropout > 0.0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, NUM_CLASSES),
        )
    else:
        model.fc = nn.Linear(in_features, NUM_CLASSES)

    return model
