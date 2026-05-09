"""Model factory for HW3 Mask R-CNN ablations."""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, ResNet101_Weights, ResNet50_Weights, convnext_tiny, resnet101, resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN, resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.roi_heads import maskrcnn_loss as torchvision_maskrcnn_loss
from torchvision.ops import FeaturePyramidNetwork


class PAFPN(nn.Module):
    """A lightweight PANet-style bottom-up augmentation after torchvision FPN."""

    def __init__(self, in_channels: int, num_levels: int) -> None:
        super().__init__()
        self.num_levels = num_levels
        self.downsamples = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, in_channels),
                nn.ReLU(inplace=True),
            )
            for _ in range(max(0, num_levels - 1))
        )
        self.refines = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.GroupNorm(32, in_channels),
                nn.ReLU(inplace=True),
            )
            for _ in range(max(0, num_levels - 1))
        )

    def forward(self, x: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
        names = list(x.keys())
        pyramid = list(x.values())
        limit = min(len(pyramid), self.num_levels)
        for i in range(1, limit):
            down = self.downsamples[i - 1](pyramid[i - 1])
            if down.shape[-2:] != pyramid[i].shape[-2:]:
                down = F.interpolate(down, size=pyramid[i].shape[-2:], mode="nearest")
            pyramid[i] = self.refines[i - 1](pyramid[i] + down)
        return OrderedDict((name, feat) for name, feat in zip(names, pyramid))


class BackboneWithOptionalPAFPN(nn.Module):
    def __init__(self, backbone: nn.Module, use_pafpn: bool, out_channels: int = 256, fpn_levels: int = 4) -> None:
        super().__init__()
        self.body = backbone
        self.out_channels = out_channels
        self.pafpn = PAFPN(out_channels, fpn_levels) if use_pafpn else None

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        features = self.body(x)
        if self.pafpn is not None:
            features = self.pafpn(features)
        return features


def dice_maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):  # noqa: ANN001
    ce_loss = torchvision_maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs)
    from torchvision.models.detection.roi_heads import project_masks_on_boxes

    labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]
    mask_targets = [
        project_masks_on_boxes(m, p, i, mask_logits.shape[-1])
        for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]
    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)
    if mask_targets.numel() == 0:
        return ce_loss
    selected = mask_logits[torch.arange(labels.shape[0], device=labels.device), labels]
    probs = selected.sigmoid()
    dims = (1, 2)
    numerator = 2 * (probs * mask_targets).sum(dim=dims)
    denominator = probs.sum(dim=dims) + mask_targets.sum(dim=dims).clamp(min=1e-6)
    dice_loss = 1 - ((numerator + 1.0) / (denominator + 1.0)).mean()
    return ce_loss + dice_loss


def patch_mask_loss(mask_loss: str) -> None:
    if mask_loss == "dice":
        import torchvision.models.detection.roi_heads as roi_heads

        roi_heads.maskrcnn_loss = dice_maskrcnn_loss


def build_resnet_backbone(name: str, trainable_layers: int, use_pafpn: bool) -> nn.Module:
    weights = ResNet50_Weights.DEFAULT if name == "resnet50" else ResNet101_Weights.DEFAULT
    backbone = resnet_fpn_backbone(backbone_name=name, weights=weights, trainable_layers=trainable_layers)
    return BackboneWithOptionalPAFPN(backbone, use_pafpn=use_pafpn)


def build_convnext_fpn(trainable_layers: int, use_pafpn: bool) -> nn.Module:
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT).features
    # torchvision convnext stages: 0 stem, 1 block, 2 down, 3 block, 4 down, 5 block, 6 down, 7 block
    return_layers = {"1": "0", "3": "1", "5": "2", "7": "3"}
    in_channels_list = [96, 192, 384, 768]
    if trainable_layers < 4:
        # Freeze early stages, keep the last N stage blocks trainable.
        trainable_stage_indices = set(range(4 - trainable_layers, 4))
        stage_to_modules = {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7]}
        for stage_idx, module_indices in stage_to_modules.items():
            if stage_idx not in trainable_stage_indices:
                for module_idx in module_indices:
                    for param in model[module_idx].parameters():
                        param.requires_grad = False
    body = IntermediateLayerGetter(model, return_layers=return_layers)
    fpn = FeaturePyramidNetwork(in_channels_list, out_channels=256)

    class ConvNeXtFPN(nn.Module):
        out_channels = 256

        def __init__(self, body: nn.Module, fpn: nn.Module, pafpn: nn.Module | None) -> None:
            super().__init__()
            self.body = body
            self.fpn = fpn
            self.pafpn = pafpn

        def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
            x = self.body(x)
            x = self.fpn(x)
            if self.pafpn is not None:
                x = self.pafpn(x)
            return x

    return ConvNeXtFPN(body, fpn, PAFPN(256, 4) if use_pafpn else None)


def build_model(
    num_classes: int = 5,
    backbone: str = "resnet50",
    trainable_backbone_layers: int = 3,
    use_pafpn: bool = False,
    small_anchors: bool = True,
    detections_per_img: int = 1000,
    box_score_thresh: float = 0.05,
    mask_loss: str = "bce",
) -> MaskRCNN:
    """Build Mask R-CNN with CLI-controlled ablation switches."""
    patch_mask_loss(mask_loss)
    if backbone in {"resnet50", "resnet101"}:
        backbone_module = build_resnet_backbone(backbone, trainable_backbone_layers, use_pafpn)
    elif backbone == "convnext_tiny":
        backbone_module = build_convnext_fpn(trainable_backbone_layers, use_pafpn)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    anchor_sizes = ((16,), (32,), (64,), (128,), (256,)) if small_anchors else ((32,), (64,), (128,), (256,), (512,))
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=((0.5, 1.0, 2.0),) * len(anchor_sizes),
    )
    model = MaskRCNN(
        backbone_module,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_detections_per_img=detections_per_img,
        box_score_thresh=box_score_thresh,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels, hidden_layer, num_classes)
    return model


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
