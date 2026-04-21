"""
Bounding box utilities.

Conventions:
- cxcywh: (center_x, center_y, width, height), normalized to [0, 1] by image size.
- xyxy:   (x_min, y_min, x_max, y_max), normalized to [0, 1] by image size.
- xywh:   (x_min, y_min, width, height), unnormalized (COCO format for output).

All tensor ops work on the last dim.
"""
from typing import Tuple

import torch
from torch import Tensor
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x: Tensor) -> Tensor:
    cx, cy, w, h = x.unbind(-1)
    b = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_xywh_to_xyxy(x: Tensor) -> Tensor:
    x0, y0, w, h = x.unbind(-1)
    return torch.stack([x0, y0, x0 + w, y0 + h], dim=-1)


def box_xyxy_to_xywh(x: Tensor) -> Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    return torch.stack([x0, y0, x1 - x0, y1 - y0], dim=-1)


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    """IoU between every pair. boxes in xyxy. Returns (iou, union) both [N, M]."""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)
    return iou, union


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """GIoU between every pair. boxes in xyxy. Returns [N, M]."""
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all(), "boxes1 must be xyxy with x2>=x1, y2>=y1"
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all(), "boxes2 must be xyxy with x2>=x1, y2>=y1"
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])

    wh = (rb - lt).clamp(min=0)
    area = wh[..., 0] * wh[..., 1]

    return iou - (area - union) / area.clamp(min=1e-6)


def masks_to_boxes(masks: Tensor) -> Tensor:
    """Compute xyxy bounding boxes from binary masks [N, H, W]. Returns [N, 4]."""
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)
    h, w = masks.shape[-2:]
    y = torch.arange(0, h, dtype=torch.float, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float, device=masks.device)
    y, x = torch.meshgrid(y, x, indexing="ij")

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~masks.bool(), 1e8).flatten(1).min(-1)[0]

    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~masks.bool(), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], dim=1)
