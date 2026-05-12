"""COCO RLE submission helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from pycocotools import mask as mask_util


def mask_to_rle(mask: np.ndarray) -> dict[str, Any]:
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def prediction_to_candidates(
    prediction: dict[str, torch.Tensor],
    image_id: int,
    mask_threshold: float = 0.5,
    score_threshold: float = 0.05,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    boxes = prediction["boxes"].detach().cpu().numpy()
    labels = prediction["labels"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().numpy()
    masks = prediction["masks"].detach().cpu().numpy()[:, 0]
    for box, label, score, mask in zip(boxes, labels, scores, masks):
        if float(score) < score_threshold:
            continue
        binary = mask >= mask_threshold
        if not binary.any():
            continue
        x1, y1, x2, y2 = [float(x) for x in box.tolist()]
        candidates.append(
            {
                "image_id": int(image_id),
                "category_id": int(label),
                "score": float(score),
                "box_xyxy": [x1, y1, x2, y2],
                "segmentation": mask_to_rle(binary),
            }
        )
    return candidates


def candidate_to_record(candidate: dict[str, Any]) -> dict[str, Any]:
    x1, y1, x2, y2 = candidate["box_xyxy"]
    return {
        "image_id": int(candidate["image_id"]),
        "category_id": int(candidate["category_id"]),
        "segmentation": candidate["segmentation"],
        "score": float(candidate["score"]),
        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
    }


def prediction_to_records(
    prediction: dict[str, torch.Tensor],
    image_id: int,
    mask_threshold: float = 0.5,
    score_threshold: float = 0.05,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    boxes = prediction["boxes"].detach().cpu().numpy()
    labels = prediction["labels"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().numpy()
    masks = prediction["masks"].detach().cpu().numpy()[:, 0]
    for box, label, score, mask in zip(boxes, labels, scores, masks):
        if float(score) < score_threshold:
            continue
        binary = mask >= mask_threshold
        if not binary.any():
            continue
        x1, y1, x2, y2 = box.tolist()
        records.append(
            {
                "image_id": int(image_id),
                "category_id": int(label),
                "segmentation": mask_to_rle(binary),
                "score": float(score),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            }
        )
    return records
