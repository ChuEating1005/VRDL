"""Training, evaluation, and inference loops for HW3."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.util.submit import prediction_to_records


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0.0,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = defaultdict(float)
    pbar = tqdm(data_loader, desc=f"train {epoch}")
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        if not math.isfinite(float(losses.item())):
            raise RuntimeError(f"Non-finite loss: {losses.item()} {loss_dict}")
        optimizer.zero_grad(set_to_none=True)
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        for key, value in loss_dict.items():
            totals[key] += float(value.item())
        totals["loss"] += float(losses.item())
        pbar.set_postfix(loss=f"{losses.item():.4f}")
    n = max(1, len(data_loader))
    return {k: v / n for k, v in totals.items()}


@torch.inference_mode()
def evaluate_ap50(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    coco_gt: Any,
    mask_threshold: float = 0.5,
    score_threshold: float = 0.05,
) -> dict[str, float]:
    model.eval()
    records: list[dict[str, Any]] = []
    for images, targets in tqdm(data_loader, desc="eval"):
        images = [img.to(device) for img in images]
        outputs = model(images)
        for output, target in zip(outputs, targets):
            records.extend(
                prediction_to_records(
                    output,
                    int(target["image_id"].item()),
                    mask_threshold=mask_threshold,
                    score_threshold=score_threshold,
                )
            )
    if not records:
        return {"ap": 0.0, "ap50": 0.0}
    coco_dt = coco_gt.loadRes(records)
    evaluator = COCOeval(coco_gt, coco_dt, "segm")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {"ap": float(evaluator.stats[0]), "ap50": float(evaluator.stats[1])}


@torch.inference_mode()
def predict_submission(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    mask_threshold: float = 0.5,
    score_threshold: float = 0.05,
    tta_hflip: bool = False,
    tta_scales: tuple[float, ...] = (1.0,),
) -> list[dict[str, Any]]:
    model.eval()
    records: list[dict[str, Any]] = []
    for images, infos in tqdm(data_loader, desc="infer"):
        images = [img.to(device) for img in images]
        outputs = [_predict_with_tta(model, img, tta_hflip=tta_hflip, tta_scales=tta_scales) for img in images]
        for output, info in zip(outputs, infos):
            records.extend(
                prediction_to_records(
                    output,
                    int(info.image_id),
                    mask_threshold=mask_threshold,
                    score_threshold=score_threshold,
                )
            )
    return records


def _merge_hflip_output(output: dict[str, torch.Tensor], flipped_output: dict[str, torch.Tensor], width: int) -> dict[str, torch.Tensor]:
    boxes = flipped_output["boxes"].clone()
    x1 = width - boxes[:, 2]
    x2 = width - boxes[:, 0]
    boxes[:, 0] = x1
    boxes[:, 2] = x2
    masks = torch.flip(flipped_output["masks"], dims=[3])
    merged = {
        "boxes": torch.cat([output["boxes"], boxes], dim=0),
        "labels": torch.cat([output["labels"], flipped_output["labels"]], dim=0),
        "scores": torch.cat([output["scores"], flipped_output["scores"]], dim=0),
        "masks": torch.cat([output["masks"], masks], dim=0),
    }
    order = torch.argsort(merged["scores"], descending=True)
    return {k: v[order] for k, v in merged.items()}


def _predict_with_tta(
    model: torch.nn.Module,
    image: torch.Tensor,
    tta_hflip: bool,
    tta_scales: tuple[float, ...],
) -> dict[str, torch.Tensor]:
    base_h, base_w = image.shape[-2:]
    merged_outputs: list[dict[str, torch.Tensor]] = []
    for scale in tta_scales:
        if scale == 1.0:
            scaled = image
        else:
            scaled = torch.nn.functional.interpolate(
                image[None], scale_factor=scale, mode="bilinear", align_corners=False
            )[0]
        out = model([scaled])[0]
        out = _resize_output_to_original(out, scaled.shape[-1], scaled.shape[-2], base_w, base_h)
        merged_outputs.append(out)
        if tta_hflip:
            flipped = torch.flip(scaled, dims=[2])
            flip_out = model([flipped])[0]
            flip_out = _merge_hflip_output({k: v[:0] for k, v in out.items()}, flip_out, scaled.shape[-1])
            flip_out = _resize_output_to_original(flip_out, scaled.shape[-1], scaled.shape[-2], base_w, base_h)
            merged_outputs.append(flip_out)
    if len(merged_outputs) == 1:
        return merged_outputs[0]
    concat = {
        key: torch.cat([o[key] for o in merged_outputs], dim=0)
        for key in ("boxes", "labels", "scores", "masks")
    }
    order = torch.argsort(concat["scores"], descending=True)
    return {k: v[order] for k, v in concat.items()}


def _resize_output_to_original(
    output: dict[str, torch.Tensor],
    scaled_w: int,
    scaled_h: int,
    base_w: int,
    base_h: int,
) -> dict[str, torch.Tensor]:
    if scaled_w == base_w and scaled_h == base_h:
        return output
    sx = base_w / scaled_w
    sy = base_h / scaled_h
    boxes = output["boxes"].clone()
    boxes[:, [0, 2]] *= sx
    boxes[:, [1, 3]] *= sy
    masks = torch.nn.functional.interpolate(output["masks"], size=(base_h, base_w), mode="bilinear", align_corners=False)
    return {**output, "boxes": boxes, "masks": masks}
