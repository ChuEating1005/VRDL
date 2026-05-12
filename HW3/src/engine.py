"""Training, evaluation, and inference loops for HW3."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, cast

import numpy as np
import torch
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from tqdm import tqdm

from pycocotools import mask as mask_util
from torchvision.ops import box_iou, nms

from src.util.submit import candidate_to_record, mask_to_rle, prediction_to_candidates, prediction_to_records


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
        loss_dict = cast(dict[str, torch.Tensor], model(images, targets))
        losses = torch.stack(list(loss_dict.values())).sum()
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
    metrics: dict[str, float] = {"ap": float(evaluator.stats[0]), "ap50": float(evaluator.stats[1])}
    precision = evaluator.eval["precision"]
    cat_ids = evaluator.params.catIds
    for k, cat_id in enumerate(cat_ids):
        per_class = precision[0, :, k, 0, -1]
        valid = per_class[per_class > -1]
        ap50_c = float(valid.mean()) if valid.size else float("nan")
        cat_name = coco_gt.cats[cat_id]["name"]
        metrics[f"ap50_{cat_name}"] = ap50_c
    return metrics


@torch.inference_mode()
def predict_submission(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    mask_threshold: float = 0.5,
    score_threshold: float = 0.05,
    tta_hflip: bool = False,
    tta_vhflip: bool = False,
    tta_scales: tuple[float, ...] = (1.0,),
    tta_fusion: str = "none",
    tta_iou_threshold: float = 0.5,
    tta_fusion_mask_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    model.eval()
    records: list[dict[str, Any]] = []
    for images, infos in tqdm(data_loader, desc="infer"):
        for image, info in zip(images, infos):
            image = image.to(device)
            records.extend(
                _predict_records_with_tta(
                    model=model,
                    image=image,
                    image_id=int(info.image_id),
                    mask_threshold=mask_threshold,
                    score_threshold=score_threshold,
                    tta_hflip=tta_hflip,
                    tta_vhflip=tta_vhflip,
                    tta_scales=tta_scales,
                    clear_cuda_cache=device.type == "cuda",
                    tta_fusion=tta_fusion,
                    tta_iou_threshold=tta_iou_threshold,
                    tta_fusion_mask_threshold=tta_fusion_mask_threshold,
                )
            )
            del image
    return records


def _predict_records_with_tta(
    model: torch.nn.Module,
    image: torch.Tensor,
    image_id: int,
    mask_threshold: float,
    score_threshold: float,
    tta_hflip: bool,
    tta_vhflip: bool,
    tta_scales: tuple[float, ...],
    clear_cuda_cache: bool,
    tta_fusion: str,
    tta_iou_threshold: float,
    tta_fusion_mask_threshold: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    use_fusion = tta_fusion != "none" and (tta_hflip or tta_vhflip or len(tta_scales) > 1)
    candidates = _predict_candidates_with_tta(
        model=model,
        image=image,
        image_id=image_id,
        mask_threshold=mask_threshold,
        score_threshold=score_threshold,
        tta_hflip=tta_hflip,
        tta_vhflip=tta_vhflip,
        tta_scales=tta_scales,
        clear_cuda_cache=clear_cuda_cache,
    )
    if use_fusion:
        return _fuse_tta_candidates(
            candidates,
            fusion=tta_fusion,
            iou_threshold=tta_iou_threshold,
            mask_threshold=tta_fusion_mask_threshold,
        )
    records.extend(candidate_to_record(candidate) for candidate in candidates)
    return records


def _predict_candidates_with_tta(
    model: torch.nn.Module,
    image: torch.Tensor,
    image_id: int,
    mask_threshold: float,
    score_threshold: float,
    tta_hflip: bool,
    tta_vhflip: bool,
    tta_scales: tuple[float, ...],
    clear_cuda_cache: bool,
) -> list[dict[str, Any]]:
    base_h, base_w = image.shape[-2:]
    candidates: list[dict[str, Any]] = []
    flip_modes = _tta_flip_modes(tta_hflip=tta_hflip, tta_vhflip=tta_vhflip)
    for scale in tta_scales:
        if scale == 1.0:
            scaled = image
        else:
            scaled = torch.nn.functional.interpolate(image[None], scale_factor=scale, mode="bilinear", align_corners=False)[0]
        scaled_h, scaled_w = scaled.shape[-2:]
        for hflip, vflip in flip_modes:
            aug_image = _flip_image(scaled, hflip=hflip, vflip=vflip)
            output = model([aug_image])[0]
            output = _unflip_output(output, width=scaled_w, height=scaled_h, hflip=hflip, vflip=vflip)
            output = _resize_output_to_original(output, scaled_w, scaled_h, base_w, base_h)
            candidates.extend(
                prediction_to_candidates(
                    output,
                    image_id,
                    mask_threshold=mask_threshold,
                    score_threshold=score_threshold,
                )
            )
            del aug_image, output
            if clear_cuda_cache:
                torch.cuda.empty_cache()
        if scale != 1.0:
            del scaled
            if clear_cuda_cache:
                torch.cuda.empty_cache()
    return candidates


def _fuse_tta_candidates(
    candidates: list[dict[str, Any]],
    fusion: str,
    iou_threshold: float,
    mask_threshold: float,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    records: list[dict[str, Any]] = []
    labels = sorted({int(c["category_id"]) for c in candidates})
    for label in labels:
        cls_candidates = [c for c in candidates if int(c["category_id"]) == label]
        boxes = torch.tensor([c["box_xyxy"] for c in cls_candidates], dtype=torch.float32)
        scores = torch.tensor([float(c["score"]) for c in cls_candidates], dtype=torch.float32)
        keep = nms(boxes, scores, iou_threshold).tolist()
        if fusion == "nms":
            records.extend(candidate_to_record(cls_candidates[i]) for i in keep)
            continue
        assigned = torch.zeros((len(cls_candidates),), dtype=torch.bool)
        ious = box_iou(boxes, boxes)
        for keep_idx in keep:
            if assigned[keep_idx]:
                continue
            group = torch.nonzero((ious[keep_idx] >= iou_threshold) & ~assigned, as_tuple=False).flatten().tolist()
            for idx in group:
                assigned[idx] = True
            records.append(_fuse_candidate_group([cls_candidates[i] for i in group], mask_threshold=mask_threshold))
    records.sort(key=lambda row: float(row["score"]), reverse=True)
    return records


def _fuse_candidate_group(candidates: list[dict[str, Any]], mask_threshold: float) -> dict[str, Any]:
    if len(candidates) == 1:
        return candidate_to_record(candidates[0])
    scores = np.asarray([float(c["score"]) for c in candidates], dtype=np.float32)
    weights = scores / max(float(scores.sum()), 1e-6)
    fused: np.ndarray | None = None
    for candidate, weight in zip(candidates, weights):
        decoded = mask_util.decode(candidate["segmentation"]).astype(np.float32)
        fused = decoded * float(weight) if fused is None else fused + decoded * float(weight)
    assert fused is not None
    binary = fused >= mask_threshold
    if not binary.any():
        best = max(candidates, key=lambda c: float(c["score"]))
        return candidate_to_record(best)
    rle = mask_to_rle(binary)
    x, y, w, h = mask_util.toBbox(cast(Any, rle)).tolist()
    return {
        "image_id": int(candidates[0]["image_id"]),
        "category_id": int(candidates[0]["category_id"]),
        "segmentation": rle,
        "score": float(scores.max()),
        "bbox": [float(x), float(y), float(w), float(h)],
    }


def _flip_image(image: torch.Tensor, hflip: bool, vflip: bool) -> torch.Tensor:
    dims: list[int] = []
    if vflip:
        dims.append(1)
    if hflip:
        dims.append(2)
    return torch.flip(image, dims=dims) if dims else image


def _unflip_output(
    output: dict[str, torch.Tensor],
    width: int,
    height: int,
    hflip: bool,
    vflip: bool,
) -> dict[str, torch.Tensor]:
    if not hflip and not vflip:
        return output
    boxes = output["boxes"].clone()
    if hflip:
        x1 = width - boxes[:, 2]
        x2 = width - boxes[:, 0]
        boxes[:, 0] = x1
        boxes[:, 2] = x2
    if vflip:
        y1 = height - boxes[:, 3]
        y2 = height - boxes[:, 1]
        boxes[:, 1] = y1
        boxes[:, 3] = y2
    mask_dims: list[int] = []
    if vflip:
        mask_dims.append(2)
    if hflip:
        mask_dims.append(3)
    masks = torch.flip(output["masks"], dims=mask_dims) if mask_dims else output["masks"]
    return {**output, "boxes": boxes, "masks": masks}


def _tta_flip_modes(tta_hflip: bool, tta_vhflip: bool) -> tuple[tuple[bool, bool], ...]:
    if tta_vhflip:
        return ((False, False), (True, False), (False, True), (True, True))
    if tta_hflip:
        return ((False, False), (True, False))
    return ((False, False),)



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
