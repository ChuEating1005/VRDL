from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any

import torch
import torch.distributed as dist
from pycocotools.cocoeval import COCOeval

from .util import box_ops
from .util import misc as utils


def _move_targets_to_device(targets: list[dict[str, Any]] | tuple[dict[str, Any], ...], device: torch.device) -> list[dict[str, Any]]:
    return [
        {
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in target.items()
        }
        for target in targets
    ]


def _reduce_dict(input_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not utils.is_dist_avail_and_initialized():
        return input_dict

    with torch.no_grad():
        keys = sorted(input_dict.keys())
        values = torch.stack([input_dict[key] for key in keys], dim=0)
        dist.all_reduce(values)
        values /= utils.get_world_size()
    return {key: value for key, value in zip(keys, values)}


def _gather_object_list(local_objects: list[Any]) -> list[Any]:
    if not utils.is_dist_avail_and_initialized():
        return local_objects

    gathered_objects: list[list[Any] | None] = [None for _ in range(utils.get_world_size())]
    dist.all_gather_object(gathered_objects, local_objects)

    merged: list[Any] = []
    for objects in gathered_objects:
        if objects:
            merged.extend(objects)
    return merged


def _summarize_coco_stats(coco_stats: list[float]) -> dict[str, float | list[float]]:
    stats = coco_stats + [0.0] * max(0, 12 - len(coco_stats))
    return {
        "mAP": float(stats[0]),
        "mAP50": float(stats[1]),
        "mAP75": float(stats[2]),
        "mAP_small": float(stats[3]),
        "mAP_medium": float(stats[4]),
        "mAP_large": float(stats[5]),
        "AR1": float(stats[6]),
        "AR10": float(stats[7]),
        "AR100": float(stats[8]),
        "AR_small": float(stats[9]),
        "AR_medium": float(stats[10]),
        "AR_large": float(stats[11]),
        "coco_eval_bbox": [float(value) for value in stats[:12]],
    }


def train_one_epoch(
    model,
    criterion,
    data_loader,
    optimizer,
    device,
    epoch,
    max_norm=0.1,
    scaler=None,
    debug_max_iters=None,
    wandb_run=None,
    wandb_log_freq=10,
    global_step_start=0,
    amp_dtype=torch.float16,
):
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("grad_norm", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = 10

    iteration = -1
    for iteration, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        targets = _move_targets_to_device(targets, device)

        amp_enabled = scaler is not None or amp_dtype == torch.bfloat16
        autocast_context = (
            torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled)
            if (scaler is not None or amp_dtype == torch.bfloat16)
            else nullcontext()
        )

        with autocast_context:
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            weighted_losses = [loss_dict[key] * weight_dict[key] for key in loss_dict.keys() if key in weight_dict]
            losses = torch.stack(weighted_losses).sum() if weighted_losses else torch.zeros((), device=device)

        loss_dict_reduced = _reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            key: loss_dict_reduced[key] * weight_dict[key]
            for key in loss_dict_reduced.keys()
            if key in weight_dict
        }
        reduced_scaled_losses = list(loss_dict_reduced_scaled.values())
        losses_reduced_scaled = (
            torch.stack(reduced_scaled_losses).sum() if reduced_scaled_losses else torch.zeros((), device=device)
        )
        loss_value = float(losses_reduced_scaled.item())

        if not math.isfinite(loss_value):
            raise RuntimeError(f"Non-finite loss detected: {loss_value}. Loss dict: {loss_dict_reduced}")

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        metric_logger.update(loss=loss_value, lr=optimizer.param_groups[0]["lr"], grad_norm=float(grad_norm))
        metric_logger.update(**{key: float(value.item()) for key, value in loss_dict_reduced_scaled.items()})

        if (
            wandb_run is not None
            and utils.is_main_process()
            and (iteration % wandb_log_freq == 0)
        ):
            step = global_step_start + iteration
            payload = {
                "train/loss": loss_value,
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/grad_norm": float(grad_norm),
                "train/epoch": epoch,
            }
            for key, value in loss_dict_reduced_scaled.items():
                payload[f"train/{key}"] = float(value.item())
            wandb_run.log(payload, step=step)

        if debug_max_iters is not None and iteration + 1 >= debug_max_iters:
            break

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {key: meter.global_avg for key, meter in metric_logger.meters.items()}
    stats["_num_iters"] = iteration + 1
    return stats


@torch.no_grad()
def evaluate(
    model,
    criterion,
    postprocessor,
    data_loader,
    base_ds,
    device,
    output_dir=None,
    debug_max_iters=None,
):
    del criterion, output_dir
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Eval:"

    predictions: list[dict[str, float | int | list[float]]] = []
    evaluated_image_ids: list[int] = []

    for iteration, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        samples = samples.to(device, non_blocking=True)
        target_sizes = torch.stack([target["orig_size"] for target in targets], dim=0).to(device, non_blocking=True)

        outputs = model(samples, None)
        results = postprocessor(outputs, target_sizes=target_sizes)

        for target, result in zip(targets, results):
            image_id = int(target["image_id"].item())
            evaluated_image_ids.append(image_id)
            boxes_xywh = box_ops.box_xyxy_to_xywh(result["boxes"])

            for score, label, box in zip(result["scores"], result["labels"], boxes_xywh):
                category_id = int(label.item())
                if category_id <= 0:
                    continue

                bbox = [float(value) for value in box.tolist()]
                predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float(score.item()),
                    }
                )

        if debug_max_iters is not None and iteration + 1 >= debug_max_iters:
            break

    predictions = _gather_object_list(predictions)
    evaluated_image_ids = sorted(set(_gather_object_list(evaluated_image_ids)))

    stats: dict[str, float | list[float]]
    if utils.is_main_process():
        if predictions and evaluated_image_ids:
            coco_dt = base_ds.loadRes(predictions)
            coco_eval = COCOeval(base_ds, coco_dt, "bbox")
            coco_eval.params.imgIds = evaluated_image_ids
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            stats = _summarize_coco_stats(coco_eval.stats.tolist())
        else:
            stats = _summarize_coco_stats([0.0] * 12)
    else:
        stats = {}

    if utils.is_dist_avail_and_initialized():
        stats_list: list[dict[str, float | list[float]]] = [stats]
        dist.broadcast_object_list(stats_list, src=0)
        stats = stats_list[0]

    return stats, predictions
