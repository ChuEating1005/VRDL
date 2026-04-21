from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from pycocotools.coco import COCO
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

from configs.dino_4scale_r50 import get_args as get_config_parser
from .datasets.nycu_hw2 import build as build_dataset
from .engine import evaluate, train_one_epoch
from .models.criterion import build_criterion_and_postprocess
from .models.dino import build_dino
from .util import box_ops
from .util import misc as utils


def _init_wandb(args):
    if not getattr(args, "wandb", False) or args.wandb_mode == "disabled":
        return None
    if not utils.is_main_process():
        return None
    if args.eval or args.test:
        return None
    try:
        import wandb
    except ImportError:
        print("wandb not installed; skipping wandb logging. Install with `pip install wandb`.")
        return None
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        mode=args.wandb_mode,
        config=vars(args),
        dir=str(args.output_dir) if args.output_dir else None,
    )
    return run


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


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_param_groups(model, args) -> list[dict[str, Any]]:
    named_params = list(model.named_parameters())
    backbone_params = [param for name, param in named_params if param.requires_grad and "backbone" in name]
    other_params = [param for name, param in named_params if param.requires_grad and "backbone" not in name]

    param_dicts: list[dict[str, Any]] = []
    if other_params:
        param_dicts.append({"params": other_params, "lr": args.lr})
    if backbone_params:
        param_dicts.append({"params": backbone_params, "lr": args.lr_backbone})
    return param_dicts


def _save_checkpoint(output_dir: Path, payload: dict[str, Any], epoch: int) -> None:
    utils.save_on_master(payload, output_dir / "checkpoint.pth")
    if (epoch + 1) % 5 == 0:
        utils.save_on_master(payload, output_dir / f"checkpoint{epoch:04d}.pth")


def _load_model_checkpoint(model, checkpoint_path: str | None) -> dict[str, Any] | None:
    if not checkpoint_path:
        return None

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    return checkpoint if isinstance(checkpoint, dict) else None


@torch.no_grad()
def run_test_inference(model, postprocessor, data_loader, device, output_path: Path, debug_max_iters=None):
    model.eval()
    predictions: list[dict[str, float | int | list[float]]] = []

    for iteration, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device, non_blocking=True)
        target_sizes = torch.stack([target["orig_size"] for target in targets], dim=0).to(device, non_blocking=True)

        outputs = model(samples, None)
        results = postprocessor(outputs, target_sizes=target_sizes)

        for target, result in zip(targets, results):
            image_id = int(target["image_id"].item())
            boxes_xywh = box_ops.box_xyxy_to_xywh(result["boxes"])

            for score, label, box in zip(result["scores"], result["labels"], boxes_xywh):
                score_value = float(score.item())
                category_id = int(label.item())
                if category_id <= 0 or score_value <= 0.05:
                    continue

                predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [float(value) for value in box.tolist()],
                        "score": score_value,
                    }
                )

        if debug_max_iters is not None and iteration + 1 >= debug_max_iters:
            break

    predictions = _gather_object_list(predictions)
    if utils.is_main_process():
        output_path.write_text(json.dumps(predictions, indent=2))
        print(f"Saved {len(predictions)} predictions to {output_path}")

    return predictions


def get_parser():
    return get_config_parser()


def main(cli_args: list[str] | None = None):
    parser = get_parser()
    args = parser.parse_args(cli_args)

    utils.init_distributed_mode(args)
    args.coco_path = args.data_path

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        args.device = "cpu"

    device = torch.device(args.device if not args.distributed else f"cuda:{args.gpu}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed + utils.get_rank()
    _seed_everything(seed)

    model = build_dino(args)
    criterion, postprocessor = build_criterion_and_postprocess(args, num_classes=args.num_classes)
    model.to(device)
    criterion.to(device)

    model_without_ddp = model
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    n_parameters = sum(param.numel() for param in model_without_ddp.parameters() if param.requires_grad)
    print(f"Trainable params: {n_parameters / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(_build_param_groups(model_without_ddp, args), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop, gamma=0.1)
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    use_fp16_scaler = args.amp and device.type == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler(device.type, enabled=use_fp16_scaler)

    checkpoint = None
    if args.resume:
        checkpoint = _load_model_checkpoint(model_without_ddp, args.resume)
    elif args.test and args.checkpoint:
        checkpoint = _load_model_checkpoint(model_without_ddp, args.checkpoint)

    if checkpoint is not None and not args.eval and not args.test:
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "lr_scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if "epoch" in checkpoint:
            args.start_epoch = int(checkpoint["epoch"]) + 1
        if scaler.is_enabled() and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test:
        if not args.checkpoint:
            raise ValueError("--test requires --checkpoint")
        dataset_test = build_dataset("test", args)
        sampler_test = DistributedSampler(dataset_test, shuffle=False) if args.distributed else SequentialSampler(dataset_test)
        data_loader_test = DataLoader(
            dataset_test,
            batch_size=args.batch_size,
            sampler=sampler_test,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=utils.collate_fn,
        )
        run_test_inference(model, postprocessor, data_loader_test, device, output_dir / "pred.json", args.debug_max_iters)
        return

    dataset_train = build_dataset("train", args)
    dataset_val = build_dataset("val", args)
    base_ds = COCO(str(Path(args.data_path) / "valid.json"))

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train, shuffle=True)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = RandomSampler(dataset_train)
        sampler_val = SequentialSampler(dataset_val)

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        sampler=sampler_train,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=utils.collate_fn,
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        sampler=sampler_val,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=utils.collate_fn,
    )

    if args.eval:
        eval_stats, _ = evaluate(
            model,
            criterion,
            postprocessor,
            data_loader_val,
            base_ds,
            device,
            output_dir=output_dir,
            debug_max_iters=args.debug_max_iters,
        )
        if utils.is_main_process():
            print(json.dumps(eval_stats, indent=2))
        return

    print("Start training")
    log_path = output_dir / "log.txt"
    wandb_run = _init_wandb(args)
    global_step = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            max_norm=args.clip_max_norm,
            scaler=scaler if scaler.is_enabled() else None,
            debug_max_iters=args.debug_max_iters,
            wandb_run=wandb_run,
            wandb_log_freq=args.wandb_log_freq,
            global_step_start=global_step,
            amp_dtype=amp_dtype if args.amp and device.type == "cuda" else torch.float32,
        )
        global_step += int(train_stats.get("_num_iters", 0))
        lr_scheduler.step()

        eval_stats, _ = evaluate(
            model,
            criterion,
            postprocessor,
            data_loader_val,
            base_ds,
            device,
            output_dir=output_dir,
            debug_max_iters=args.debug_max_iters,
        )

        if wandb_run is not None and utils.is_main_process():
            wandb_payload = {"epoch": epoch, "train/lr_end": optimizer.param_groups[0]["lr"]}
            for key, value in eval_stats.items():
                if isinstance(value, (int, float)):
                    wandb_payload[f"val/{key}"] = float(value)
            wandb_run.log(wandb_payload, step=global_step)

        checkpoint_payload = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        }
        _save_checkpoint(output_dir, checkpoint_payload, epoch)

        if utils.is_main_process():
            log_stats = {
                "epoch": epoch,
                "n_parameters": n_parameters,
                **{f"train_{key}": value for key, value in train_stats.items() if not key.startswith("_")},
                **{f"val_{key}": value for key, value in eval_stats.items()},
            }
            with log_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(log_stats) + "\n")

    if wandb_run is not None:
        wandb_run.finish()
    print("Training completed")


if __name__ == "__main__":
    main()
