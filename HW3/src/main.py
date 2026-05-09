"""Unified HW3 entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.datasets.hw3_cells import HW3CellsDataset, HW3TestDataset, collate_fn
from src.datasets.transforms import build_train_transforms
from src.engine import evaluate_ap50, predict_submission, train_one_epoch
from src.models.build import build_model, count_trainable_parameters
from src.util.coco import dataset_to_coco
from src.util.misc import ensure_dir, get_device, save_checkpoint, save_json, seed_worker, set_seed


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VRDL HW3 Mask R-CNN pipeline")
    parser.add_argument("--mode", choices=["train", "eval", "infer"], default="train")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="outputs/maskrcnn_r50")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--clip-grad-norm", type=float, default=0.0)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default="vrdl-hw3")
    parser.add_argument("--run-name", default="", help="Weights & Biases run name")

    # Architecture / enhancement ablations (requested 1~5 switches).
    parser.add_argument("--backbone", choices=["resnet50", "resnet101", "convnext_tiny"], default="resnet50")
    parser.add_argument("--trainable-backbone-layers", type=int, default=3)
    parser.add_argument("--use-pafpn", action="store_true", help="PANet-style bottom-up FPN augmentation")
    parser.add_argument("--mask-loss", choices=["bce", "dice"], default="bce")
    parser.add_argument("--small-anchors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--detections-per-img", type=int, default=1000)
    parser.add_argument("--box-score-thresh", type=float, default=0.05)

    # Data / competition tricks.
    parser.add_argument("--strong-aug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-scale", type=float, default=0.6)
    parser.add_argument("--max-scale", type=float, default=1.6)
    parser.add_argument("--max-size", type=int, default=1333)
    parser.add_argument("--copy-paste-prob", type=float, default=0.0)
    parser.add_argument("--repeat-threshold", type=float, default=0.0, help=">0 enables rare-class weighted sampling")

    # Inference / submission.
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--tta-hflip", action="store_true")
    parser.add_argument("--tta-scales", default="1.0", help="Comma-separated scales, e.g. 0.83,1.0,1.2")
    parser.add_argument("--submission", default="outputs/test-results.json")
    parser.add_argument("--debug-samples", type=int, default=0, help="Use a tiny subset for smoke tests")
    return parser.parse_args()


def make_loader(dataset, args: argparse.Namespace, train: bool) -> DataLoader:  # noqa: ANN001
    sampler = None
    shuffle = train
    if train and args.repeat_threshold > 0:
        weights = make_repeat_weights(dataset, rare_classes={3, 4}, rare_weight=1.0 + args.repeat_threshold)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=args.batch_size if train else 1,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        pin_memory=torch.cuda.is_available(),
    )


def make_repeat_weights(dataset: HW3CellsDataset, rare_classes: set[int], rare_weight: float) -> list[float]:
    weights: list[float] = []
    for idx in range(len(dataset)):
        _, target = dataset._load_raw(idx)  # noqa: SLF001 - intentional cheap metadata computation
        labels = set(int(x) for x in target["labels"].tolist())
        weights.append(rare_weight if labels & rare_classes else 1.0)
    return weights


def build_datasets(args: argparse.Namespace) -> tuple[HW3CellsDataset, HW3CellsDataset]:
    transforms = build_train_transforms(
        strong_aug=args.strong_aug,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        max_size=args.max_size,
    )
    train_dataset = HW3CellsDataset(
        args.data_root,
        split="train",
        val_ratio=args.val_ratio,
        seed=args.seed,
        transforms=transforms,
        copy_paste_prob=args.copy_paste_prob,
    )
    val_dataset = HW3CellsDataset(args.data_root, split="val", val_ratio=args.val_ratio, seed=args.seed)
    if args.debug_samples > 0:
        train_dataset.sample_ids = train_dataset.sample_ids[: args.debug_samples]
        val_dataset.sample_ids = val_dataset.sample_ids[: max(1, args.debug_samples)]
    return train_dataset, val_dataset


def make_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    model = build_model(
        backbone=args.backbone,
        trainable_backbone_layers=args.trainable_backbone_layers,
        use_pafpn=args.use_pafpn,
        small_anchors=args.small_anchors,
        detections_per_img=args.detections_per_img,
        box_score_thresh=args.box_score_thresh,
        mask_loss=args.mask_loss,
    )
    print(f"Trainable params: {count_trainable_parameters(model) / 1e6:.2f}M")
    return model.to(device)


def load_checkpoint_if_needed(model: torch.nn.Module, checkpoint: str, device: torch.device) -> None:
    if not checkpoint:
        return
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state.get("model", state))


def init_wandb(args: argparse.Namespace, trainable_params: int) -> Any | None:
    if not args.wandb:
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        name=args.run_name or None,
        config={**vars(args), "trainable_params": trainable_params},
    )


def run_train(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    train_dataset, val_dataset = build_datasets(args)
    train_loader = make_loader(train_dataset, args, train=True)
    val_loader = make_loader(val_dataset, args, train=False)
    model = make_model(args, device)
    trainable_params = count_trainable_parameters(model)
    wandb_run = init_wandb(args, trainable_params)
    load_checkpoint_if_needed(model, args.checkpoint, device)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[max(1, args.epochs * 2 // 3)], gamma=0.1)
    coco_gt = dataset_to_coco(val_dataset)
    best_ap50 = -1.0
    for epoch in range(args.epochs):
        losses = train_one_epoch(model, train_loader, optimizer, device, epoch, max_norm=args.clip_grad_norm)
        scheduler.step()
        metrics = evaluate_ap50(
            model,
            val_loader,
            device,
            coco_gt,
            mask_threshold=args.mask_threshold,
            score_threshold=args.score_threshold,
        )
        state = {"model": model.state_dict(), "epoch": epoch, "args": vars(args), "metrics": metrics}
        save_checkpoint(state, output_dir / "last.pth")
        if metrics["ap50"] > best_ap50:
            best_ap50 = metrics["ap50"]
            save_checkpoint(state, output_dir / "best.pth")
        log_row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train/{k}": v for k, v in losses.items()},
            **{f"val/{k}": v for k, v in metrics.items()},
            "val/best_ap50": best_ap50,
        }
        if wandb_run is not None:
            wandb_run.log(log_row, step=epoch)
        print(log_row)
    if wandb_run is not None:
        wandb_run.finish()


def run_eval(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    _, val_dataset = build_datasets(args)
    val_loader = make_loader(val_dataset, args, train=False)
    model = make_model(args, device)
    load_checkpoint_if_needed(model, args.checkpoint, device)
    metrics = evaluate_ap50(
        model,
        val_loader,
        device,
        dataset_to_coco(val_dataset),
        mask_threshold=args.mask_threshold,
        score_threshold=args.score_threshold,
    )
    print(metrics)


def run_infer(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    dataset = HW3TestDataset(args.data_root)
    if args.debug_samples > 0:
        dataset.infos = dataset.infos[: args.debug_samples]
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    model = make_model(args, device)
    load_checkpoint_if_needed(model, args.checkpoint, device)
    records = predict_submission(
        model,
        loader,
        device,
        mask_threshold=args.mask_threshold,
        score_threshold=args.score_threshold,
        tta_hflip=args.tta_hflip,
        tta_scales=tuple(float(x) for x in args.tta_scales.split(",") if x),
    )
    save_json(records, args.submission)
    print(f"Saved {len(records)} records to {args.submission}")


def main() -> None:
    args = get_args()
    set_seed(args.seed)
    if args.mode == "train":
        run_train(args)
    elif args.mode == "eval":
        run_eval(args)
    elif args.mode == "infer":
        run_infer(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
