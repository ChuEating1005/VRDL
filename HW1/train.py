"""Training script for HW1 Image Classification."""

import argparse
import os
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms_v2
import wandb
from tqdm import tqdm

from dataset import ImageClassificationDataset
from model import build_model, NUM_CLASSES
from utils import AverageMeter, accuracy, save_checkpoint, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="HW1 Image Classification Training")
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Root directory of the dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet50",
        choices=[
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
            "resnext101",
            "resnest200",
        ],
        help="Architecture (resnet18/34/50/101/152, resnext101, resnest200)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout rate before the final FC layer",
    )
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "plateau", "onecycle", "warm_restarts"],
        help="Learning rate scheduler type",
    )

    return parser.parse_args()


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    scheduler=None,
    cutmix_fn=None,
    mixup_fn=None,
):
    """Run one training epoch.

    Consider adding:
        - Gradient clipping (torch.nn.utils.clip_grad_norm_)
        - Learning rate warmup
        - Mixed precision training (torch.amp) for faster training on RTX 4090

    Args:
        model: The neural network.
        dataloader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: torch.device (cuda/cpu).

    Returns:
        Tuple of (average_loss, average_accuracy) for this epoch.
    """
    model.train()
    losses = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)

        # 25% CutMix, 25% Mixup, 50% none
        mixed = False
        if cutmix_fn is not None and mixup_fn is not None:
            r = random.random()
            if r < 0.25:
                images, labels = cutmix_fn(images, labels)
                mixed = True
            elif r < 0.5:
                images, labels = mixup_fn(images, labels)
                mixed = True

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        losses.update(loss.item(), images.size(0))
        if not mixed:
            acc = accuracy(outputs, labels)
            acc_meter.update(acc, images.size(0))

    return losses.avg, acc_meter.avg


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Run validation.

    Args:
        model: The neural network.
        dataloader: Validation DataLoader.
        criterion: Loss function.
        device: torch.device.

    Returns:
        Tuple of (average_loss, average_accuracy).
    """
    model.eval()
    losses = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in tqdm(dataloader, desc="Validation"):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        acc = accuracy(outputs, labels)
        losses.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))

    return losses.avg, acc_meter.avg


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    train_dataset = ImageClassificationDataset(
        root_dir=os.path.join(args.data_dir, "train"),
        split="train",
        img_size=args.img_size,
    )
    val_dataset = ImageClassificationDataset(
        root_dir=os.path.join(args.data_dir, "val"),
        split="val",
        img_size=args.img_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")

    # --- Model ---
    model = build_model(arch=args.arch, dropout=args.dropout)
    model = model.to(device)

    # --- Loss, Optimizer, Scheduler ---
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    cutmix_fn = transforms_v2.CutMix(num_classes=NUM_CLASSES)
    mixup_fn = transforms_v2.MixUp(num_classes=NUM_CLASSES)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
        )
    elif args.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy="cos",
        )
    elif args.scheduler == "warm_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

    # --- Weights & Biases ---
    proj_name = f"{args.arch}_lr{args.lr}_bs{args.batch_size}_{args.scheduler}"
    if args.dropout > 0.0:
        proj_name += f"_dropout{args.dropout}"
    wandb.init(
        project="nycu-dlcv-hw1",
        config=vars(args),
        name=proj_name,
    )

    # --- Training Loop ---
    best_val_acc = 0.0
    start_epoch = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        print(f"Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.4f}")

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'=' * 60}")
        print(f"Epoch [{epoch + 1}/{args.epochs}]")
        print(f"{'=' * 60}")

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scheduler if args.scheduler == "onecycle" else None,
            cutmix_fn=cutmix_fn,
            mixup_fn=mixup_fn,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        epoch_time = time.time() - t0

        # Step scheduler
        if args.scheduler == "cosine":
            scheduler.step()
        elif args.scheduler == "plateau":
            scheduler.step(val_acc)
        elif args.scheduler == "warm_restarts":
            scheduler.step()

        # Logging
        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )
        log_dict = {
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch": epoch,
        }
        wandb.log(log_dict)

        # Save best model
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "args": vars(args),
            },
            is_best=is_best,
            output_dir=args.output_dir,
        )

    wandb.finish()
    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
