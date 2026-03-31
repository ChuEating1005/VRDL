"""Inference script: generate prediction.csv for CodaBench submission."""

import argparse
import os

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
from PIL import Image
from pathlib import Path

from model import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="HW1 Inference / Prediction")
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Root directory of the dataset"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)"
    )
    parser.add_argument(
        "--output", type=str, default="prediction.csv", help="Output CSV filename"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout rate before the final FC layer",
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
    )
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--tta", action="store_true", help="Enable Test-Time Augmentation"
    )
    return parser.parse_args()


def build_tta_transforms(img_size):
    """Build a list of TTA transforms: original + multi-scale + flips."""
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transforms_list = []

    for scale in [img_size, int(img_size * 0.875), int(img_size * 1.125)]:
        transforms_list.append(
            T.Compose(
                [
                    T.Resize(scale + 32),
                    T.CenterCrop(scale),
                    T.Resize(img_size),
                    T.ToTensor(),
                    normalize,
                ]
            )
        )
        transforms_list.append(
            T.Compose(
                [
                    T.Resize(scale + 32),
                    T.CenterCrop(scale),
                    T.Resize(img_size),
                    T.RandomHorizontalFlip(p=1.0),
                    T.ToTensor(),
                    normalize,
                ]
            )
        )

    return transforms_list


class TestDatasetForTTA(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = sorted(
            [
                p
                for p in self.root_dir.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            ]
        )
        self.image_names = [p.name for p in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img), self.image_names[idx]


@torch.no_grad()
def predict(model, dataloader, device):
    model.eval()
    predictions = []

    for images, filenames in tqdm(dataloader, desc="Predicting"):
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().tolist()

        for fname, pred in zip(filenames, preds):
            predictions.append((fname, pred))

    return predictions


@torch.no_grad()
def predict_tta(model, test_dir, tta_transforms, batch_size, num_workers, device):
    model.eval()

    all_probs = None
    filenames = None

    for i, tfm in enumerate(tta_transforms):
        dataset = TestDatasetForTTA(test_dir, transform=tfm)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        if filenames is None:
            filenames = dataset.image_names

        probs_list = []
        for images, _ in tqdm(loader, desc=f"TTA {i + 1}/{len(tta_transforms)}"):
            images = images.to(device)
            logits = model(images)
            probs_list.append(F.softmax(logits, dim=1).cpu())

        probs = torch.cat(probs_list, dim=0)
        if all_probs is None:
            all_probs = probs
        else:
            all_probs += probs

    all_probs /= len(tta_transforms)
    preds = all_probs.argmax(dim=1).tolist()

    return list(zip(filenames, preds))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(arch=args.arch, dropout=args.dropout)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    test_dir = os.path.join(args.data_dir, "test")

    if args.tta:
        tta_transforms = build_tta_transforms(args.img_size)
        print(f"TTA enabled: {len(tta_transforms)} augmentations")
        predictions = predict_tta(
            model,
            test_dir,
            tta_transforms,
            args.batch_size,
            args.num_workers,
            device,
        )
    else:
        from dataset import TestDataset

        test_dataset = TestDataset(root_dir=test_dir, img_size=args.img_size)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        print(f"Test set: {len(test_dataset)} images")
        predictions = predict(model, test_loader, device)

    df = pd.DataFrame(predictions, columns=["filename", "label"])
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} predictions to {args.output}")


if __name__ == "__main__":
    main()
