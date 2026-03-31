"""
Dataset module for HW1 Image Classification.

Handles loading images from the folder structure:
    data/train/{class_id}/*.jpg   (20,724 images, 100 classes)
    data/val/{class_id}/*.jpg     (300 images, 100 classes)
    data/test/*.jpg               (2,344 images, no labels)
"""

import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class ImageClassificationDataset(Dataset):
    """Dataset for train/val splits where images are organized by class folders.

    Directory structure expected:
        root_dir/{class_id}/*.jpg

    Attributes:
        root_dir: Path to the split directory (e.g., 'data/train').
        transform: Torchvision transforms to apply to each image.
        samples: List of (image_path, label) tuples.
        classes: Sorted list of class names (folder names).
        class_to_idx: Mapping from class name to integer label.
    """

    def __init__(self, root_dir: str, split: str, img_size: int = 224):
        self.root_dir = Path(root_dir)
        if split == "train":
            self.transform = T.Compose(
                [
                    T.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.autoaugment.TrivialAugmentWide(),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    T.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
                ]
            )
        else:
            self.transform = T.Compose(
                [
                    T.Resize(img_size + 32),
                    T.CenterCrop(img_size),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

        # Build class mapping: folder names → integer labels
        self.classes = sorted(
            [d.name for d in self.root_dir.iterdir() if d.is_dir()],
            key=lambda x: int(x),
        )
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Collect all (image_path, label) pairs
        self.samples = []
        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            label = self.class_to_idx[cls_name]
            for img_name in sorted(os.listdir(cls_dir)):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((cls_dir / img_name, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """Load and return a single sample."""
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class TestDataset(Dataset):
    """Dataset for test split where images are flat (no class subfolders).

    Directory structure expected:
        root_dir/*.jpg

    Attributes:
        root_dir: Path to the test directory (e.g., 'data/test').
        transform: Torchvision transforms to apply to each image.
        image_paths: Sorted list of image file paths.
        image_names: Corresponding filenames (used for submission CSV).
    """

    def __init__(self, root_dir: str, img_size: int = 224):
        self.root_dir = Path(root_dir)
        self.transform = T.Compose(
            [
                T.Resize(img_size + 32),
                T.CenterCrop(img_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.image_paths = sorted(
            [
                p
                for p in self.root_dir.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            ]
        )
        self.image_names = [p.name for p in self.image_paths]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """Load and return a single test sample."""
        img_path = self.image_paths[idx]
        img_name = self.image_names[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_name
