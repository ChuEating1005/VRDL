"""Dataset utilities for VRDL HW3 colored cell instance segmentation."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset
from torchvision.ops import masks_to_boxes


@dataclass(frozen=True)
class TestImageInfo:
    file_name: str
    image_id: int
    height: int
    width: int


def read_rgb_tif(path: str | Path) -> np.ndarray:
    """Read a TIFF image and drop the constant alpha channel if present."""
    image = tifffile.imread(path)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.shape[-1] > 3:
        image = image[..., :3]
    return image.astype(np.uint8, copy=False)


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert HWC uint8 RGB image to CHW float tensor in [0, 1]."""
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255.0


def load_test_infos(data_root: str | Path) -> list[TestImageInfo]:
    with open(Path(data_root) / "test_image_name_to_ids.json", encoding="utf-8") as f:
        rows = json.load(f)
    return [
        TestImageInfo(
            file_name=row["file_name"],
            image_id=int(row["id"]),
            height=int(row["height"]),
            width=int(row["width"]),
        )
        for row in rows
    ]


class HW3CellsDataset(Dataset):
    """VRDL HW3 training dataset.

    Each sample directory contains `image.tif` and optional `class1.tif` ...
    `class4.tif`. In class masks, every non-zero unique value is one instance.
    """

    def __init__(
        self,
        data_root: str | Path = "data",
        split: str = "train",
        val_ratio: float = 0.2,
        seed: int = 42,
        transforms: Any | None = None,
        copy_paste_prob: float = 0.0,
        copy_paste_classes: tuple[int, ...] = (3, 4),
    ) -> None:
        self.data_root = Path(data_root)
        self.train_root = self.data_root / "train"
        self.transforms = transforms
        self.copy_paste_prob = copy_paste_prob
        self.copy_paste_classes = set(copy_paste_classes)
        all_ids = sorted(p.name for p in self.train_root.iterdir() if p.is_dir())
        rng = random.Random(seed)
        shuffled = all_ids[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_ratio)))
        val_ids = set(shuffled[:n_val])
        if split == "train":
            self.sample_ids = [sid for sid in all_ids if sid not in val_ids]
        elif split == "val":
            self.sample_ids = [sid for sid in all_ids if sid in val_ids]
        elif split == "all":
            self.sample_ids = all_ids
        else:
            raise ValueError(f"Unsupported split: {split}")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image, target = self._load_raw(index)
        if self.copy_paste_prob > 0 and random.random() < self.copy_paste_prob:
            image, target = self._copy_paste(image, target)
        if self.transforms is not None:
            transformed = self.transforms(
                image=image,
                masks=[m.numpy() for m in target["masks"]],
                bboxes=target["boxes"].tolist(),
                labels=target["labels"].tolist(),
            )
            image = transformed["image"]
            masks = np.asarray(transformed["masks"], dtype=np.uint8)
            if masks.size == 0:
                masks = np.zeros((0, image.shape[0], image.shape[1]), dtype=np.uint8)
            boxes = np.asarray(transformed["bboxes"], dtype=np.float32).reshape(-1, 4)
            labels = np.asarray(transformed["labels"], dtype=np.int64)
            target = self._build_target(
                masks=masks,
                labels=labels,
                image_id=int(target["image_id"].item()),
            )
            if len(boxes) == len(target["boxes"]):
                target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
                target = self._filter_empty_target(target)
        return image_to_tensor(image), target

    def _load_raw(self, index: int) -> tuple[np.ndarray, dict[str, torch.Tensor]]:
        sample_id = self.sample_ids[index]
        sample_dir = self.train_root / sample_id
        image = read_rgb_tif(sample_dir / "image.tif")
        masks: list[np.ndarray] = []
        labels: list[int] = []
        for cls in range(1, 5):
            mask_path = sample_dir / f"class{cls}.tif"
            if not mask_path.exists():
                continue
            class_mask = tifffile.imread(mask_path)
            obj_ids = np.unique(class_mask)
            obj_ids = obj_ids[obj_ids != 0]
            for obj_id in obj_ids:
                instance = (class_mask == obj_id).astype(np.uint8)
                if instance.any():
                    masks.append(instance)
                    labels.append(cls)
        if masks:
            masks_arr = np.stack(masks, axis=0)
            labels_arr = np.asarray(labels, dtype=np.int64)
        else:
            h, w = image.shape[:2]
            masks_arr = np.zeros((0, h, w), dtype=np.uint8)
            labels_arr = np.zeros((0,), dtype=np.int64)
        target = self._build_target(masks_arr, labels_arr, index)
        return image, target

    def _build_target(
        self,
        masks: np.ndarray,
        labels: np.ndarray,
        image_id: int,
    ) -> dict[str, torch.Tensor]:
        masks_t = torch.as_tensor(masks, dtype=torch.uint8)
        labels_t = torch.as_tensor(labels, dtype=torch.int64)
        if masks_t.numel() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes = masks_to_boxes(masks_t).float()
        area = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        target = {
            "boxes": boxes,
            "labels": labels_t,
            "masks": masks_t,
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "area": area.float(),
            "iscrowd": torch.zeros((len(labels_t),), dtype=torch.int64),
        }
        return self._filter_empty_target(target)

    @staticmethod
    def _filter_empty_target(target: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        boxes = target["boxes"]
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]) if len(boxes) else torch.zeros((0,), dtype=torch.bool)
        for key in ("boxes", "labels", "masks", "area", "iscrowd"):
            target[key] = target[key][keep]
        return target

    def _copy_paste(
        self,
        image: np.ndarray,
        target: dict[str, torch.Tensor],
    ) -> tuple[np.ndarray, dict[str, torch.Tensor]]:
        """Lightweight same-canvas copy-paste for rare classes."""
        labels = target["labels"].numpy()
        rare_indices = [i for i, label in enumerate(labels) if int(label) in self.copy_paste_classes]
        if not rare_indices:
            return image, target
        h, w = image.shape[:2]
        masks = target["masks"].numpy().copy()
        new_masks = [m for m in masks]
        new_labels = [int(x) for x in labels]
        image_out = image.copy()
        for idx in random.sample(rare_indices, k=min(3, len(rare_indices))):
            ys, xs = np.where(masks[idx] > 0)
            if len(xs) == 0:
                continue
            x1, x2 = xs.min(), xs.max() + 1
            y1, y2 = ys.min(), ys.max() + 1
            obj_w, obj_h = x2 - x1, y2 - y1
            if obj_w <= 1 or obj_h <= 1 or obj_w >= w or obj_h >= h:
                continue
            nx = random.randint(0, max(0, w - obj_w))
            ny = random.randint(0, max(0, h - obj_h))
            patch_mask = masks[idx, y1:y2, x1:x2].astype(bool)
            patch_img = image[y1:y2, x1:x2]
            image_out[ny:ny + obj_h, nx:nx + obj_w][patch_mask] = patch_img[patch_mask]
            pasted = np.zeros((h, w), dtype=np.uint8)
            pasted[ny:ny + obj_h, nx:nx + obj_w] = patch_mask.astype(np.uint8)
            new_masks.append(pasted)
            new_labels.append(int(labels[idx]))
        masks_arr = np.stack(new_masks, axis=0).astype(np.uint8)
        labels_arr = np.asarray(new_labels, dtype=np.int64)
        return image_out, self._build_target(masks_arr, labels_arr, int(target["image_id"].item()))


class HW3TestDataset(Dataset):
    """Test split for submission generation."""

    def __init__(self, data_root: str | Path = "data") -> None:
        self.data_root = Path(data_root)
        self.test_root = self.data_root / "test_release"
        self.infos = load_test_infos(self.data_root)

    def __len__(self) -> int:
        return len(self.infos)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, TestImageInfo]:
        info = self.infos[index]
        image = read_rgb_tif(self.test_root / info.file_name)
        return image_to_tensor(image), info


def collate_fn(batch: list[tuple[Any, Any]]) -> tuple[list[Any], list[Any]]:
    return tuple(zip(*batch))  # type: ignore[return-value]
