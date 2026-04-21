from pathlib import Path

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from .transforms import make_nycu_transforms


class ConvertCocoPolysToMask:
    def __init__(self, return_masks=False):
        if return_masks:
            raise NotImplementedError("Masks are not supported for NYCU HW2 detection.")
        self.return_masks = return_masks

    def __call__(self, image, annotations, image_id):
        width, height = image.size

        annotations = [ann for ann in annotations if ann.get("iscrowd", 0) == 0 and ann.get("area", 0) > 1]
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        for ann in annotations:
            x, y, w, h = ann["bbox"]
            x0 = max(0.0, min(x, width))
            y0 = max(0.0, min(y, height))
            x1 = max(0.0, min(x + w, width))
            y1 = max(0.0, min(y + h, height))
            if x1 <= x0 or y1 <= y0:
                continue
            boxes.append([x0, y0, x1, y1])
            labels.append(ann["category_id"])
            areas.append((x1 - x0) * (y1 - y0))
            iscrowd.append(ann.get("iscrowd", 0))

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
            "orig_size": torch.tensor([height, width], dtype=torch.int64),
            "size": torch.tensor([height, width], dtype=torch.int64),
        }
        return image, target


class NYCUHW2Detection(Dataset):
    def __init__(self, img_folder: str, ann_file=None, transforms=None, return_masks=False):
        self.img_folder = Path(img_folder)
        self.ann_file = ann_file
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks=return_masks)

        if ann_file is not None:
            self.coco = COCO(ann_file)
            self.ids = sorted(self.coco.getImgIds())
            self.files = None
        else:
            self.coco = None
            self.ids = None
            self.files = sorted(
                [
                    path for path in self.img_folder.iterdir()
                    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
                ],
                key=lambda path: int(path.stem),
            )

    def __len__(self):
        return len(self.ids) if self.coco is not None else len(self.files)

    def __getitem__(self, idx):
        if self.coco is not None:
            image_id = self.ids[idx]
            image_info = self.coco.loadImgs(image_id)[0]
            image_path = self.img_folder / image_info["file_name"]
            annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))
            with Image.open(image_path) as img:
                image = img.convert("RGB")
            image, target = self.prepare(image, annotations, image_id)
        else:
            image_path = self.files[idx]
            image_id = int(image_path.stem)
            with Image.open(image_path) as img:
                image = img.convert("RGB")
            width, height = image.size
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "image_id": torch.tensor([image_id], dtype=torch.int64),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
                "orig_size": torch.tensor([height, width], dtype=torch.int64),
                "size": torch.tensor([height, width], dtype=torch.int64),
            }

        if self._transforms is not None:
            image, target = self._transforms(image, target)
        return image, target


def build(image_set: str, args):
    if image_set not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported image_set: {image_set}")

    coco_path = Path(getattr(args, "coco_path", "/project3/chueating/VRDL/HW2/data/nycu-hw2-data"))
    if image_set == "train":
        img_folder = coco_path / "train"
        ann_file = coco_path / "train.json"
    elif image_set == "val":
        img_folder = coco_path / "valid"
        ann_file = coco_path / "valid.json"
    else:
        img_folder = coco_path / "test"
        ann_file = None

    return NYCUHW2Detection(
        img_folder=str(img_folder),
        ann_file=str(ann_file) if ann_file is not None else None,
        transforms=make_nycu_transforms(image_set, args),
        return_masks=False,
    )
