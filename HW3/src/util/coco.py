"""Build in-memory COCO annotations from HW3 dataset."""

from __future__ import annotations

from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mask_util
from tqdm import tqdm


def dataset_to_coco(dataset) -> COCO:  # noqa: ANN001
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ann_id = 1
    for idx in tqdm(range(len(dataset)), desc="build coco gt"):
        image, target = dataset[idx]
        _, height, width = image.shape
        image_id = int(target["image_id"].item())
        images.append({"id": image_id, "height": height, "width": width, "file_name": str(idx)})
        masks = target["masks"].numpy()
        boxes = target["boxes"].numpy()
        labels = target["labels"].numpy()
        areas = target["area"].numpy()
        for mask, box, label, area in zip(masks, boxes, labels, areas):
            rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("utf-8")
            x1, y1, x2, y2 = box.tolist()
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(label),
                    "segmentation": rle,
                    "area": float(area),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    coco = COCO()
    coco.dataset = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": f"class{i}"} for i in range(1, 5)],
    }
    coco.createIndex()
    return coco
