"""NYCU HW2 detection transforms.

Flow: COCO xywh -> dataset xyxy pixels -> spatial augments keep xyxy pixels ->
Normalize converts boxes to cxcywh normalized by current image size.
"""

import random
from typing import Iterable

import torch
import torchvision.transforms as tv_transforms
import torchvision.transforms.functional as F

from ..util.box_ops import box_xyxy_to_cxcywh


def _clone_target(target):
    return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in target.items()}


def _resize_image_size(image_size, size, max_size=None):
    width, height = image_size
    if max_size is not None:
        min_original = min(width, height)
        max_original = max(width, height)
        if max_original / min_original * size > max_size:
            size = int(round(max_size * min_original / max_original))

    if width < height:
        new_width = size
        new_height = int(round(size * height / width))
    else:
        new_height = size
        new_width = int(round(size * width / height))
    return new_height, new_width


def _filter_target(target, keep):
    for key in ("boxes", "labels", "area", "iscrowd"):
        if key in target:
            target[key] = target[key][keep]
    return target


def _crop(image, target, top, left, height, width, fallback_if_empty=False):
    cropped_image = F.crop(image, top, left, height, width)
    target = _clone_target(target)

    if "boxes" in target:
        boxes = target["boxes"]
        if boxes.numel() > 0:
            shifted_boxes = boxes.clone()
            shifted_boxes[:, [0, 2]] -= left
            shifted_boxes[:, [1, 3]] -= top
            centers = (shifted_boxes[:, :2] + shifted_boxes[:, 2:]) / 2
            keep = (
                (centers[:, 0] > 0)
                & (centers[:, 0] < width)
                & (centers[:, 1] > 0)
                & (centers[:, 1] < height)
            )
            if fallback_if_empty and boxes.shape[0] > 0 and keep.sum().item() == 0:
                return image, target

            shifted_boxes[:, 0::2].clamp_(min=0, max=width)
            shifted_boxes[:, 1::2].clamp_(min=0, max=height)
            target["boxes"] = shifted_boxes
            target = _filter_target(target, keep)
            if target["boxes"].numel() > 0:
                wh = (target["boxes"][:, 2:] - target["boxes"][:, :2]).clamp(min=0)
                target["area"] = wh[:, 0] * wh[:, 1]
        else:
            target["boxes"] = boxes.reshape(0, 4)

    target["size"] = torch.tensor([height, width], dtype=torch.int64)
    return cropped_image, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target

        image = F.hflip(image)
        target = _clone_target(target)
        width, _ = image.size
        if "boxes" in target and target["boxes"].numel() > 0:
            boxes = target["boxes"].clone()
            boxes[:, 0] = width - target["boxes"][:, 2]
            boxes[:, 2] = width - target["boxes"][:, 0]
            target["boxes"] = boxes
        return image, target


class RandomSelect:
    def __init__(self, t1, t2, p=0.5):
        self.t1 = t1
        self.t2 = t2
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            return self.t1(image, target)
        return self.t2(image, target)


class RandomResize:
    def __init__(self, sizes, max_size=None):
        self.sizes = list(sizes)
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.choice(self.sizes)
        orig_width, orig_height = image.size
        new_height, new_width = _resize_image_size(image.size, size, self.max_size)
        image = F.resize(image, [new_height, new_width], interpolation=F.InterpolationMode.BILINEAR)

        target = _clone_target(target)
        if "boxes" in target and target["boxes"].numel() > 0:
            ratio_width = new_width / orig_width
            ratio_height = new_height / orig_height
            boxes = target["boxes"].clone()
            boxes[:, 0::2] *= ratio_width
            boxes[:, 1::2] *= ratio_height
            target["boxes"] = boxes
            if "area" in target:
                target["area"] = target["area"] * (ratio_width * ratio_height)
        target["size"] = torch.tensor([new_height, new_width], dtype=torch.int64)
        return image, target


class RandomSizeCrop:
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        width, height = image.size
        crop_width_max = min(width, self.max_size)
        crop_height_max = min(height, self.max_size)
        crop_width_min = min(crop_width_max, self.min_size)
        crop_height_min = min(crop_height_max, self.min_size)

        crop_width = random.randint(crop_width_min, crop_width_max)
        crop_height = random.randint(crop_height_min, crop_height_max)
        top = random.randint(0, height - crop_height)
        left = random.randint(0, width - crop_width)
        return _crop(image, target, top, left, crop_height, crop_width, fallback_if_empty=True)


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        if isinstance(self.size, Iterable) and not isinstance(self.size, (str, bytes)):
            crop_height, crop_width = self.size
        else:
            crop_height = crop_width = self.size
        width, height = image.size
        crop_height = min(crop_height, height)
        crop_width = min(crop_width, width)
        top = max((height - crop_height) // 2, 0)
        left = max((width - crop_width) // 2, 0)
        return _crop(image, target, top, left, crop_height, crop_width, fallback_if_empty=False)


class ColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.transform = tv_transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, image, target):
        return self.transform(image), target


class ToTensor:
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        target = _clone_target(target)
        if "boxes" in target:
            boxes = target["boxes"]
            if boxes.numel() > 0:
                boxes = box_xyxy_to_cxcywh(boxes)
                height, width = target["size"].tolist()
                scale = torch.tensor([width, height, width, height], dtype=boxes.dtype)
                target["boxes"] = boxes / scale
            else:
                target["boxes"] = boxes.reshape(0, 4)
        return image, target


def make_nycu_transforms(image_set: str, args):
    normalize = Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    scales = getattr(args, "data_aug_scales", [320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640])
    max_size = getattr(args, "data_aug_max_size", 1024)

    if image_set == "train":
        return Compose([
            RandomSelect(
                RandomResize(scales, max_size=max_size),
                Compose([
                    RandomResize([400, 500, 600], max_size=max_size),
                    RandomSizeCrop(384, 600),
                    RandomResize(scales, max_size=max_size),
                ]),
            ),
            ColorJitter(0.4, 0.4, 0.4, 0.1),
            normalize,
        ])

    if image_set in {"val", "test"}:
        return Compose([
            RandomResize([640], max_size=max_size),
            normalize,
        ])

    raise ValueError(f"Unsupported image_set: {image_set}")
