"""Albumentations transforms for HW3."""

from __future__ import annotations

import albumentations as A


def build_train_transforms(
    strong_aug: bool = True,
    min_scale: float = 0.6,
    max_scale: float = 1.6,
    max_size: int = 1333,
) -> A.Compose:
    transforms: list[A.BasicTransform] = [
        A.LongestMaxSize(max_size=max_size, p=1.0),
        A.RandomScale(scale_limit=(min_scale - 1.0, max_scale - 1.0), p=0.8),
        A.PadIfNeeded(min_height=128, min_width=128, border_mode=0, fill=0, fill_mask=0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]
    if strong_aug:
        transforms.extend(
            [
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(p=1.0),
                        A.HueSaturationValue(p=1.0),
                        A.CLAHE(p=1.0),
                    ],
                    p=0.8,
                ),
                A.OneOf([A.GaussianBlur(p=1.0), A.MotionBlur(p=1.0)], p=0.2),
            ]
        )
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_area=1, min_visibility=0.1),
    )
