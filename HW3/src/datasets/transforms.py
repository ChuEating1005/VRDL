"""Albumentations transforms for HW3."""

from __future__ import annotations

import albumentations as A


def build_train_transforms(
    strong_aug: bool = True,
    min_scale: float = 0.6,
    max_scale: float = 1.6,
    max_size: int = 1333,
    deformation_prob: float = 0.0,
) -> A.Compose:
    transforms: list[A.BasicTransform] = [
        A.LongestMaxSize(max_size=max_size, p=1.0),
        A.RandomScale(scale_limit=(min_scale - 1.0, max_scale - 1.0), p=0.8),
        # RandomScale can enlarge images after the first cap. Apply a final
        # hard cap so dense instance masks never reach the model above max_size.
        A.LongestMaxSize(max_size=max_size, p=1.0),
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
    if deformation_prob > 0:
        transforms.append(
            A.OneOf(
                [
                    A.ElasticTransform(alpha=120, sigma=6, p=1.0),
                    A.GridDistortion(p=1.0),
                    A.OpticalDistortion(distort_limit=0.2, p=1.0),
                ],
                p=deformation_prob,
            )
        )
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_area=1, min_visibility=0.1),
    )
