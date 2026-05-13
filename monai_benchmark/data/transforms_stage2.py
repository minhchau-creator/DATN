#!/usr/bin/env python3
"""
Stage-2 Transforms — crop around GT pancreas+tumor mask before patch extraction.

Sự khác biệt so với transforms.py (stage 1):
  - CropForegroundd dùng source_key="label" thay vì "image"
    → crop chỉ quanh vùng pancreas+tumor (margin 20 voxel)
  - RandCropByPosNegLabeld: pos=3, neg=1  (oversample foreground mạnh hơn)
  - Kết quả: mỗi patch 96³ chứa tỉ lệ tumor/background cao hơn nhiều
"""

import logging
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ScaleIntensityRanged,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    SpatialPadd,
    CropForegroundd,
    ToTensord,
)

logger = logging.getLogger(__name__)

HU_WINDOW   = [-74, 140]
PATCH_SIZE  = [96, 96, 96]
MASK_MARGIN = 20   # voxels around GT bounding box


def get_train_transforms_stage2(hu_window=None, patch_size=None):
    """
    Training pipeline for stage 2.
    Crops around the GT pancreas+tumor mask before patch extraction so that
    patches are densely packed with relevant anatomy instead of background.
    """
    if hu_window is None:
        hu_window = HU_WINDOW
    if patch_size is None:
        patch_size = PATCH_SIZE

    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=hu_window[0], a_max=hu_window[1],
            b_min=0.0, b_max=1.0, clip=True,
        ),
        # Crop around GT pancreas+tumor mask (label > 0) with margin
        CropForegroundd(
            keys=["image", "label"],
            source_key="label",
            select_fn=lambda x: x > 0,
            margin=MASK_MARGIN,
        ),
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
        # pos=3: 75% patches centred on a foreground voxel → denser tumor sampling
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=3,
            neg=1,
            num_samples=4,
            image_key="image",
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.3, max_k=3),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.2),
        ToTensord(keys=["image", "label"]),
    ])

    logger.info(f"Stage-2 train transforms: HU={hu_window}, patch={patch_size}, margin={MASK_MARGIN}")
    return transforms


def get_val_transforms_stage2(hu_window=None):
    """
    Validation pipeline for stage 2.
    Crops full volume around GT mask (no random patch) — sliding window done in loop.
    """
    if hu_window is None:
        hu_window = HU_WINDOW

    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=hu_window[0], a_max=hu_window[1],
            b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForegroundd(
            keys=["image", "label"],
            source_key="label",
            select_fn=lambda x: x > 0,
            margin=MASK_MARGIN,
        ),
        SpatialPadd(keys=["image", "label"], spatial_size=PATCH_SIZE),
        ToTensord(keys=["image", "label"]),
    ])

    logger.info(f"Stage-2 val transforms: HU={hu_window}, margin={MASK_MARGIN}")
    return transforms


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    get_train_transforms_stage2()
    get_val_transforms_stage2()
    print("Stage-2 transforms OK.")
