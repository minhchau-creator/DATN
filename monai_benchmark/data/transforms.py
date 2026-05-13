#!/usr/bin/env python3
"""
Data Preprocessing and Augmentation Transforms
Dict-based MONAI pipeline for all 8 models.
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

# Default dataset location — update if different
DATA_DIR = "/home/minhc/DATN/dataset/Task_7"


def build_data_list(file_names, data_dir=DATA_DIR):
    """
    Convert fold filenames (e.g. 'pancreas_001.nii') to MONAI data dicts.
    Returns list of {'image': str, 'label': str}.
    """
    data_list = []
    for fname in file_names:
        base = fname.replace(".nii.gz", "").replace(".nii", "")
        data_list.append({
            "image": f"{data_dir}/imagesTr/{base}.nii.gz",
            "label": f"{data_dir}/labelsTr/{base}.nii.gz",
        })
    return data_list


def get_train_transforms(hu_window=None, patch_size=None, augmentation_config=None):
    """
    Training transforms with random patch extraction and augmentation.
    Uses dict-based transforms with 'image' and 'label' keys.
    """
    if hu_window is None:
        hu_window = [-74, 140]
    if patch_size is None:
        patch_size = [96, 96, 96]
    if augmentation_config is None:
        augmentation_config = {
            "flip_prob": 0.5,
            "rotate_prob": 0.3,
            "scale_prob": 0.2,
        }

    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=hu_window[0],
            a_max=hu_window[1],
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        # Crop to foreground before patch extraction to reduce background sampling
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # Pad if any axis is smaller than the crop size (e.g. thin-slice volumes)
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
        # Random patch extraction: pos=1 means at least 1 patch contains label voxel
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
        ),
        RandFlipd(keys=["image", "label"], prob=augmentation_config["flip_prob"], spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=augmentation_config["flip_prob"], spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=augmentation_config["flip_prob"], spatial_axis=2),
        RandRotate90d(
            keys=["image", "label"],
            prob=augmentation_config["rotate_prob"],
            max_k=3,
        ),
        RandScaleIntensityd(
            keys=["image"],
            factors=0.1,
            prob=augmentation_config["scale_prob"],
        ),
        ToTensord(keys=["image", "label"]),
    ])

    logger.info(f"Train transforms: HU={hu_window}, patch={patch_size}")
    return transforms


def get_val_transforms(hu_window=None):
    """
    Validation transforms — no patch extraction, full volume.
    Sliding window inference is done in the training loop.
    """
    if hu_window is None:
        hu_window = [-74, 140]

    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=hu_window[0],
            a_max=hu_window[1],
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        ToTensord(keys=["image", "label"]),
    ])

    logger.info(f"Val transforms: HU={hu_window}")
    return transforms


def get_inference_transforms(hu_window=None):
    """
    Inference transforms for a single image (no label key).
    """
    if hu_window is None:
        hu_window = [-74, 140]

    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=hu_window[0],
            a_max=hu_window[1],
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        ToTensord(keys=["image"]),
    ])

    logger.info(f"Inference transforms: HU={hu_window}")
    return transforms


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_tf = get_train_transforms()
    val_tf = get_val_transforms()
    inf_tf = get_inference_transforms()
    print("All transforms created successfully.")
