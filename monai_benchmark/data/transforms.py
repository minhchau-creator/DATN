#!/usr/bin/env python3
"""
Data Preprocessing and Augmentation Transforms
Unified pipeline for all 8 models using MONAI transforms.
"""

import numpy as np
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, Reorient, Spacing,
    ScaleIntensityRanged, NormalizeIntensity, CropForeground,
    RandFlip, RandRotate90, RandRotate, RandAffine, RandScaleIntensity,
    EnsureTupleSize, Resize, SpatialPad, ToTensor
)
import logging

logger = logging.getLogger(__name__)


def get_train_transforms(hu_window=None, patch_size=None, augmentation_config=None):
    """
    Get training transforms with data augmentation.
    
    Args:
        hu_window: HU clipping range, e.g., [-74, 140]
        patch_size: Target patch size, e.g., [96, 96, 96]
        augmentation_config: Dict with augmentation probabilities and ranges
    
    Returns:
        Compose object with training transforms
    """
    
    if hu_window is None:
        hu_window = [-74, 140]
    if patch_size is None:
        patch_size = [96, 96, 96]
    if augmentation_config is None:
        augmentation_config = {
            'flip_prob': 0.5,
            'rotate_prob': 0.3,
            'rotate_range': 15,
            'scale_prob': 0.2,
            'scale_range': [0.9, 1.1]
        }
    
    transforms = [
        # Load image
        LoadImage(image_only=True),
        EnsureChannelFirst(),  # Add channel dim if missing
        
        # Reorient to RAS
        Reorient(axcodes="RAS"),
        
        # Intensity normalization (CRITICAL: must match training config)
        ScaleIntensityRanged(
            a_min=hu_window[0],
            a_max=hu_window[1],
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        
        # Spatial augmentation (training only)
        RandFlip(
            spatial_axis=0,
            prob=augmentation_config['flip_prob']
        ),
        RandFlip(
            spatial_axis=1,
            prob=augmentation_config['flip_prob']
        ),
        RandFlip(
            spatial_axis=2,
            prob=augmentation_config['flip_prob']
        ),
        
        # Random rotation
        RandRotate90(
            prob=augmentation_config['rotate_prob'],
            spatial_axes=(0, 1)
        ),
        RandRotate90(
            prob=augmentation_config['rotate_prob'],
            spatial_axes=(1, 2)
        ),
        
        # Random scale intensity (mimics scanner variations)
        RandScaleIntensity(
            factors=augmentation_config['scale_range'],
            prob=augmentation_config['scale_prob']
        ),
        
        # Pad to patch size if necessary
        EnsureTupleSize(spatial_size=patch_size, mode="constant"),
        
        # Convert to tensor
        ToTensor()
    ]
    
    logger.info(f"✓ Training transforms initialized (HU: {hu_window}, Patch: {patch_size})")
    return Compose(transforms)


def get_val_transforms(hu_window=None, patch_size=None):
    """
    Get validation/test transforms (NO augmentation).
    
    Args:
        hu_window: HU clipping range, e.g., [-74, 140]
        patch_size: Target patch size, e.g., [96, 96, 96]
    
    Returns:
        Compose object with validation transforms
    """
    
    if hu_window is None:
        hu_window = [-74, 140]
    if patch_size is None:
        patch_size = [96, 96, 96]
    
    transforms = [
        # Load image
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        
        # Reorient
        Reorient(axcodes="RAS"),
        
        # Intensity normalization (SAME as training)
        ScaleIntensityRanged(
            a_min=hu_window[0],
            a_max=hu_window[1],
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        
        # Pad to patch size
        EnsureTupleSize(spatial_size=patch_size, mode="constant"),
        
        # Convert to tensor
        ToTensor()
    ]
    
    logger.info(f"✓ Validation transforms initialized (HU: {hu_window}, Patch: {patch_size})")
    return Compose(transforms)


def get_inference_transforms(hu_window=None):
    """
    Get inference transforms (minimal, no resizing to allow variable input sizes).
    Used for sliding window inference on full volumes.
    
    Args:
        hu_window: HU clipping range
    
    Returns:
        Compose object with inference transforms
    """
    
    if hu_window is None:
        hu_window = [-74, 140]
    
    transforms = [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Reorient(axcodes="RAS"),
        ScaleIntensityRanged(
            a_min=hu_window[0],
            a_max=hu_window[1],
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        ToTensor()
    ]
    
    logger.info(f"✓ Inference transforms initialized (HU: {hu_window})")
    return Compose(transforms)


def get_postprocessing_transforms():
    """
    Get post-processing transforms for predictions.
    Converts probabilities to class labels.
    """
    # Used in evaluation script to convert soft predictions to hard labels
    return None  # Implemented in eval script


# Backward compatibility / export
training_transforms = get_train_transforms
validation_transforms = get_val_transforms
inference_transforms = get_inference_transforms


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*60)
    print("Testing Data Transforms")
    print("="*60)
    
    # Get transforms
    train_tf = get_train_transforms()
    val_tf = get_val_transforms()
    inf_tf = get_inference_transforms()
    
    print("\n✓ All transforms created successfully")
