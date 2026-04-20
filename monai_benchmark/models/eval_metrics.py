#!/usr/bin/env python3
"""
Evaluation Script - Compute Metrics (Dice, IoU, Hausdorff) for predictions
"""

import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation
import torch
from monai.metrics import compute_hausdorff_distance, compute_dice
import logging

logger = logging.getLogger(__name__)


def dice_coefficient(pred, target, class_idx=None, smooth=1e-6):
    """
    Compute Dice coefficient for single class or average across classes.
    
    Args:
        pred: (D, H, W) predicted class labels or (C, D, H, W) one-hot
        target: (D, H, W) ground truth class labels or (C, D, H, W) one-hot
        class_idx: If specified, compute Dice for this class only
        smooth: Smoothing factor
    
    Returns:
        float: Dice coefficient [0, 1]
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    # Convert to binary if needed (per-class Dice)
    if class_idx is not None:
        if pred.ndim == 4:  # One-hot format (C, D, H, W)
            pred_binary = (pred[class_idx] > 0.5).astype(np.float32)
        else:  # Label format (D, H, W)
            pred_binary = (pred == class_idx).astype(np.float32)
        
        if target.ndim == 4:
            target_binary = target[class_idx].astype(np.float32)
        else:
            target_binary = (target == class_idx).astype(np.float32)
    else:
        pred_binary = (pred > 0.5).astype(np.float32) if pred.ndim == 4 else pred.astype(np.float32)
        target_binary = target.astype(np.float32)
    
    # Compute Dice
    intersection = np.sum(pred_binary * target_binary)
    union = np.sum(pred_binary) + np.sum(target_binary)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return float(dice)


def iou_coefficient(pred, target, class_idx=None, smooth=1e-6):
    """
    Compute Intersection over Union (Jaccard Index).
    
    Args:
        pred: Predicted segmentation
        target: Ground truth segmentation
        class_idx: Class to compute IoU for
        smooth: Smoothing factor
    
    Returns:
        float: IoU [0, 1]
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    # Convert to binary
    if class_idx is not None:
        if pred.ndim == 4:
            pred_binary = (pred[class_idx] > 0.5).astype(np.float32)
        else:
            pred_binary = (pred == class_idx).astype(np.float32)
        
        if target.ndim == 4:
            target_binary = target[class_idx].astype(np.float32)
        else:
            target_binary = (target == class_idx).astype(np.float32)
    else:
        pred_binary = (pred > 0.5).astype(np.float32) if pred.ndim == 4 else pred.astype(np.float32)
        target_binary = target.astype(np.float32)
    
    # Compute IoU
    intersection = np.sum(pred_binary * target_binary)
    union = np.sum(pred_binary) + np.sum(target_binary) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return float(iou)


def hausdorff_distance(pred, target, class_idx=None):
    """
    Compute Hausdorff Distance (95th percentile).
    
    Args:
        pred: Predicted segmentation
        target: Ground truth segmentation
        class_idx: Class to compute HD for
    
    Returns:
        float: Hausdorff distance in mm (or voxels if no spacing provided)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    # Convert to binary
    if class_idx is not None:
        if pred.ndim == 4:
            pred_binary = (pred[class_idx] > 0.5).astype(np.uint8)
        else:
            pred_binary = (pred == class_idx).astype(np.uint8)
        
        if target.ndim == 4:
            target_binary = target[class_idx].astype(np.uint8)
        else:
            target_binary = (target == class_idx).astype(np.uint8)
    else:
        pred_binary = (pred > 0.5).astype(np.uint8) if pred.ndim == 4 else pred.astype(np.uint8)
        target_binary = target.astype(np.uint8)
    
    # If either is empty, return large distance
    if np.sum(pred_binary) == 0 or np.sum(target_binary) == 0:
        return 1000.0
    
    # Compute Hausdorff (95th percentile)
    try:
        from scipy.spatial.distance import directed_hausdorff
        pred_pts = np.argwhere(pred_binary)
        target_pts = np.argwhere(target_binary)
        
        if len(pred_pts) == 0 or len(target_pts) == 0:
            return 1000.0
        
        d1 = directed_hausdorff(pred_pts, target_pts)[0]
        d2 = directed_hausdorff(target_pts, pred_pts)[0]
        hd = max(d1, d2)
        return float(hd)
    except Exception as e:
        logger.warning(f"Could not compute Hausdorff: {e}")
        return 0.0


def calculate_metrics(pred, target, spacing=None):
    """
    Calculate comprehensive metrics for 3-class segmentation (Background, Pancreas, Tumor).
    
    Args:
        pred: (D, H, W) predicted class labels [0, 1, 2]
        target: (D, H, W) ground truth class labels [0, 1, 2]
        spacing: Physical spacing in mm (default: None → use voxel distance)
    
    Returns:
        dict: Metrics including Dice, IoU, Hausdorff for each class
    """
    
    metrics = {
        'pancreas_dice': dice_coefficient(pred, target, class_idx=1),
        'pancreas_iou': iou_coefficient(pred, target, class_idx=1),
        'pancreas_hausdorff': hausdorff_distance(pred, target, class_idx=1),
        'tumor_dice': dice_coefficient(pred, target, class_idx=2),
        'tumor_iou': iou_coefficient(pred, target, class_idx=2),
        'tumor_hausdorff': hausdorff_distance(pred, target, class_idx=2),
        'avg_dice': (dice_coefficient(pred, target, class_idx=1) + dice_coefficient(pred, target, class_idx=2)) / 2.0,
        'avg_iou': (iou_coefficient(pred, target, class_idx=1) + iou_coefficient(pred, target, class_idx=2)) / 2.0,
    }
    
    return metrics


def print_metrics(metrics, scan_name=""):
    """Pretty print metrics."""
    print(f"\n  {scan_name}")
    print(f"    Pancreas Dice: {metrics['pancreas_dice']:.4f}, IoU: {metrics['pancreas_iou']:.4f}")
    print(f"    Tumor Dice:    {metrics['tumor_dice']:.4f}, IoU: {metrics['tumor_iou']:.4f}")
    print(f"    Avg Dice:      {metrics['avg_dice']:.4f}")


if __name__ == "__main__":
    # Test metrics
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy predictions and targets
    D, H, W = 64, 64, 64
    pred = np.random.randint(0, 3, (D, H, W))
    target = np.random.randint(0, 3, (D, H, W))
    
    metrics = calculate_metrics(pred, target)
    print("\n✅ Metrics computed:")
    print_metrics(metrics, "Test scan")
    print(f"\nAll metrics: {metrics}")
