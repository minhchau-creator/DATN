#!/usr/bin/env python3
"""
Test DiNTS model on PDAC dataset
Runs inference on all CT scans and saves tumor segmentation results with Dice/IoU metrics
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List
import numpy as np
import torch
import nibabel as nib
from tqdm import tqdm

from monai.transforms import Compose, LoadImageD, EnsureChannelFirstD, Orientationd, ScaleIntensityRanged
from monai.networks.nets import DiNTS, TopologyInstance
from monai.inferers import SlidingWindowInferer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def dice_coefficient(pred, target, class_idx):
    """
    Calculate Dice coefficient for a specific class
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    """
    pred_class = (pred == class_idx).astype(np.float32)
    target_class = (target == class_idx).astype(np.float32)
    
    intersection = np.sum(pred_class * target_class)
    union = np.sum(pred_class) + np.sum(target_class)
    
    if union == 0:
        return 1.0 if np.array_equal(pred_class, target_class) else 0.0
    
    return 2.0 * intersection / union


def iou_coefficient(pred, target, class_idx):
    """
    Calculate Intersection over Union (IoU) for a specific class
    IoU = |X ∩ Y| / (|X ∪ Y|)
    """
    pred_class = (pred == class_idx).astype(np.float32)
    target_class = (target == class_idx).astype(np.float32)
    
    intersection = np.sum(pred_class * target_class)
    union = np.sum(np.maximum(pred_class, target_class))
    
    if union == 0:
        return 1.0 if np.array_equal(pred_class, target_class) else 0.0
    
    return intersection / union


def calculate_metrics(pred, target):
    """
    Calculate Dice and IoU for all classes
    Returns dict with per-class and average metrics (all as Python floats)
    """
    metrics = {}
    
    # Class 0: Background (usually not reported)
    # Class 1: Pancreas
    # Class 2: Tumor
    
    for class_idx in [1, 2]:
        dice = float(dice_coefficient(pred, target, class_idx))
        iou = float(iou_coefficient(pred, target, class_idx))
        
        class_name = "Pancreas" if class_idx == 1 else "Tumor"
        metrics[f"{class_name}_Dice"] = dice
        metrics[f"{class_name}_IoU"] = iou
    
    # Calculate average (exclude background)
    avg_dice = float(np.mean([metrics["Pancreas_Dice"], metrics["Tumor_Dice"]]))
    avg_iou = float(np.mean([metrics["Pancreas_IoU"], metrics["Tumor_IoU"]]))
    
    metrics["Avg_Dice"] = avg_dice
    metrics["Avg_IoU"] = avg_iou
    
    return metrics


class DiNTSInferencer:
    """Run DiNTS model inference on CT scans"""
    
    def __init__(self, bundle_dir: str, device: str = "cuda"):
        """
        Initialize the DiNTS model
        
        Args:
            bundle_dir: Path to downloaded DiNTS bundle
            device: 'cuda' or 'cpu'
        """
        self.bundle_dir = Path(bundle_dir)
        self.device = torch.device(device)
        
        # Load architecture code from search results
        logger.info(f"Loading DiNTS model from {bundle_dir}")
        models_dir = self.bundle_dir / "models"
        
        # Load architecture code (allows pickle for old models)
        arch_ckpt_path = models_dir / "search_code_18590.pt"
        logger.info(f"Loading architecture code from {arch_ckpt_path}")
        arch_ckpt = torch.load(arch_ckpt_path, map_location="cpu", weights_only=False)
        
        # Build DiNTS space
        dints_space = TopologyInstance(
            channel_mul=1,
            num_blocks=12,
            num_depths=4,
            use_downsample=True,
            arch_code=[
                arch_ckpt["arch_code_a"],
                arch_ckpt["arch_code_c"]
            ],
            device=self.device,
        )
        
        # Build network
        self.network = DiNTS(
            dints_space=dints_space,
            in_channels=1,
            num_classes=3,
            use_downsample=True,
            node_a=torch.from_numpy(arch_ckpt["node_a"]),
        )
        
        # Load pretrained weights
        model_path = models_dir / "model.pt"
        logger.info(f"Loading weights from {model_path}")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        self.network.load_state_dict(state_dict)
        self.network.to(self.device)
        self.network.eval()
        
        # Create preprocessing transforms - MUST match training config!
        self.transforms = Compose([
            LoadImageD(keys=["image"]),
            EnsureChannelFirstD(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            # ✅ CRITICAL: Clip to Hounsfield Units window for pancreas
            # Optimized based on data analysis: [-74, 140] HU → [0, 1]
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-74,      # Min HU value (optimized)
                a_max=140,      # Max HU value (optimized for pancreas region)
                b_min=0.0,      # Output min
                b_max=1.0,      # Output max
                clip=True       # Clip values outside range
            ),
        ])
        
        # Create sliding window inferer for variable-size inputs
        self.inferer = SlidingWindowInferer(
            roi_size=(96, 96, 96),
            sw_batch_size=1,
            overlap=0.25,
            mode="gaussian",
            cache_roi_weight_map=True,
        )
        
        logger.info("✅ Model loaded successfully")
    
    def infer_single(self, image_path: str):
        """
        Run inference on a single CT scan using sliding window
        
        Args:
            image_path: Path to CT scan (.nii.gz)
            
        Returns:
            Tuple of (segmentation_mask, affine_matrix)
        """
        # Load original image to get affine matrix
        orig_img = nib.load(image_path)
        orig_affine = orig_img.affine
        
        # Load and preprocess
        data = {"image": image_path}
        data = self.transforms(data)
        
        image = data["image"]
        
        # Convert to tensor if needed
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        
        # Ensure 5D (B, C, D, H, W)
        if image.ndim == 4:
            image = image.unsqueeze(0)
        
        image = image.float().to(self.device)
        
        # Use sliding window inference for variable-size inputs
        with torch.no_grad():
            logits = self.inferer(image, self.network)  # (B, 3, D, H, W)
            pred = torch.argmax(logits, dim=1)  # (B, D, H, W)
        
        pred_mask = pred[0].cpu().numpy().astype(np.uint8)
        return pred_mask, orig_affine
    
    def infer_batch(self, image_paths: List[str], output_dir: str, labels_dir: str = None) -> dict:
        """
        Run inference on multiple CT scans
        
        Args:
            image_paths: List of CT scan paths
            output_dir: Directory to save segmentation results
            labels_dir: Optional directory containing ground truth labels
            
        Returns:
            Dictionary with results and metrics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        labels_dir = Path(labels_dir) if labels_dir else None
        
        results = {
            "total": len(image_paths),
            "successful": 0,
            "failed": 0,
            "has_gt": labels_dir is not None,
            "outputs": [],
            "metrics": {
                "per_scan": [],
                "aggregated": {}
            }
        }
        
        logger.info(f"Running inference on {len(image_paths)} scans...")
        
        for image_path in tqdm(image_paths, desc="Inference"):
            try:
                image_path = Path(image_path)
                
                # Run inference
                pred_mask, orig_affine = self.infer_single(str(image_path))
                
                # Save 1️⃣ RAW prediction (0/1/2) - for Slicer3D labels
                output_file_raw = output_dir / f"{image_path.stem}_pred_raw.nii.gz"
                pred_nii_raw = nib.Nifti1Image(pred_mask, affine=orig_affine)
                nib.save(pred_nii_raw, output_file_raw)
                
                # Save 2️⃣ SCALED prediction (0/127/254) - for visualization
                pred_mask_scaled = pred_mask * 127
                output_file = output_dir / f"{image_path.stem}_pred.nii.gz"
                pred_nii = nib.Nifti1Image(pred_mask_scaled, affine=orig_affine)
                nib.save(pred_nii, output_file)
                
                # Extract and save tumor mask (class 2 only) - RAW VERSION (0/1)
                tumor_mask = (pred_mask == 2).astype(np.uint8)
                tumor_file_raw = output_dir / f"{image_path.stem}_tumor_raw.nii.gz"
                tumor_nii_raw = nib.Nifti1Image(tumor_mask, affine=orig_affine)
                nib.save(tumor_nii_raw, tumor_file_raw)
                
                # Extract and save tumor mask - SCALED VERSION (0/255)
                tumor_mask_scaled = tumor_mask * 255
                tumor_file = output_dir / f"{image_path.stem}_tumor.nii.gz"
                tumor_nii = nib.Nifti1Image(tumor_mask_scaled, affine=orig_affine)
                nib.save(tumor_nii, tumor_file)
                
                # Calculate metrics if ground truth is available
                # Get base name without extensions (pancreas_001.nii.gz -> pancreas_001)
                scan_name = image_path.name.replace(".nii.gz", "")
                scan_metrics = {"scan": scan_name}
                if labels_dir and labels_dir.exists():
                    # Find corresponding label file
                    label_file = labels_dir / f"{scan_name}.nii.gz"
                    if label_file.exists():
                        gt_img = nib.load(label_file)
                        gt_mask = gt_img.get_fdata().astype(np.uint8)
                        
                        # Calculate metrics
                        metrics = calculate_metrics(pred_mask, gt_mask)
                        scan_metrics.update(metrics)
                        results["metrics"]["per_scan"].append(scan_metrics)
                        
                        logger.info(f"  Pancreas Dice: {metrics['Pancreas_Dice']:.4f} | IoU: {metrics['Pancreas_IoU']:.4f}")
                        logger.info(f"  Tumor    Dice: {metrics['Tumor_Dice']:.4f} | IoU: {metrics['Tumor_IoU']:.4f}")
                    else:
                        logger.warning(f"  Ground truth not found: {label_file}")
                        results["metrics"]["per_scan"].append(scan_metrics)
                else:
                    results["metrics"]["per_scan"].append(scan_metrics)
                
                results["successful"] += 1
                results["outputs"].append({
                    "input": str(image_path),
                    "prediction_raw": str(output_file_raw),
                    "prediction_scaled": str(output_file),
                    "tumor_raw": str(tumor_file_raw),
                    "tumor_scaled": str(tumor_file),
                    "metrics": scan_metrics
                })
                
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {str(e)}")
                results["failed"] += 1
        
        # Calculate aggregated metrics
        if results["metrics"]["per_scan"]:
            per_scan = results["metrics"]["per_scan"]
            
            for metric_key in ["Pancreas_Dice", "Pancreas_IoU", "Tumor_Dice", "Tumor_IoU", "Avg_Dice", "Avg_IoU"]:
                values = [m[metric_key] for m in per_scan if metric_key in m]
                if values:
                    results["metrics"]["aggregated"][f"Mean_{metric_key}"] = float(np.mean(values))
                    results["metrics"]["aggregated"][f"Std_{metric_key}"] = float(np.std(values))
                    results["metrics"]["aggregated"][f"Min_{metric_key}"] = float(np.min(values))
                    results["metrics"]["aggregated"][f"Max_{metric_key}"] = float(np.max(values))
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Test DiNTS model on PDAC dataset"
    )
    parser.add_argument(
        "--bundle-dir",
        type=str,
        default="/home/minhchau/anaconda3/envs/datn/DATN/DiNTS/pancreas_model/pancreas_ct_dints_segmentation",
        help="Path to downloaded DiNTS bundle"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/minhchau/anaconda3/envs/datn/DATN/dataset/Task_7/imagesTr",
        help="Path to input CT scans folder"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/minhchau/anaconda3/envs/datn/DATN/code/segmentation_results",
        help="Path to save segmentation results"
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default="/home/minhchau/anaconda3/envs/datn/DATN/dataset/Task_7/labelsTr",
        help="Path to ground truth labels folder (optional)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--num-scans",
        type=int,
        default=None,
        help="Number of scans to process (None = all)"
    )
    
    args = parser.parse_args()
    
    # Verify paths
    bundle_dir = Path(args.bundle_dir)
    if not bundle_dir.exists():
        logger.error(f"Bundle directory not found: {bundle_dir}")
        return
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    # Get list of CT scans
    image_files = sorted(data_dir.glob("*.nii.gz"))[:args.num_scans]
    logger.info(f"Found {len(image_files)} CT scans to process")
    
    if not image_files:
        logger.error(f"No .nii.gz files found in {data_dir}")
        return
    
    # Initialize model
    if args.device == "cuda":
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"✅ GPU Available - Using CUDA")
            logger.info(f"GPU Count: {torch.cuda.device_count()}")
            logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.warning(f"⚠️  CUDA requested but not available - Falling back to CPU")
    else:
        device = "cpu"
    
    logger.info(f"Device: {device}")
    inferencer = DiNTSInferencer(str(bundle_dir), device=device)
    
    # Run inference
    results = inferencer.infer_batch(
        [str(f) for f in image_files],
        args.output_dir,
        labels_dir=args.labels_dir
    )
    
    # Print summary
    logger.info("=" * 80)
    logger.info("INFERENCE SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total scans: {results['total']}")
    logger.info(f"Successful: {results['successful']} ✅")
    logger.info(f"Failed: {results['failed']} ❌")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Print metrics summary if available
    if results["metrics"]["aggregated"]:
        logger.info("-" * 80)
        logger.info("SEGMENTATION METRICS (AGGREGATED)")
        logger.info("-" * 80)
        
        metrics_agg = results["metrics"]["aggregated"]
        
        # Pancreas metrics
        logger.info("Pancreas:")
        if "Mean_Pancreas_Dice" in metrics_agg:
            logger.info(f"  Dice: {metrics_agg['Mean_Pancreas_Dice']:.4f} ± {metrics_agg['Std_Pancreas_Dice']:.4f} "
                       f"(min: {metrics_agg['Min_Pancreas_Dice']:.4f}, max: {metrics_agg['Max_Pancreas_Dice']:.4f})")
        if "Mean_Pancreas_IoU" in metrics_agg:
            logger.info(f"  IoU:  {metrics_agg['Mean_Pancreas_IoU']:.4f} ± {metrics_agg['Std_Pancreas_IoU']:.4f} "
                       f"(min: {metrics_agg['Min_Pancreas_IoU']:.4f}, max: {metrics_agg['Max_Pancreas_IoU']:.4f})")
        
        # Tumor metrics
        logger.info("Tumor:")
        if "Mean_Tumor_Dice" in metrics_agg:
            logger.info(f"  Dice: {metrics_agg['Mean_Tumor_Dice']:.4f} ± {metrics_agg['Std_Tumor_Dice']:.4f} "
                       f"(min: {metrics_agg['Min_Tumor_Dice']:.4f}, max: {metrics_agg['Max_Tumor_Dice']:.4f})")
        if "Mean_Tumor_IoU" in metrics_agg:
            logger.info(f"  IoU:  {metrics_agg['Mean_Tumor_IoU']:.4f} ± {metrics_agg['Std_Tumor_IoU']:.4f} "
                       f"(min: {metrics_agg['Min_Tumor_IoU']:.4f}, max: {metrics_agg['Max_Tumor_IoU']:.4f})")
        
        # Average metrics
        logger.info("Average (Pancreas + Tumor):")
        if "Mean_Avg_Dice" in metrics_agg:
            logger.info(f"  Dice: {metrics_agg['Mean_Avg_Dice']:.4f} ± {metrics_agg['Std_Avg_Dice']:.4f} "
                       f"(min: {metrics_agg['Min_Avg_Dice']:.4f}, max: {metrics_agg['Max_Avg_Dice']:.4f})")
        if "Mean_Avg_IoU" in metrics_agg:
            logger.info(f"  IoU:  {metrics_agg['Mean_Avg_IoU']:.4f} ± {metrics_agg['Std_Avg_IoU']:.4f} "
                       f"(min: {metrics_agg['Min_Avg_IoU']:.4f}, max: {metrics_agg['Max_Avg_IoU']:.4f})")
    
    logger.info("=" * 80)
    
    # Save results to JSON
    results_file = Path(args.output_dir) / "inference_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Detailed results saved to {results_file}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Test DiNTS model on PDAC dataset
Runs inference on all CT scans and saves tumor segmentation results
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List
import numpy as np
import torch
import nibabel as nib
from tqdm import tqdm

from monai.transforms import Compose, LoadImageD, EnsureChannelFirstD, Orientationd, ScaleIntensityRanged
from monai.networks.nets import DiNTS, TopologyInstance
from monai.inferers import SlidingWindowInferer

# Metric calculation functions
def dice_coefficient(pred, target, class_idx):
    """
    Calculate Dice coefficient for a specific class
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    """
    pred_class = (pred == class_idx).astype(np.float32)
    target_class = (target == class_idx).astype(np.float32)
    
    intersection = np.sum(pred_class * target_class)
    union = np.sum(pred_class) + np.sum(target_class)
    
    if union == 0:
        return 1.0 if np.array_equal(pred_class, target_class) else 0.0
    
    return 2.0 * intersection / union


def iou_coefficient(pred, target, class_idx):
    """
    Calculate Intersection over Union (IoU) for a specific class
    IoU = |X ∩ Y| / (|X ∪ Y|)
    """
    pred_class = (pred == class_idx).astype(np.float32)
    target_class = (target == class_idx).astype(np.float32)
    
    intersection = np.sum(pred_class * target_class)
    union = np.sum(np.maximum(pred_class, target_class))
    
    if union == 0:
        return 1.0 if np.array_equal(pred_class, target_class) else 0.0
    
    return intersection / union


def calculate_metrics(pred, target):
    """
    Calculate Dice and IoU for all classes
    Returns dict with per-class and average metrics (all as Python floats)
    """
    metrics = {}
    
    # Class 0: Background (usually not reported)
    # Class 1: Pancreas
    # Class 2: Tumor
    
    for class_idx in [1, 2]:
        dice = float(dice_coefficient(pred, target, class_idx))
        iou = float(iou_coefficient(pred, target, class_idx))
        
        class_name = "Pancreas" if class_idx == 1 else "Tumor"
        metrics[f"{class_name}_Dice"] = dice
        metrics[f"{class_name}_IoU"] = iou
    
    # Calculate average (exclude background)
    avg_dice = float(np.mean([metrics["Pancreas_Dice"], metrics["Tumor_Dice"]]))
    avg_iou = float(np.mean([metrics["Pancreas_IoU"], metrics["Tumor_IoU"]]))
    
    metrics["Avg_Dice"] = avg_dice
    metrics["Avg_IoU"] = avg_iou
    
    return metrics


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DiNTSInferencer:
    """Run DiNTS model inference on CT scans"""
    
    def __init__(self, bundle_dir: str, device: str = "cuda"):
        """
        Initialize the DiNTS model
        
        Args:
            bundle_dir: Path to downloaded DiNTS bundle
            device: 'cuda' or 'cpu'
        """
        self.bundle_dir = Path(bundle_dir)
        self.device = torch.device(device)
        
        # Load architecture code from search results
        logger.info(f"Loading DiNTS model from {bundle_dir}")
        models_dir = self.bundle_dir / "models"
        
        # Load architecture code (allows pickle for old models)
        arch_ckpt_path = models_dir / "search_code_18590.pt"
        logger.info(f"Loading architecture code from {arch_ckpt_path}")
        arch_ckpt = torch.load(arch_ckpt_path, map_location="cpu", weights_only=False)
        
        # Build DiNTS space
        dints_space = TopologyInstance(
            channel_mul=1,
            num_blocks=12,
            num_depths=4,
            use_downsample=True,
            arch_code=[
                arch_ckpt["arch_code_a"],
                arch_ckpt["arch_code_c"]
            ],
            device=self.device,
        )
        
        # Build network
        self.network = DiNTS(
            dints_space=dints_space,
            in_channels=1,
            num_classes=3,
            use_downsample=True,
            node_a=torch.from_numpy(arch_ckpt["node_a"]),
        )
        
        # Load pretrained weights
        model_path = models_dir / "model.pt"
        logger.info(f"Loading weights from {model_path}")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        self.network.load_state_dict(state_dict)
        self.network.to(self.device)
        self.network.eval()
        
        # Create preprocessing transforms - MUST match training config!
        self.transforms = Compose([
            LoadImageD(keys=["image"]),
            EnsureChannelFirstD(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            # ✅ CRITICAL: Clip to Hounsfield Units window for pancreas
            # Optimized based on data analysis: [-74, 140] HU → [0, 1]
            # (Original config [-87, 199] was less optimal)
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-74,      # Min HU value (optimized)
                a_max=140,      # Max HU value (optimized for pancreas region)
                b_min=0.0,      # Output min
                b_max=1.0,      # Output max
                clip=True       # Clip values outside range
            ),
        ])
        
        # Create sliding window inferer for variable-size inputs
        self.inferer = SlidingWindowInferer(
            roi_size=(96, 96, 96),
            sw_batch_size=1,
            overlap=0.25,
            mode="gaussian",
            cache_roi_weight_map=True,
        )
        
        logger.info("✅ Model loaded successfully")
    
    def infer_single(self, image_path: str):
        """
        Run inference on a single CT scan using sliding window
        
        Args:
            image_path: Path to CT scan (.nii.gz)
            
        Returns:
            Tuple of (segmentation_mask, affine_matrix)
        """
        # Load original image to get affine matrix
        orig_img = nib.load(image_path)
        orig_affine = orig_img.affine
        
        # Load and preprocess
        data = {"image": image_path}
        data = self.transforms(data)
        
        image = data["image"]
        
        # Convert to tensor if needed
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        
        # Ensure 5D (B, C, D, H, W)
        if image.ndim == 4:
            image = image.unsqueeze(0)
        
        image = image.float().to(self.device)
        
        # Use sliding window inference for variable-size inputs
        with torch.no_grad():
            logits = self.inferer(image, self.network)  # (B, 3, D, H, W)
            pred = torch.argmax(logits, dim=1)  # (B, D, H, W)
        
        pred_mask = pred[0].cpu().numpy().astype(np.uint8)
        return pred_mask, orig_affine
    
    def infer_batch(self, image_paths: List[str], output_dir: str, labels_dir: str = None) -> dict:
        """
        Run inference on multiple CT scans
        
        Args:
            image_paths: List of CT scan paths
            output_dir: Directory to save segmentation results
            labels_dir: Optional directory containing ground truth labels
            
        Returns:
            Dictionary with results and metrics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        labels_dir = Path(labels_dir) if labels_dir else None
        
        results = {
            "total": len(image_paths),
            "successful": 0,
            "failed": 0,
            "has_gt": labels_dir is not None,
            "outputs": [],
            "metrics": {
                "per_scan": [],
                "aggregated": {}
            }
        }
        
        logger.info(f"Running inference on {len(image_paths)} scans...")
        
        for image_path in tqdm(image_paths, desc="Inference"):
            try:
                image_path = Path(image_path)
                
                # Run inference
                pred_mask, orig_affine = self.infer_single(str(image_path))
                
                # Save 1️⃣ RAW prediction (0/1/2) - for Slicer3D labels
                output_file_raw = output_dir / f"{image_path.stem}_pred_raw.nii.gz"
                pred_nii_raw = nib.Nifti1Image(pred_mask, affine=orig_affine)
                nib.save(pred_nii_raw, output_file_raw)
                
                # Save 2️⃣ SCALED prediction (0/127/254) - for visualization
                pred_mask_scaled = pred_mask * 127
                output_file = output_dir / f"{image_path.stem}_pred.nii.gz"
                pred_nii = nib.Nifti1Image(pred_mask_scaled, affine=orig_affine)
                nib.save(pred_nii, output_file)
                
                # Extract and save tumor mask (class 2 only) - RAW VERSION (0/1)
                tumor_mask = (pred_mask == 2).astype(np.uint8)
                tumor_file_raw = output_dir / f"{image_path.stem}_tumor_raw.nii.gz"
                tumor_nii_raw = nib.Nifti1Image(tumor_mask, affine=orig_affine)
                nib.save(tumor_nii_raw, tumor_file_raw)
                
                # Extract and save tumor mask - SCALED VERSION (0/255)
                tumor_mask_scaled = tumor_mask * 255
                tumor_file = output_dir / f"{image_path.stem}_tumor.nii.gz"
                tumor_nii = nib.Nifti1Image(tumor_mask_scaled, affine=orig_affine)
                nib.save(tumor_nii, tumor_file)
                
                # Calculate metrics if ground truth is available
                # Get base name without extensions (pancreas_001.nii.gz -> pancreas_001)
                scan_name = image_path.name.replace(".nii.gz", "")
                scan_metrics = {"scan": scan_name}
                if labels_dir and labels_dir.exists():
                    # Find corresponding label file
                    label_file = labels_dir / f"{scan_name}.nii.gz"
                    if label_file.exists():
                        gt_img = nib.load(label_file)
                        gt_mask = gt_img.get_fdata().astype(np.uint8)
                        
                        # Calculate metrics
                        metrics = calculate_metrics(pred_mask, gt_mask)
                        scan_metrics.update(metrics)
                        results["metrics"]["per_scan"].append(scan_metrics)
                        
                        logger.info(f"  Pancreas Dice: {metrics['Pancreas_Dice']:.4f} | IoU: {metrics['Pancreas_IoU']:.4f}")
                        logger.info(f"  Tumor    Dice: {metrics['Tumor_Dice']:.4f} | IoU: {metrics['Tumor_IoU']:.4f}")
                    else:
                        logger.warning(f"  Ground truth not found: {label_file}")
                        results["metrics"]["per_scan"].append(scan_metrics)
                else:
                    results["metrics"]["per_scan"].append(scan_metrics)
                
                results["successful"] += 1
                results["outputs"].append({
                    "input": str(image_path),
                    "prediction_raw": str(output_file_raw),
                    "prediction_scaled": str(output_file),
                    "tumor_raw": str(tumor_file_raw),
                    "tumor_scaled": str(tumor_file),
                    "metrics": scan_metrics
                })
                
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {str(e)}")
                results["failed"] += 1
        
        # Calculate aggregated metrics
        if results["metrics"]["per_scan"]:
            per_scan = results["metrics"]["per_scan"]
            
            for metric_key in ["Pancreas_Dice", "Pancreas_IoU", "Tumor_Dice", "Tumor_IoU", "Avg_Dice", "Avg_IoU"]:
                values = [m[metric_key] for m in per_scan if metric_key in m]
                if values:
                    results["metrics"]["aggregated"][f"Mean_{metric_key}"] = float(np.mean(values))
                    results["metrics"]["aggregated"][f"Std_{metric_key}"] = float(np.std(values))
                    results["metrics"]["aggregated"][f"Min_{metric_key}"] = float(np.min(values))
                    results["metrics"]["aggregated"][f"Max_{metric_key}"] = float(np.max(values))
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Test DiNTS model on PDAC dataset"
    )
    parser.add_argument(
        "--bundle-dir",
        type=str,
        default="/home/minhchau/anaconda3/envs/datn/DATN/DiNTS/pancreas_model/pancreas_ct_dints_segmentation",
        help="Path to downloaded DiNTS bundle"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/minhchau/anaconda3/envs/datn/DATN/dataset/Task_7/imagesTr",
        help="Path to input CT scans folder"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/minhchau/anaconda3/envs/datn/DATN/code/segmentation_results",
        help="Path to save segmentation results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--num-scans",
        type=int,
        default=None,
        help="Number of scans to process (None = all)"
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default="/home/minhchau/anaconda3/envs/datn/DATN/dataset/Task_7/labelsTr",
        help="Path to ground truth labels folder (optional)"
    )
    
    args = parser.parse_args()
    
    # Verify paths
    bundle_dir = Path(args.bundle_dir)
    if not bundle_dir.exists():
        logger.error(f"Bundle directory not found: {bundle_dir}")
        return
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    # Get list of CT scans
    image_files = sorted(data_dir.glob("*.nii.gz"))[:args.num_scans]
    logger.info(f"Found {len(image_files)} CT scans to process")
    
    if not image_files:
        logger.error(f"No .nii.gz files found in {data_dir}")
        return
    
    # Initialize model
    if args.device == "cuda":
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"✅ GPU Available - Using CUDA")
            logger.info(f"GPU Count: {torch.cuda.device_count()}")
            logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.warning(f"⚠️  CUDA requested but not available - Falling back to CPU")
    else:
        device = "cpu"
    
    logger.info(f"Device: {device}")
    inferencer = DiNTSInferencer(str(bundle_dir), device=device)
    
    # Run inference
    results = inferencer.infer_batch(
        [str(f) for f in image_files],
        args.output_dir,
        labels_dir=args.labels_dir
    )
    
    # Print summary
    logger.info("=" * 80)
    logger.info("INFERENCE SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total scans: {results['total']}")
    logger.info(f"Successful: {results['successful']} ✅")
    logger.info(f"Failed: {results['failed']} ❌")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Print metrics summary if available
    if results["metrics"]["aggregated"]:
        logger.info("-" * 80)
        logger.info("SEGMENTATION METRICS (AGGREGATED)")
        logger.info("-" * 80)
        
        metrics_agg = results["metrics"]["aggregated"]
        
        # Pancreas metrics
        logger.info("Pancreas:")
        if "Mean_Pancreas_Dice" in metrics_agg:
            logger.info(f"  Dice: {metrics_agg['Mean_Pancreas_Dice']:.4f} ± {metrics_agg['Std_Pancreas_Dice']:.4f} "
                       f"(min: {metrics_agg['Min_Pancreas_Dice']:.4f}, max: {metrics_agg['Max_Pancreas_Dice']:.4f})")
        if "Mean_Pancreas_IoU" in metrics_agg:
            logger.info(f"  IoU:  {metrics_agg['Mean_Pancreas_IoU']:.4f} ± {metrics_agg['Std_Pancreas_IoU']:.4f} "
                       f"(min: {metrics_agg['Min_Pancreas_IoU']:.4f}, max: {metrics_agg['Max_Pancreas_IoU']:.4f})")
        
        # Tumor metrics
        logger.info("Tumor:")
        if "Mean_Tumor_Dice" in metrics_agg:
            logger.info(f"  Dice: {metrics_agg['Mean_Tumor_Dice']:.4f} ± {metrics_agg['Std_Tumor_Dice']:.4f} "
                       f"(min: {metrics_agg['Min_Tumor_Dice']:.4f}, max: {metrics_agg['Max_Tumor_Dice']:.4f})")
        if "Mean_Tumor_IoU" in metrics_agg:
            logger.info(f"  IoU:  {metrics_agg['Mean_Tumor_IoU']:.4f} ± {metrics_agg['Std_Tumor_IoU']:.4f} "
                       f"(min: {metrics_agg['Min_Tumor_IoU']:.4f}, max: {metrics_agg['Max_Tumor_IoU']:.4f})")
        
        # Average metrics
        logger.info("Average (Pancreas + Tumor):")
        if "Mean_Avg_Dice" in metrics_agg:
            logger.info(f"  Dice: {metrics_agg['Mean_Avg_Dice']:.4f} ± {metrics_agg['Std_Avg_Dice']:.4f} "
                       f"(min: {metrics_agg['Min_Avg_Dice']:.4f}, max: {metrics_agg['Max_Avg_Dice']:.4f})")
        if "Mean_Avg_IoU" in metrics_agg:
            logger.info(f"  IoU:  {metrics_agg['Mean_Avg_IoU']:.4f} ± {metrics_agg['Std_Avg_IoU']:.4f} "
                       f"(min: {metrics_agg['Min_Avg_IoU']:.4f}, max: {metrics_agg['Max_Avg_IoU']:.4f})")
    
    logger.info("=" * 80)
    
    # Save results to JSON
    results_file = Path(args.output_dir) / "inference_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Detailed results saved to {results_file}")


if __name__ == "__main__":
    main()
