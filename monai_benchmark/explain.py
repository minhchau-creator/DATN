#!/usr/bin/env python3
"""
XAI Explanation CLI — Pancreas / PDAC Tumor Segmentation

Runs the full explainability pipeline on a single NIfTI CT scan:
  1. Sliding-window inference → prediction mask
  2. GradCAM-3D              → what spatial features drove the tumor prediction
  3. Integrated Gradients    → which input voxels contributed most
  4. MC-Dropout Uncertainty  → how confident is the model per voxel
  5. Radiological features   → HU stats, location, morphology, PDAC reference comparison
  6. Visualisation           → PNG figures (axial/coronal/sagittal) + NIfTI heatmaps
  7. Text report             → saved to results/xai/<case>_report.txt

Usage:
    python monai_benchmark/explain.py \\
        --input  /mnt/d/DATN/dataset/Task_7/imagesTr/pancreas_001.nii.gz \\
        --label  /mnt/d/DATN/dataset/Task_7/labelsTr/pancreas_001.nii.gz \\
        --model  nnunet \\
        --fold   0 \\
        --device cuda:0 \\
        --output results/xai
"""

import sys
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, Orientation,
    ScaleIntensityRange, ToTensor,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from monai_benchmark.models.builder import get_model
from monai_benchmark.xai.gradcam import GradCAM3D, saliency_map, integrated_gradients
from monai_benchmark.xai.uncertainty import mc_dropout_uncertainty
from monai_benchmark.xai.radiological_features import extract_radiological_features
from monai_benchmark.xai.visualize import save_xai_figure, save_nifti_maps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing (single image, no label key)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(nifti_path: str, hu_min=-74.0, hu_max=140.0) -> torch.Tensor:
    """Return (1, 1, D, H, W) float tensor ready for the model."""
    tfm = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        ScaleIntensityRange(a_min=hu_min, a_max=hu_max,
                            b_min=0.0, b_max=1.0, clip=True),
        ToTensor(),
    ])
    t = tfm(nifti_path)
    if t.dim() == 3:
        t = t.unsqueeze(0)
    return t.unsqueeze(0).float()   # (1, 1, D, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_name: str, fold: int, device: torch.device) -> torch.nn.Module:
    ckpt_path = Path(f"checkpoints/{model_name}/fold_{fold}/best_model.pth")
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Train the model first:  python monai_benchmark/train_models.py "
            f"--model {model_name} --folds {fold}"
        )
    model = get_model(model_name, pretrained=False, device=device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    model.eval()
    logger.info(f"Loaded: {ckpt_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    model: torch.nn.Module,
    image: torch.Tensor,
    roi_size=(96, 96, 96),
    sw_batch=4,
    overlap=0.25,
) -> np.ndarray:
    """Return (D, H, W) int8 argmax prediction."""
    def _fwd(x):
        out = model(x)
        return out[0] if isinstance(out, (list, tuple)) else out

    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=image,
            roi_size=roi_size,
            sw_batch_size=sw_batch,
            predictor=_fwd,
            overlap=overlap,
        )
    return torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int8)


# ─────────────────────────────────────────────────────────────────────────────
# Report writer
# ─────────────────────────────────────────────────────────────────────────────

def write_report(
    output_dir: Path,
    case_name: str,
    args,
    pred_mask: np.ndarray,
    rad_features: dict,
    saved_pngs: dict,
    saved_niftis: dict,
):
    report_path = output_dir / f"{case_name}_report.txt"

    n_tumor = int((pred_mask == 2).sum())
    n_pan   = int((pred_mask == 1).sum())

    lines = [
        "=" * 60,
        "  PDAC XAI EXPLANATION REPORT",
        "=" * 60,
        f"  Input CT    : {args.input}",
        f"  Model       : {args.model}  (fold {args.fold})",
        f"  Device      : {args.device}",
        "",
        f"[SEGMENTATION SUMMARY]",
        f"  Pancreas voxels : {n_pan:,}",
        f"  Tumor voxels    : {n_tumor:,}",
        f"  Tumor found     : {'YES' if n_tumor > 0 else 'NO'}",
        "",
    ]

    # Radiological section
    lines.append(rad_features.get("explanation", "No radiological features computed."))
    lines.append("")

    # Output files
    lines += ["[OUTPUT FILES]"]
    for view, path in saved_pngs.items():
        lines.append(f"  PNG ({view:9s}) : {path}")
    for name, path in saved_niftis.items():
        lines.append(f"  NIfTI ({name:8s}) : {path}")
    lines += ["", "=" * 60]

    report_text = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report_text)

    print("\n" + report_text)
    logger.info(f"Report saved → {report_path}")
    return str(report_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device   = torch.device(args.device)
    out_dir  = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    case_name = Path(args.input).stem.replace(".nii", "")

    logger.info(f"\n{'='*60}\nExplaining: {case_name}\n{'='*60}")

    # ── 1. Load & preprocess ──────────────────────────────────────────────────
    logger.info("Step 1/6 — Preprocessing CT")
    image_tensor = preprocess(args.input).to(device)          # (1,1,D,H,W)

    # Load raw HU for radiological features
    raw_nib = nib.load(args.input)
    ct_hu   = raw_nib.get_fdata().astype(np.float32)          # (D, H, W)

    spacing = tuple(raw_nib.header.get_zooms()[:3])

    # ── 2. Load model + predict ───────────────────────────────────────────────
    logger.info("Step 2/6 — Loading model & running inference")
    model = load_model(args.model, args.fold, device)
    pred_mask = run_inference(model, image_tensor,
                              roi_size=args.roi_size,
                              sw_batch=args.sw_batch,
                              overlap=args.overlap)

    n_tumor = (pred_mask == 2).sum()
    if n_tumor == 0:
        logger.warning("No tumor voxels predicted. XAI maps will be empty.")

    # ── 3. GradCAM ────────────────────────────────────────────────────────────
    gradcam_map = None
    ig_map      = None

    if not args.skip_gradcam:
        logger.info("Step 3/6 — GradCAM-3D")
        try:
            # Work on a centre crop around tumour for efficiency
            cam_obj = GradCAM3D(model)
            gradcam_map = cam_obj.generate(image_tensor)
            cam_obj.remove_hooks()
            logger.info(f"  GradCAM range: [{gradcam_map.min():.3f}, {gradcam_map.max():.3f}]")
        except Exception as e:
            logger.warning(f"GradCAM failed: {e}")
    else:
        logger.info("Step 3/6 — GradCAM skipped (--skip-gradcam)")

    # ── 4. Integrated Gradients ───────────────────────────────────────────────
    if not args.skip_ig:
        logger.info("Step 4/6 — Integrated Gradients")
        try:
            ig_map = integrated_gradients(model, image_tensor,
                                          n_steps=args.ig_steps)
            logger.info(f"  IG range: [{ig_map.min():.3f}, {ig_map.max():.3f}]")
        except Exception as e:
            logger.warning(f"Integrated Gradients failed: {e}")
    else:
        logger.info("Step 4/6 — Integrated Gradients skipped (--skip-ig)")

    # ── 5. Uncertainty ────────────────────────────────────────────────────────
    uncertainty_map = None
    if not args.skip_uncertainty:
        logger.info(f"Step 5/6 — MC-Dropout uncertainty ({args.mc_passes} passes)")
        try:
            _, tumor_prob, uncertainty_map = mc_dropout_uncertainty(
                model, image_tensor,
                n_passes=args.mc_passes,
                roi_size=args.roi_size,
                sw_batch_size=args.sw_batch,
            )
            logger.info(f"  Uncertainty range: [{uncertainty_map.min():.3f}, {uncertainty_map.max():.3f}]")
        except Exception as e:
            logger.warning(f"MC-Dropout failed (model may have no dropout layers): {e}")
    else:
        logger.info("Step 5/6 — Uncertainty skipped (--skip-uncertainty)")

    # ── 6. Radiological features ──────────────────────────────────────────────
    logger.info("Step 6/6 — Radiological feature extraction")
    rad_features = extract_radiological_features(ct_hu, pred_mask, spacing_mm=spacing)

    # ── 7. Visualise & save ───────────────────────────────────────────────────
    logger.info("Saving visualisations")
    saved_pngs = save_xai_figure(
        ct_image=ct_hu,
        pred_mask=pred_mask,
        gradcam=gradcam_map,
        uncertainty=uncertainty_map,
        ig_map=ig_map,
        output_dir=str(out_dir),
        case_name=case_name,
    )
    saved_niftis = save_nifti_maps(
        gradcam=gradcam_map,
        uncertainty=uncertainty_map,
        ig_map=ig_map,
        reference_nifti=args.input,
        output_dir=str(out_dir),
        case_name=case_name,
    )

    # Save prediction mask as NIfTI
    pred_nii = nib.Nifti1Image(pred_mask.astype(np.int16), affine=raw_nib.affine)
    pred_path = out_dir / f"{case_name}_prediction.nii.gz"
    nib.save(pred_nii, pred_path)
    saved_niftis["prediction"] = str(pred_path)

    # Save radiological features as JSON
    rad_json_path = out_dir / f"{case_name}_radiological.json"
    rad_serialisable = {k: v for k, v in rad_features.items() if k != "explanation"}
    with open(rad_json_path, "w") as f:
        json.dump(rad_serialisable, f, indent=2)

    # ── 8. Text report ────────────────────────────────────────────────────────
    write_report(out_dir, case_name, args, pred_mask,
                 rad_features, saved_pngs, saved_niftis)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Explainable AI for PDAC segmentation model"
    )
    # Required
    p.add_argument("--input",  required=True, help="Input NIfTI CT scan (.nii.gz)")
    p.add_argument("--model",  required=True, help="Model name (e.g. nnunet)")
    p.add_argument("--fold",   type=int, default=0, help="Checkpoint fold index")

    # Optional
    p.add_argument("--label",  default=None, help="Ground-truth NIfTI (for reference only)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", default="results/xai")

    # Inference settings
    p.add_argument("--roi-size",  type=int, nargs=3, default=[96, 96, 96])
    p.add_argument("--sw-batch",  type=int, default=4)
    p.add_argument("--overlap",   type=float, default=0.25)

    # XAI toggles
    p.add_argument("--skip-gradcam",     action="store_true")
    p.add_argument("--skip-ig",          action="store_true")
    p.add_argument("--skip-uncertainty", action="store_true")
    p.add_argument("--ig-steps",  type=int, default=30,
                   help="Integration steps for Integrated Gradients")
    p.add_argument("--mc-passes", type=int, default=20,
                   help="MC-Dropout forward passes for uncertainty")

    return p.parse_args()


if __name__ == "__main__":
    main()
