#!/usr/bin/env python3
"""
Two-Stage Inference Pipeline.

Stage 1: Full-CT inference với model fold 0 → bounding box quanh pancreas+tumor
Stage 2: Inference trên vùng crop → kết quả chi tiết hơn về pancreas/tumor

Usage:
    python monai_benchmark/predict_twostage.py \
        --input  dataset/Task_7/imagesTr/pancreas_001.nii.gz \
        --output result_mask/pancreas_001_pred.nii.gz \
        --stage1-ckpt checkpoints/nnunet/fold_0/best_model.pth \
        --stage2-ckpt checkpoints/nnunet_stage2/fold_0/best_model.pth
"""

import sys, argparse, logging
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, Orientation,
    ScaleIntensityRange, ToTensor, SpatialPad,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from monai_benchmark.models.builder import get_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROI_SIZE   = [96, 96, 96]
SW_BATCH   = 4
SW_OVERLAP = 0.25
MARGIN     = 20   # voxels around stage-1 bounding box


# ── Preprocessing ──────────────────────────────────────────────────────────────

_preprocess = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    Orientation(axcodes="RAS"),
    ScaleIntensityRange(a_min=-74, a_max=140, b_min=0.0, b_max=1.0, clip=True),
    ToTensor(),
])

_padder = SpatialPad(spatial_size=ROI_SIZE)


def _load_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    model = get_model("nnunet", pretrained=False, device=device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    model.eval()
    logger.info(f"Loaded: {ckpt_path}")
    return model


def _sliding_window(model, image_batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    def _fwd(x):
        o = model(x)
        return o[0] if isinstance(o, (list, tuple)) else o
    with torch.no_grad():
        return sliding_window_inference(
            image_batch.to(device), ROI_SIZE, SW_BATCH, _fwd, overlap=SW_OVERLAP
        )


def _compute_bbox(mask: np.ndarray, shape: tuple, margin: int):
    """Return (mins, maxs) bounding box of mask>0 with margin, clamped to shape."""
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None
    mins = np.maximum(coords.min(axis=0) - margin, 0)
    maxs = np.minimum(coords.max(axis=0) + margin, np.array(shape) - 1)
    return mins, maxs


# ── Main inference function ────────────────────────────────────────────────────

def predict_twostage(
    nifti_path: str,
    stage1_ckpt: str,
    stage2_ckpt: str,
    device: torch.device,
    margin: int = MARGIN,
) -> np.ndarray:
    """
    Run two-stage inference on a NIfTI CT scan.

    Returns:
        mask (D, H, W) int8 with labels 0=background, 1=pancreas, 2=tumor
    """
    # ── 1. Preprocess full CT ─────────────────────────────────────────────────
    image = _preprocess(nifti_path)       # (1, D, H, W)
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image_batch = image.unsqueeze(0)      # (1, 1, D, H, W)
    full_shape  = tuple(image.shape[1:])  # (D, H, W)
    logger.info(f"Input shape: {full_shape}")

    # ── 2. Stage 1: coarse segmentation on full CT ───────────────────────────
    stage1 = _load_model(stage1_ckpt, device)
    logits1 = _sliding_window(stage1, image_batch, device)
    pred1   = torch.argmax(logits1, dim=1).squeeze(0).cpu().numpy()   # (D, H, W)
    del stage1, logits1
    torch.cuda.empty_cache()

    foreground = (pred1 > 0).astype(np.uint8)
    bbox = _compute_bbox(foreground, full_shape, margin)

    if bbox is None:
        logger.warning("Stage 1 không phát hiện pancreas — dùng kết quả stage 1 trực tiếp.")
        return pred1.astype(np.int8)

    mins, maxs = bbox
    d0, h0, w0 = mins.tolist()
    d1, h1, w1 = (maxs + 1).tolist()
    logger.info(f"Bounding box: D[{d0}:{d1}]  H[{h0}:{h1}]  W[{w0}:{w1}]")

    # ── 3. Crop CT tại bounding box ───────────────────────────────────────────
    crop = image_batch[:, :, d0:d1, h0:h1, w0:w1]   # (1, 1, D', H', W')

    # Pad nếu crop nhỏ hơn ROI_SIZE
    c_shape = crop.shape[2:]
    if any(c_shape[i] < ROI_SIZE[i] for i in range(3)):
        crop = _padder(crop.squeeze(0)).unsqueeze(0)
        logger.info(f"Padded crop to: {crop.shape[2:]}")

    # ── 4. Stage 2: fine segmentation on crop ────────────────────────────────
    stage2  = _load_model(stage2_ckpt, device)
    logits2 = _sliding_window(stage2, crop, device)
    pred2   = torch.argmax(logits2, dim=1).squeeze(0).cpu().numpy()   # (D', H', W')
    del stage2, logits2, crop
    torch.cuda.empty_cache()

    # ── 5. Paste stage-2 result back into full-volume mask ───────────────────
    full_mask = np.zeros(full_shape, dtype=np.int8)
    cd, ch, cw = pred2.shape
    # Actual size might differ from bbox if padding extended beyond bounds
    actual_d = min(d1 - d0, cd)
    actual_h = min(h1 - h0, ch)
    actual_w = min(w1 - w0, cw)
    full_mask[d0:d0+actual_d, h0:h0+actual_h, w0:w0+actual_w] = \
        pred2[:actual_d, :actual_h, :actual_w].astype(np.int8)

    pan_vox = int((full_mask == 1).sum())
    tum_vox = int((full_mask == 2).sum())
    logger.info(f"Final mask — pancreas: {pan_vox} voxels  tumor: {tum_vox} voxels")

    return full_mask


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Two-stage pancreas/tumor segmentation")
    parser.add_argument("--input",        required=True,  help="Input NIfTI (.nii/.nii.gz)")
    parser.add_argument("--output",       required=True,  help="Output mask NIfTI (.nii.gz)")
    parser.add_argument("--stage1-ckpt",  required=True,  help="Stage-1 checkpoint (.pth)")
    parser.add_argument("--stage2-ckpt",  required=True,  help="Stage-2 checkpoint (.pth)")
    parser.add_argument("--device",       default="cuda", help="cuda or cpu")
    parser.add_argument("--margin",       type=int, default=MARGIN)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    mask = predict_twostage(
        nifti_path=args.input,
        stage1_ckpt=args.stage1_ckpt,
        stage2_ckpt=args.stage2_ckpt,
        device=device,
        margin=args.margin,
    )

    ref = nib.load(args.input)
    out = nib.Nifti1Image(mask, affine=ref.affine, header=ref.header)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, args.output)
    logger.info(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
