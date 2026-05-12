#!/usr/bin/env python3
"""
Inference script — segment pancreas and PDAC from a 3D CT volume.

Usage:
    python predict.py input.nii.gz output_mask.nii.gz
    python predict.py input.nii.gz output_mask.nii.gz --checkpoint checkpoints/nnunet/fold_2/best_model.pth
    python predict.py input.nii.gz output_mask.nii.gz --device cpu

Output mask labels:
    0 = Background
    1 = Pancreas
    2 = PDAC (tumor)
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference

sys.path.insert(0, str(Path(__file__).resolve().parent))
from monai_benchmark.data.transforms import get_inference_transforms
from monai_benchmark.models.builder import get_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT     = Path(__file__).resolve().parent
DEFAULT_CKPT  = REPO_ROOT / "checkpoints/nnunet/fold_0/best_model.pth"
HU_WINDOW     = [-74, 140]
ROI_SIZE      = (96, 96, 96)
SW_BATCH_SIZE = 4
OVERLAP       = 0.25
LABEL_NAMES   = {0: "Background", 1: "Pancreas", 2: "PDAC"}


def load_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    model = get_model("nnunet", pretrained=False, device=device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    logger.info(f"Loaded: {ckpt_path}")
    return model


def predict(model: torch.nn.Module, image_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    """Run sliding-window inference, return (D, H, W) label map."""
    img = image_tensor.to(device)
    if img.dim() == 4:          # (1, D, H, W) → (1, 1, D, H, W)
        img = img.unsqueeze(0)

    def _fwd(x):
        out = model(x)
        return out[0] if isinstance(out, (list, tuple)) else out

    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=img,
            roi_size=ROI_SIZE,
            sw_batch_size=SW_BATCH_SIZE,
            predictor=_fwd,
            overlap=OVERLAP,
        )

    return torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Pancreas + PDAC segmentation inference.")
    parser.add_argument("input",       type=str, help="Input CT volume (.nii.gz)")
    parser.add_argument("output",      type=str, help="Output mask (.nii.gz)")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT),
                        help=f"Model checkpoint path (default: {DEFAULT_CKPT})")
    parser.add_argument("--device",    type=str, default="cuda:0")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)
    ckpt_path   = Path(args.checkpoint)

    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        sys.exit(1)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model(ckpt_path, device)

    # ── Preprocess ────────────────────────────────────────────────────────────
    tf      = get_inference_transforms(hu_window=HU_WINDOW)
    dataset = Dataset(data=[{"image": str(input_path)}], transform=tf)
    loader  = DataLoader(dataset, batch_size=1, num_workers=0)
    batch   = next(iter(loader))
    image   = batch["image"]   # (1, 1, D, H, W)

    # ── Inference ─────────────────────────────────────────────────────────────
    logger.info(f"Running inference on: {input_path.name}  shape={tuple(image.shape)}")
    pred = predict(model, image.squeeze(0), device)   # (D, H, W)

    # ── Print stats ───────────────────────────────────────────────────────────
    total = pred.size
    for label_id, label_name in LABEL_NAMES.items():
        voxels = int((pred == label_id).sum())
        logger.info(f"  {label_name:12s} (label {label_id}): {voxels:>8,} voxels  ({100*voxels/total:.1f}%)")

    # ── Save mask — preserve original affine/header ───────────────────────────
    orig_nib = nib.load(str(input_path))
    mask_nib = nib.Nifti1Image(pred, affine=orig_nib.affine, header=orig_nib.header)
    mask_nib.header.set_data_dtype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(mask_nib, str(output_path))
    logger.info(f"Mask saved → {output_path}")


if __name__ == "__main__":
    main()
