#!/usr/bin/env python3
"""
FastAPI Inference Server — Pancreas / Tumor Segmentation

Endpoints:
    GET  /health          — liveness check
    GET  /models          — list available model checkpoints
    POST /predict         — upload a NIfTI CT scan, get back a segmentation mask

Environment variables:
    MODEL_NAME   e.g. "nnunet"  (default: nnunet)
    MODEL_FOLD   e.g. "0"       (default: 0)
    CKPT_DIR     path to checkpoints dir (default: /app/checkpoints)
    HU_MIN / HU_MAX             (default: -74 / 140)
    DEVICE                      (default: cuda if available, else cpu)
"""

import io
import os
import sys
import logging
import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import Response

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Add project root so we can import from monai_benchmark ────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from monai_benchmark.models.builder import get_model
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, Orientation,
    ScaleIntensityRange, ToTensor,
)

# ── Config from environment ───────────────────────────────────────────────────
MODEL_NAME  = os.environ.get("MODEL_NAME", "nnunet")
MODEL_FOLD  = int(os.environ.get("MODEL_FOLD", "0"))
CKPT_DIR    = Path(os.environ.get("CKPT_DIR", "/app/checkpoints"))
HU_MIN      = float(os.environ.get("HU_MIN", "-74"))
HU_MAX      = float(os.environ.get("HU_MAX", "140"))
DEVICE_STR  = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
ROI_SIZE    = [96, 96, 96]
SW_BATCH    = int(os.environ.get("SW_BATCH", "4"))
SW_OVERLAP  = float(os.environ.get("SW_OVERLAP", "0.25"))

device = torch.device(DEVICE_STR)

app = FastAPI(
    title="Pancreas Segmentation API",
    description="3-class CT segmentation: background / pancreas / tumor",
    version="1.0.0",
)

# ── Model loading (done once at startup) ─────────────────────────────────────

_model = None

def _get_model():
    global _model
    if _model is not None:
        return _model

    ckpt_path = CKPT_DIR / MODEL_NAME / f"fold_{MODEL_FOLD}" / "best_model.pth"
    if not ckpt_path.exists():
        raise RuntimeError(f"Checkpoint not found: {ckpt_path}")

    model = get_model(MODEL_NAME, pretrained=False, device=device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    model.eval()
    logger.info(f"Model loaded: {ckpt_path}")
    _model = model
    return _model


# ── Preprocessing pipeline (CPU, no label) ───────────────────────────────────

_preprocess = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    Orientation(axcodes="RAS"),
    ScaleIntensityRange(a_min=HU_MIN, a_max=HU_MAX, b_min=0.0, b_max=1.0, clip=True),
    ToTensor(),
])


def _preprocess_file(nifti_path: str) -> torch.Tensor:
    """Load NIfTI, apply preprocessing, return (1, D, H, W) tensor."""
    tensor = _preprocess(nifti_path)
    if tensor.dim() == 3:          # (D, H, W)
        tensor = tensor.unsqueeze(0)
    return tensor.unsqueeze(0)     # (1, 1, D, H, W)


def _run_inference(image_tensor: torch.Tensor) -> np.ndarray:
    """Return (D, H, W) int8 segmentation mask."""
    model = _get_model()

    def _fwd(x):
        out = model(x)
        return out[0] if isinstance(out, (list, tuple)) else out

    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=image_tensor.to(device),
            roi_size=ROI_SIZE,
            sw_batch_size=SW_BATCH,
            predictor=_fwd,
            overlap=SW_OVERLAP,
        )
    return torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int8)


def _mask_to_nifti_bytes(mask: np.ndarray, reference_path: str) -> bytes:
    """Save prediction as NIfTI, preserving affine from original CT."""
    ref = nib.load(reference_path)
    nii = nib.Nifti1Image(mask, affine=ref.affine, header=ref.header)
    buf = io.BytesIO()
    nib.save(nii, buf)
    return buf.getvalue()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    try:
        _get_model()
    except Exception as e:
        logger.warning(f"Model not pre-loaded at startup: {e}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "fold": MODEL_FOLD,
        "device": DEVICE_STR,
        "model_loaded": _model is not None,
    }


@app.get("/models")
def list_models():
    """Return all available checkpoint paths under CKPT_DIR."""
    checkpoints = []
    for p in sorted(CKPT_DIR.rglob("best_model.pth")):
        rel = p.relative_to(CKPT_DIR)
        parts = rel.parts  # e.g. ("nnunet", "fold_0", "best_model.pth")
        checkpoints.append({
            "model": parts[0] if len(parts) > 0 else "?",
            "fold":  parts[1] if len(parts) > 1 else "?",
            "path":  str(p),
        })
    return {"checkpoints": checkpoints}


@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="CT scan in NIfTI (.nii or .nii.gz) format"),
    return_nifti: bool = Query(True, description="Return NIfTI file (True) or raw JSON stats (False)"),
):
    """
    Upload a CT NIfTI scan and receive the segmentation mask.

    - **file**: .nii or .nii.gz CT image
    - **return_nifti**: if True (default), returns the mask as .nii.gz;
                        if False, returns JSON with per-class voxel counts
    """
    suffix = ".nii.gz" if file.filename.endswith(".gz") else ".nii"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        image_tensor = _preprocess_file(tmp_path)
        mask = _run_inference(image_tensor)

        if return_nifti:
            nifti_bytes = _mask_to_nifti_bytes(mask, tmp_path)
            return Response(
                content=nifti_bytes,
                media_type="application/gzip",
                headers={"Content-Disposition": "attachment; filename=segmentation.nii.gz"},
            )
        else:
            unique, counts = np.unique(mask, return_counts=True)
            label_names = {0: "background", 1: "pancreas", 2: "tumor"}
            stats = {label_names.get(int(u), f"class_{u}"): int(c)
                     for u, c in zip(unique, counts)}
            total = int(mask.size)
            return {
                "model": MODEL_NAME,
                "fold": MODEL_FOLD,
                "voxel_counts": stats,
                "total_voxels": total,
                "pancreas_volume_pct": round(stats.get("pancreas", 0) / total * 100, 2),
                "tumor_volume_pct":    round(stats.get("tumor", 0)    / total * 100, 2),
            }

    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
