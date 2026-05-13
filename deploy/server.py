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
MODEL_NAME      = os.environ.get("MODEL_NAME", "nnunet")
MODEL_FOLD      = int(os.environ.get("MODEL_FOLD", "0"))
CKPT_DIR        = Path(os.environ.get("CKPT_DIR", "/app/checkpoints"))
CKPT_PATH       = os.environ.get("CKPT_PATH", "")   # optional: direct path to a .pth file
HU_MIN          = float(os.environ.get("HU_MIN", "-74"))
HU_MAX          = float(os.environ.get("HU_MAX", "140"))
DEVICE_STR      = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
ROI_SIZE        = [96, 96, 96]
SW_BATCH        = int(os.environ.get("SW_BATCH", "4"))
SW_OVERLAP      = float(os.environ.get("SW_OVERLAP", "0.25"))

# Two-stage config
USE_TWO_STAGE   = os.environ.get("USE_TWO_STAGE", "false").lower() == "true"
STAGE2_FOLD     = int(os.environ.get("STAGE2_FOLD", str(MODEL_FOLD)))
STAGE2_CKPT_PATH = os.environ.get("STAGE2_CKPT_PATH", "")
BBOX_MARGIN     = int(os.environ.get("BBOX_MARGIN", "20"))

# Architecture name: strip "_finetuned" suffix so builder.get_model() resolves correctly
ARCH_NAME       = MODEL_NAME.replace("_finetuned", "")

device = torch.device(DEVICE_STR)

_stage2_model = None


def _resolve_stage2_ckpt_path() -> Path:
    if STAGE2_CKPT_PATH:
        p = Path(STAGE2_CKPT_PATH)
        if not p.exists():
            raise RuntimeError(f"STAGE2_CKPT_PATH không tồn tại: {p}")
        return p
    fold_dir = CKPT_DIR / "nnunet_stage2" / f"fold_{STAGE2_FOLD}"
    best = fold_dir / "best_model.pth"
    resume = fold_dir / "resume.pth"
    if best.exists():
        return best
    if resume.exists():
        logger.warning(f"stage2 best_model.pth không tồn tại, dùng resume.pth: {resume}")
        return resume
    raise RuntimeError(f"Không tìm thấy stage-2 checkpoint trong {fold_dir}")


def _get_stage2_model():
    global _stage2_model
    if _stage2_model is not None:
        return _stage2_model
    ckpt_path = _resolve_stage2_ckpt_path()
    model = get_model("nnunet", pretrained=False, device=device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    model.eval()
    logger.info(f"Stage-2 model loaded: {ckpt_path}")
    _stage2_model = model
    return _stage2_model


def _compute_bbox(mask: np.ndarray, margin: int):
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None
    mins = np.maximum(coords.min(axis=0) - margin, 0)
    maxs = np.minimum(coords.max(axis=0) + margin, np.array(mask.shape) - 1)
    return mins, maxs


app = FastAPI(
    title="Pancreas Segmentation API",
    description="3-class CT segmentation: background / pancreas / tumor",
    version="1.0.0",
)

# ── Model loading (done once at startup) ─────────────────────────────────────

_model = None

def _resolve_ckpt_path() -> Path:
    """Resolve checkpoint path with fallback: CKPT_PATH > best_model.pth > resume.pth."""
    if CKPT_PATH:
        p = Path(CKPT_PATH)
        if not p.exists():
            raise RuntimeError(f"CKPT_PATH không tồn tại: {p}")
        return p

    fold_dir = CKPT_DIR / MODEL_NAME / f"fold_{MODEL_FOLD}"
    best = fold_dir / "best_model.pth"
    resume = fold_dir / "resume.pth"

    if best.exists():
        return best
    if resume.exists():
        logger.warning(f"best_model.pth không tồn tại, dùng resume.pth: {resume}")
        return resume
    raise RuntimeError(
        f"Không tìm thấy checkpoint trong {fold_dir} "
        f"(thử best_model.pth và resume.pth)"
    )


def _get_model():
    global _model
    if _model is not None:
        return _model

    ckpt_path = _resolve_ckpt_path()
    model = get_model(ARCH_NAME, pretrained=False, device=device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    model.eval()
    logger.info(f"Model loaded: {ckpt_path}  (arch={ARCH_NAME})")
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


def _sliding_window(model, image_tensor: torch.Tensor) -> torch.Tensor:
    def _fwd(x):
        out = model(x)
        return out[0] if isinstance(out, (list, tuple)) else out
    with torch.no_grad():
        return sliding_window_inference(
            inputs=image_tensor.to(device),
            roi_size=ROI_SIZE,
            sw_batch_size=SW_BATCH,
            predictor=_fwd,
            overlap=SW_OVERLAP,
        )


def _run_inference(image_tensor: torch.Tensor) -> np.ndarray:
    """Return (D, H, W) int8 segmentation mask — single-stage."""
    logits = _sliding_window(_get_model(), image_tensor)
    return torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int8)


def _run_two_stage_inference(image_tensor: torch.Tensor) -> np.ndarray:
    """Return (D, H, W) int8 mask using two-stage pipeline."""
    full_shape = tuple(image_tensor.shape[2:])   # (D, H, W)

    # Stage 1: coarse segmentation on full CT
    logits1  = _sliding_window(_get_model(), image_tensor)
    pred1    = torch.argmax(logits1, dim=1).squeeze(0).cpu().numpy()
    del logits1

    bbox = _compute_bbox((pred1 > 0).astype(np.uint8), BBOX_MARGIN)
    if bbox is None:
        logger.warning("Stage 1 không phát hiện pancreas — fallback single-stage.")
        return pred1.astype(np.int8)

    mins, maxs = bbox
    d0, h0, w0 = mins.tolist()
    d1, h1, w1 = (maxs + 1).tolist()
    logger.info(f"Two-stage bbox: D[{d0}:{d1}] H[{h0}:{h1}] W[{w0}:{w1}]")

    # Crop CT tensor
    crop = image_tensor[:, :, d0:d1, h0:h1, w0:w1]
    if any(crop.shape[2 + i] < ROI_SIZE[i] for i in range(3)):
        from monai.transforms import SpatialPad
        crop = SpatialPad(spatial_size=ROI_SIZE)(crop.squeeze(0)).unsqueeze(0)

    # Stage 2: fine segmentation on crop
    torch.cuda.empty_cache()
    logits2 = _sliding_window(_get_stage2_model(), crop)
    pred2   = torch.argmax(logits2, dim=1).squeeze(0).cpu().numpy()
    del logits2

    # Paste back
    full_mask = np.zeros(full_shape, dtype=np.int8)
    cd, ch, cw = pred2.shape
    full_mask[d0:d0+min(d1-d0, cd),
              h0:h0+min(h1-h0, ch),
              w0:w0+min(w1-w0, cw)] = pred2[:min(d1-d0, cd),
                                             :min(h1-h0, ch),
                                             :min(w1-w0, cw)].astype(np.int8)
    return full_mask


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
        logger.warning(f"Stage-1 model not pre-loaded at startup: {e}")
    if USE_TWO_STAGE:
        try:
            _get_stage2_model()
        except Exception as e:
            logger.warning(f"Stage-2 model not pre-loaded at startup: {e}")


@app.get("/health")
def health():
    try:
        ckpt_path = str(_resolve_ckpt_path())
    except Exception as e:
        ckpt_path = f"ERROR: {e}"
    info = {
        "status": "ok",
        "model": MODEL_NAME,
        "arch": ARCH_NAME,
        "fold": MODEL_FOLD,
        "checkpoint": ckpt_path,
        "device": DEVICE_STR,
        "model_loaded": _model is not None,
        "two_stage": USE_TWO_STAGE,
    }
    if USE_TWO_STAGE:
        try:
            s2_ckpt = str(_resolve_stage2_ckpt_path())
        except Exception as e:
            s2_ckpt = f"ERROR: {e}"
        info["stage2_checkpoint"] = s2_ckpt
        info["stage2_loaded"] = _stage2_model is not None
    return info


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
        mask = _run_two_stage_inference(image_tensor) if USE_TWO_STAGE else _run_inference(image_tensor)

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
