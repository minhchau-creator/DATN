#!/usr/bin/env python3
"""
Visualization utilities for XAI outputs.

Generates multi-panel PNG figures with:
  - Original CT slice (windowed)
  - Predicted segmentation overlay
  - GradCAM heatmap overlay
  - Uncertainty map
  - Integrated Gradients overlay

One figure per best axial slice (highest tumor content),
plus separate coronal and sagittal views.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

logger = logging.getLogger(__name__)

# Colour scheme
SEG_COLORS = {
    0: (0.0, 0.0, 0.0, 0.0),    # background: transparent
    1: (0.2, 0.8, 0.2, 0.45),   # pancreas: green
    2: (1.0, 0.2, 0.2, 0.55),   # tumor: red
}
CMAP_CAM  = "hot"
CMAP_UNC  = "plasma"
HU_WINDOW = (-100, 200)          # CT display window (soft tissue)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _window_ct(ct: np.ndarray, lo=HU_WINDOW[0], hi=HU_WINDOW[1]) -> np.ndarray:
    """Clip and normalise CT slice to [0, 1] for display."""
    ct = np.clip(ct, lo, hi)
    return (ct - lo) / (hi - lo + 1e-8)


def _seg_rgba(mask_2d: np.ndarray) -> np.ndarray:
    """Convert 2D integer mask to RGBA overlay image."""
    H, W = mask_2d.shape
    rgba = np.zeros((H, W, 4), dtype=np.float32)
    for cls, color in SEG_COLORS.items():
        where = mask_2d == cls
        rgba[where] = color
    return rgba


def _best_slice(volume: np.ndarray, axis: int = 0) -> int:
    """Return the slice index along `axis` with the most non-zero voxels."""
    sums = []
    for i in range(volume.shape[axis]):
        sl = np.take(volume, i, axis=axis)
        sums.append((sl > 0).sum())
    return int(np.argmax(sums))


def _take(volume: np.ndarray, idx: int, axis: int) -> np.ndarray:
    return np.take(volume, idx, axis=axis)


# ─────────────────────────────────────────────────────────────────────────────
# Main visualisation function
# ─────────────────────────────────────────────────────────────────────────────

def save_xai_figure(
    ct_image:     np.ndarray,        # (D, H, W) raw HU
    pred_mask:    np.ndarray,        # (D, H, W) int labels
    gradcam:      Optional[np.ndarray] = None,    # (D, H, W) [0,1]
    uncertainty:  Optional[np.ndarray] = None,    # (D, H, W) [0,1]
    ig_map:       Optional[np.ndarray] = None,    # (D, H, W) [0,1]
    output_dir:   str = "results/xai",
    case_name:    str = "case",
) -> Dict[str, str]:
    """
    Save multi-panel XAI figures (axial / coronal / sagittal) as PNGs.

    Returns dict mapping view_name → saved file path.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tumor_mask = (pred_mask == 2)
    saved = {}

    for view_name, axis in [("axial", 0), ("coronal", 1), ("sagittal", 2)]:
        idx = _best_slice(tumor_mask, axis=axis) if tumor_mask.any() else ct_image.shape[axis] // 2

        ct_sl   = _take(ct_image,  idx, axis)
        seg_sl  = _take(pred_mask, idx, axis)
        cam_sl  = _take(gradcam,   idx, axis) if gradcam   is not None else None
        unc_sl  = _take(uncertainty, idx, axis) if uncertainty is not None else None
        ig_sl   = _take(ig_map,    idx, axis) if ig_map    is not None else None

        # Count how many panels we need
        panels = [("CT + Prediction", ct_sl, seg_sl, None, None)]
        if cam_sl is not None:
            panels.append(("GradCAM", ct_sl, None, cam_sl, CMAP_CAM))
        if ig_sl is not None:
            panels.append(("Integrated Gradients", ct_sl, None, ig_sl, "viridis"))
        if unc_sl is not None:
            panels.append(("Uncertainty", ct_sl, None, unc_sl, CMAP_UNC))

        n = len(panels)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]

        for ax, (title, ct_sl_, seg_sl_, hmap_sl, cmap) in zip(axes, panels):
            ct_disp = _window_ct(ct_sl_)
            ax.imshow(ct_disp, cmap="gray", origin="lower", interpolation="bilinear")

            if seg_sl_ is not None:
                ax.imshow(_seg_rgba(seg_sl_), origin="lower", interpolation="nearest")

            if hmap_sl is not None:
                # Only show heatmap where tumor is predicted
                hmap_masked = np.ma.masked_where(hmap_sl < 0.05, hmap_sl)
                ax.imshow(hmap_masked, cmap=cmap, alpha=0.65, vmin=0, vmax=1,
                          origin="lower", interpolation="bilinear")

            ax.set_title(f"{title}\n({view_name} slice {idx})", fontsize=9)
            ax.axis("off")

        # Legend for first panel
        legend_patches = [
            Patch(color=(0.2, 0.8, 0.2), label="Pancreas"),
            Patch(color=(1.0, 0.2, 0.2), label="PDAC Tumor"),
        ]
        axes[0].legend(handles=legend_patches, loc="lower right",
                       fontsize=7, framealpha=0.7)

        fig.suptitle(f"{case_name} — XAI Explanation", fontsize=11, y=1.01)
        plt.tight_layout()

        save_path = out_dir / f"{case_name}_{view_name}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved[view_name] = str(save_path)
        logger.info(f"Saved {view_name} view → {save_path}")

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# NIfTI export (for 3D Slicer / ITK-SNAP)
# ─────────────────────────────────────────────────────────────────────────────

def save_nifti_maps(
    gradcam:     Optional[np.ndarray],
    uncertainty: Optional[np.ndarray],
    ig_map:      Optional[np.ndarray],
    reference_nifti: str,
    output_dir:  str = "results/xai",
    case_name:   str = "case",
) -> Dict[str, str]:
    """
    Save heatmaps as NIfTI files so they can be overlaid in 3D Slicer.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref  = nib.load(reference_nifti)
    saved = {}

    maps = {
        "gradcam":      gradcam,
        "uncertainty":  uncertainty,
        "intgrad":      ig_map,
    }
    for name, arr in maps.items():
        if arr is None:
            continue
        nii = nib.Nifti1Image(arr.astype(np.float32), affine=ref.affine)
        path = out_dir / f"{case_name}_{name}.nii.gz"
        nib.save(nii, path)
        saved[name] = str(path)
        logger.info(f"Saved {name} NIfTI → {path}")

    return saved
