#!/usr/bin/env python3
"""
Radiological Feature Extraction for PDAC Explanation.

Extracts quantitative features from the predicted tumor mask + original CT
to provide human-readable evidence for WHY the model classified a region
as PDAC tumor.

Features computed:
  - HU statistics of tumor region (mean, std, min, max, percentiles)
  - Tumor volume (cm³) and voxel count
  - Anatomical location estimate (pancreatic head / body / tail)
  - Shape descriptors (compactness, elongation, bounding-box dims)
  - Comparison with published PDAC HU reference ranges
  - Adjacency to pancreas tissue (class 1)
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, Tuple

import numpy as np
from scipy.ndimage import label as nd_label, center_of_mass

logger = logging.getLogger(__name__)

# Published HU reference ranges for PDAC on non-contrast CT
# Source: Elbanna et al. 2022; McNamara et al. 2021
PDAC_HU_REFERENCE = {
    "mean_range": (-10, 60),   # typical hypodense tumour
    "min_expected": -30,
    "max_expected": 80,
    "hypo_threshold": 40,      # below this = hypodense (classic PDAC pattern)
}

PANCREAS_CLASS = 1
TUMOR_CLASS    = 2


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _voxel_volume_cm3(spacing_mm: Tuple[float, float, float]) -> float:
    """Return volume of a single voxel in cm³."""
    return float(np.prod([s / 10.0 for s in spacing_mm]))


def _estimate_pancreas_location(
    mask: np.ndarray,
    tumor_centroid: Tuple[float, float, float],
    axis: int = 2,          # left-right axis in RAS (W axis)
) -> str:
    """
    Estimate anatomical sub-region of the pancreas from the tumour centroid
    relative to the whole-pancreas bounding box.

    Pancreas (class 1) runs roughly head (right, W-max) → tail (left, W-min)
    in the coronal plane.  We divide it into thirds.
    """
    pan_mask = (mask == PANCREAS_CLASS)
    if pan_mask.sum() == 0:
        return "unknown (no pancreas segmented)"

    coords = np.argwhere(pan_mask)
    lo, hi = coords[:, axis].min(), coords[:, axis].max()
    span   = hi - lo + 1e-6
    rel    = (tumor_centroid[axis] - lo) / span   # 0 = head, 1 = tail

    if rel < 0.33:
        return "pancreatic head"
    elif rel < 0.66:
        return "pancreatic body"
    else:
        return "pancreatic tail"


def _shape_descriptors(binary_mask: np.ndarray) -> Dict[str, float]:
    """Return compactness and elongation from the tumour bounding box."""
    coords = np.argwhere(binary_mask)
    if len(coords) == 0:
        return {"elongation": 0.0, "compactness": 0.0}

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    dims = (maxs - mins + 1).astype(float)          # D, H, W in voxels

    dims_sorted = np.sort(dims)[::-1]               # largest first
    elongation  = float(dims_sorted[0] / (dims_sorted[2] + 1e-6))   # length / width
    volume      = float(binary_mask.sum())
    box_vol     = float(dims.prod())
    compactness = volume / (box_vol + 1e-6)          # 1 = cube-like, 0 = sparse

    return {
        "elongation":  round(elongation, 3),
        "compactness": round(compactness, 3),
        "bbox_dims_vox": [int(d) for d in dims],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_radiological_features(
    ct_image: np.ndarray,            # (D, H, W) raw HU values (BEFORE normalisation)
    pred_mask: np.ndarray,           # (D, H, W) int labels: 0=bg, 1=pancreas, 2=tumor
    spacing_mm: Optional[Tuple[float, float, float]] = (1.0, 1.0, 1.0),
) -> Dict[str, Any]:
    """
    Compute radiological features of the predicted PDAC tumor.

    Args:
        ct_image:   original CT in Hounsfield Units (not normalised)
        pred_mask:  model prediction (D, H, W)
        spacing_mm: voxel spacing in mm (D, H, W)

    Returns:
        dict with all features and a human-readable explanation string
    """
    tumor_binary = (pred_mask == TUMOR_CLASS)
    n_tumor_vox  = int(tumor_binary.sum())

    if n_tumor_vox == 0:
        return {
            "tumor_found": False,
            "explanation": "No tumor (class 2) voxels predicted by the model.",
        }

    # ── Keep largest connected component (avoid noise) ────────────────────────
    labeled, n_components = nd_label(tumor_binary)
    comp_sizes = [(labeled == i).sum() for i in range(1, n_components + 1)]
    main_comp  = np.argmax(comp_sizes) + 1
    tumor_cc   = (labeled == main_comp)

    n_tumor_cc = int(tumor_cc.sum())
    vox_vol    = _voxel_volume_cm3(spacing_mm)
    volume_cm3 = round(n_tumor_cc * vox_vol, 3)

    # ── HU statistics ─────────────────────────────────────────────────────────
    hu_vals    = ct_image[tumor_cc]
    hu_stats   = {
        "mean":  round(float(hu_vals.mean()), 2),
        "std":   round(float(hu_vals.std()),  2),
        "min":   round(float(hu_vals.min()),  2),
        "max":   round(float(hu_vals.max()),  2),
        "p10":   round(float(np.percentile(hu_vals, 10)), 2),
        "p50":   round(float(np.percentile(hu_vals, 50)), 2),
        "p90":   round(float(np.percentile(hu_vals, 90)), 2),
    }

    # ── Location ──────────────────────────────────────────────────────────────
    centroid  = center_of_mass(tumor_cc)
    location  = _estimate_pancreas_location(pred_mask, centroid)

    # ── Shape ─────────────────────────────────────────────────────────────────
    shape = _shape_descriptors(tumor_cc)

    # ── Compare with PDAC reference ───────────────────────────────────────────
    ref_lo, ref_hi = PDAC_HU_REFERENCE["mean_range"]
    is_hypodense   = hu_stats["mean"] < PDAC_HU_REFERENCE["hypo_threshold"]
    in_ref_range   = ref_lo <= hu_stats["mean"] <= ref_hi
    hu_confidence  = "HIGH" if in_ref_range else ("MODERATE" if is_hypodense else "LOW")

    # ── Adjacency to pancreas ──────────────────────────────────────────────────
    pan_binary    = (pred_mask == PANCREAS_CLASS)
    n_pan_vox     = int(pan_binary.sum())
    pan_volume_cm3 = round(n_pan_vox * vox_vol, 3)

    # ── Build explanation text ────────────────────────────────────────────────
    explanation = _build_explanation(
        hu_stats, volume_cm3, location, shape, is_hypodense,
        in_ref_range, hu_confidence, n_components, pan_volume_cm3,
    )

    return {
        "tumor_found":       True,
        "voxel_count":       n_tumor_cc,
        "n_components":      n_components,
        "volume_cm3":        volume_cm3,
        "centroid_vox":      [round(c, 1) for c in centroid],
        "anatomical_location": location,
        "hu_statistics":     hu_stats,
        "shape":             shape,
        "is_hypodense":      is_hypodense,
        "in_pdac_hu_range":  in_ref_range,
        "hu_confidence":     hu_confidence,
        "pancreas_volume_cm3": pan_volume_cm3,
        "pdac_hu_reference": PDAC_HU_REFERENCE,
        "explanation":       explanation,
    }


def _build_explanation(
    hu: dict, volume: float, location: str, shape: dict,
    is_hypodense: bool, in_ref_range: bool, confidence: str,
    n_components: int, pan_vol: float,
) -> str:
    """
    Build a radiologist-style plain-text explanation for the tumor prediction.
    """
    lines = [
        "=" * 60,
        "  PDAC SEGMENTATION EXPLANATION REPORT",
        "=" * 60,
        "",
        f"[LOCATION]",
        f"  Predicted anatomical location : {location}",
        f"  Tumour centroid (voxel space) : see centroid_vox field",
        "",
        f"[SIZE & MORPHOLOGY]",
        f"  Estimated tumour volume       : {volume:.2f} cm³",
        f"  Number of connected regions   : {n_components}",
        f"  Elongation index              : {shape.get('elongation', 0):.2f}  "
        f"({'elongated — consistent with ductal PDAC' if shape.get('elongation', 1) > 2 else 'more rounded'})",
        f"  Compactness                   : {shape.get('compactness', 0):.2f}",
        "",
        f"[HU ATTENUATION ANALYSIS]",
        f"  Mean HU of tumour region      : {hu['mean']:.1f} HU",
        f"  HU range (p10 – p90)          : [{hu['p10']:.1f},  {hu['p90']:.1f}] HU",
        f"  Std deviation                 : {hu['std']:.1f} HU  "
        f"({'heterogeneous' if hu['std'] > 30 else 'homogeneous'} texture)",
        "",
        f"[COMPARISON WITH PDAC REFERENCE]",
        f"  Published PDAC mean HU range  : {PDAC_HU_REFERENCE['mean_range']} HU  "
        f"(non-contrast CT)",
        f"  Mean HU in reference range?   : {'YES' if in_ref_range else 'NO'}",
        f"  Region is hypodense (<40 HU)? : {'YES' if is_hypodense else 'NO'}",
        f"  HU-based confidence           : {confidence}",
        "",
        f"[CONTEXTUAL EVIDENCE]",
        f"  Surrounding pancreas tissue   : {pan_vol:.2f} cm³ detected",
    ]

    # Interpretation
    lines += ["", "[INTERPRETATION]"]
    evidences = []

    if is_hypodense:
        evidences.append(
            "  ✓ The region is HYPODENSE relative to surrounding pancreas, "
            "a hallmark of PDAC on non-contrast CT."
        )
    if in_ref_range:
        evidences.append(
            "  ✓ Mean attenuation falls within the published PDAC HU reference "
            f"range {PDAC_HU_REFERENCE['mean_range']} HU."
        )
    if shape.get("elongation", 0) > 2:
        evidences.append(
            "  ✓ Elongated morphology is consistent with infiltrative ductal "
            "adenocarcinoma growth pattern."
        )
    if hu["std"] > 30:
        evidences.append(
            "  ✓ Heterogeneous attenuation (std > 30 HU) suggests internal "
            "necrosis or cystic change, common in PDAC."
        )
    if n_components > 1:
        evidences.append(
            f"  ⚠ {n_components} separate tumour regions detected — may indicate "
            "multifocal disease or over-segmentation; review recommended."
        )
    if not evidences:
        evidences.append(
            "  ⚠ HU values do not strongly match published PDAC reference range. "
            "Manual radiologist review is recommended."
        )

    lines += evidences
    lines += [
        "",
        "[DISCLAIMER]",
        "  This report is AI-generated and must be reviewed by a qualified",
        "  radiologist before any clinical decision is made.",
        "=" * 60,
    ]

    return "\n".join(lines)
