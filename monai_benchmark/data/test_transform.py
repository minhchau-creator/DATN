#!/usr/bin/env python3
"""
PANORAMA baseline preprocessing.

Usage:
    # Pipeline lowres — resample về (4.5, 4.5, 9.0) mm
    python test_transform.py input.nii.gz output.nii.gz

    # Pipeline highres — crop ROI tụy (cần thêm seg mask)
    python test_transform.py input.nii.gz output.nii.gz --seg pancreas_seg.nii.gz

    # Ảnh nhãn (dùng NearestNeighbor thay BSpline)
    python test_transform.py label.nii.gz label_out.nii.gz --label

    # Tùy chỉnh spacing
    python test_transform.py input.nii.gz output.nii.gz --spacing 4.5 4.5 9.0
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LOWRES_SPACING   = [4.5, 4.5, 9.0]
DEFAULT_MARGINS  = [100.0, 50.0, 15.0]


# ---------------------------------------------------------------------------
# Hàm gốc từ PANORAMA baseline — data_utils.py
# ---------------------------------------------------------------------------

def resample_img(
    itk_image: sitk.Image,
    out_spacing: list = None,
    is_label: bool = False,
    out_size: list = None,
    out_origin: list = None,
    out_direction: list = None,
) -> sitk.Image:
    """Resample ảnh về target spacing. BSpline cho CT, NearestNeighbor cho nhãn."""
    if out_spacing is None:
        out_spacing = [2.0, 2.0, 2.0]

    original_spacing = itk_image.GetSpacing()
    original_size    = itk_image.GetSize()

    if not out_size:
        out_size = [
            int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
            int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
            int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2]))),
        ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(out_direction if out_direction else itk_image.GetDirection())
    resample.SetOutputOrigin(out_origin if out_origin else itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)

    return resample.Execute(itk_image)


def CropPancreasROI(
    image: sitk.Image,
    low_res_segmentation: sitk.Image,
    margins: list,
) -> tuple[sitk.Image, dict]:
    """
    Crop vùng ROI xung quanh tụy trong ảnh full-resolution.
    low_res_segmentation phải là binary mask (0/1). margins tính bằng mm.
    """
    pancreas_mask_np = sitk.GetArrayFromImage(low_res_segmentation)
    assert len(np.unique(pancreas_mask_np)) == 2, (
        "Segmentation mask phải là binary. Dùng --pancreas-label để chỉ định nhãn tụy."
    )

    nz = np.nonzero(pancreas_mask_np)
    min_x, min_y, min_z = int(nz[2].min()), int(nz[1].min()), int(nz[0].min())
    max_x, max_y, max_z = int(nz[2].max()), int(nz[1].max()), int(nz[0].max())

    start_phys  = low_res_segmentation.TransformIndexToPhysicalPoint((min_x, min_y, min_z))
    finish_phys = low_res_segmentation.TransformIndexToPhysicalPoint((max_x, max_y, max_z))
    start  = image.TransformPhysicalPointToIndex(start_phys)
    finish = image.TransformPhysicalPointToIndex(finish_phys)

    spacing = image.GetSpacing()
    size    = image.GetSize()
    mv = [int(margins[i] / spacing[i]) for i in range(3)]

    xs = max(0,       start[0]  - mv[0]);  xf = min(size[0], finish[0] + mv[0])
    ys = max(0,       start[1]  - mv[1]);  yf = min(size[1], finish[1] + mv[1])
    zs = max(0,       start[2]  - mv[2]);  zf = min(size[2], finish[2] + mv[2])

    cropped = image[xs:xf, ys:yf, zs:zf]
    coords  = {"x": (xs, xf), "y": (ys, yf), "z": (zs, zf)}
    return cropped, coords


# ---------------------------------------------------------------------------
# Pipeline chính
# ---------------------------------------------------------------------------

def preprocess(
    input_path: Path,
    output_path: Path,
    seg_path: Path | None = None,
    spacing: list = None,
    is_label: bool = False,
    margins: list = None,
    pancreas_label: int = 1,
) -> None:
    """
    Nếu seg_path=None  → lowres: resample về target spacing.
    Nếu seg_path!=None → highres: crop ROI tụy dựa trên segmentation mask.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input không tồn tại: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = sitk.ReadImage(str(input_path))
    logger.info(f"Input : {input_path.name}  spacing={list(image.GetSpacing())}  size={list(image.GetSize())}")

    if seg_path is None:
        # ── Pipeline lowres ───────────────────────────────────────────────
        target = spacing or LOWRES_SPACING
        result = resample_img(image, out_spacing=target, is_label=is_label)
        logger.info(f"[lowres] spacing {list(image.GetSpacing())} → {list(result.GetSpacing())}")
        logger.info(f"[lowres] size    {list(image.GetSize())} → {list(result.GetSize())}")

    else:
        # ── Pipeline highres ──────────────────────────────────────────────
        if not seg_path.exists():
            raise FileNotFoundError(f"Seg mask không tồn tại: {seg_path}")

        seg = sitk.ReadImage(str(seg_path))
        seg_np = sitk.GetArrayFromImage(seg)

        # Binarize nếu cần
        unique = np.unique(seg_np)
        if len(unique) != 2 or pancreas_label not in unique:
            logger.info(f"  Binarize: label {pancreas_label} → 1  (có các nhãn: {unique})")
            binary_np = (seg_np == pancreas_label).astype(seg_np.dtype)
            seg_bin = sitk.GetImageFromArray(binary_np)
            seg_bin.CopyInformation(seg)
        else:
            seg_bin = seg

        mg = margins or DEFAULT_MARGINS
        result, coords = CropPancreasROI(image, seg_bin, mg)
        logger.info(f"[highres] margins={mg} mm")
        logger.info(f"[highres] crop x={coords['x']} y={coords['y']} z={coords['z']}")
        logger.info(f"[highres] size {list(image.GetSize())} → {list(result.GetSize())}")

    sitk.WriteImage(result, str(output_path))
    logger.info(f"Output: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess CT .nii.gz: lowres resample hoặc highres crop ROI tụy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input",  type=str, help="Input CT (.nii.gz)")
    parser.add_argument("output", type=str, help="Output (.nii.gz)")
    parser.add_argument(
        "--seg", type=str, default=None, metavar="SEG",
        help="Pancreas segmentation mask (.nii.gz). Nếu cung cấp → chạy highres crop.",
    )
    parser.add_argument(
        "--spacing", type=float, nargs=3, default=LOWRES_SPACING,
        metavar=("SX", "SY", "SZ"),
        help="Target spacing mm cho lowres (x y z)",
    )
    parser.add_argument(
        "--margins", type=float, nargs=3, default=DEFAULT_MARGINS,
        metavar=("MX", "MY", "MZ"),
        help="Margin mm quanh bounding box tụy cho highres (x y z)",
    )
    parser.add_argument(
        "--pancreas-label", type=int, default=1,
        help="Nhãn tụy trong seg mask (PANORAMA ground truth = 4)",
    )
    parser.add_argument(
        "--label", action="store_true",
        help="Input là ảnh nhãn → dùng NearestNeighbor interpolation",
    )

    args = parser.parse_args(argv)

    preprocess(
        input_path=Path(args.input),
        output_path=Path(args.output),
        seg_path=Path(args.seg) if args.seg else None,
        spacing=args.spacing,
        is_label=args.label,
        margins=args.margins,
        pancreas_label=args.pancreas_label,
    )


if __name__ == "__main__":
    main()
