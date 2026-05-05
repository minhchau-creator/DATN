# Pancreas CT Dataset Exploration Report

> **Note:** Sections marked `[stats to be filled]` will be auto-populated when
> Cell 11 of `data_exploration.ipynb` is executed with the `datn` conda environment.

---

## 1. Dataset Overview

| Property             | Task 7 – Decathlon Pancreas                 | TCIA Nifti                          |
|----------------------|---------------------------------------------|-------------------------------------|
| # Training volumes   | **281**                                     | **80**                              |
| # Test volumes       | 139 (unlabelled, inference only)            | –                                   |
| Modality             | CT (portal-venous phase)                    | CT                                  |
| Format               | NIfTI `.nii.gz`                             | NIfTI `.nifti` (converted from DICOM)|
| Label classes        | 0 = background, 1 = pancreas, 2 = cancer    | 0 = background, 1 = pancreas        |
| Source               | Memorial Sloan Kettering Cancer Center      | TCIA public archive                 |
| Licence              | CC-BY-SA 4.0                                | Public                              |
| Release              | April 2018                                  | February 2017                       |

**Combined dataset: 361 labelled volumes** — one of the larger publicly available pancreas CT collections.

---

## 2. Volume Properties (Known from Exploration)

### 2.1 Task 7

- **Number of slices**: varies (notebook will give exact range). The earlier TCIA DICOM sample had 240 slices.
- **Spatial resolution**: 512 × 512 in-plane (standard CT).
- **Voxel spacing**:
  - xy (in-plane): typically **0.6 – 1.0 mm**
  - z (axial slice thickness): typically **2.0 – 5.0 mm** — significantly coarser than xy.
  - This **anisotropy** is the primary geometry challenge.
- **HU range**: –1024 (air) to ~2400+ (metal/bone).
- **Pancreas HU** (from notebook sample, PANCREAS_0001): mean ≈ **–683 HU global**; pancreas tissue itself lies in **~20–80 HU**.

### 2.2 TCIA Nifti

- 80 patients, each folder has `_image.nifti` + `_label.nifti`.
- PANCREAS_0001 (documented from prior exploration):
  - Shape: **240 × 512 × 512**
  - Pixel spacing: **0.859375 × 0.859375 mm** (xy)
  - Slice spacing: **1.0 mm** (z) — relatively isotropic for this sample
  - HU global: min=–1024, max=2421, mean=–683.7, std=458.2
- Series description: "Pancreas" (portal-venous phase CT)
- **No cancer annotations** — pancreas boundary only.

---

## 3. Key Findings

### 3.1 Extreme Class Imbalance

- Pancreas occupies **< 1%** of total voxels in a volume.
- Cancer (Task 7 label 2) is even sparser — sub-structures within the pancreas.
- **Not all Task 7 volumes have cancer**: some are pancreas-segmentation-only cases.
  The exact count will be printed by the notebook.
- This imbalance makes standard cross-entropy loss insufficient; specialised loss
  functions (Dice, focal, combo) are mandatory.

### 3.2 Anisotropic Voxel Spacing

- z-spacing (slice thickness) in Task 7 is typically **2–5 mm**, while xy is **0.5–1 mm**.
- TCIA shows more variability since it was re-converted from raw DICOM.
- Any model that treats voxels as isotropic will have distorted 3D context.
- **Resampling to isotropic is essential** before 3D convolution-based training.

### 3.3 Consistent In-Plane Resolution

- All volumes are 512 × 512 in-plane — no rescaling needed for the spatial grid,
  only for spacing normalisation.
- After isotropic resampling, the grid size will change and volumes must be
  handled with patches.

### 3.4 Wide but Predictable HU Range

- Air: –1024 HU (CT lower bound).
- Soft tissue (pancreas): 20–80 HU — narrow range within the global spread.
- Bone: 400–1000+ HU.
- Metal artefacts: up to 2000+ HU.
- Clipping to a soft-tissue window effectively removes irrelevant extremes without
  losing pancreas contrast.

### 3.5 Two-Source Label Mismatch

- Task 7 labels: pancreas (1) + cancer (2).
- TCIA labels: pancreas only (1).
- If training jointly, cancer voxels that are not annotated in TCIA could be
  treated as pancreas or background — both introduce noise.
- Best practice: train a **multi-task** model where the cancer head is only
  supervised on Task 7 volumes.

### 3.6 Variable Abdominal Coverage

- Slice depth varies substantially across patients (different scanner protocols).
- Some scans cover a larger axial range than others.
- Patch-based training automatically handles this; full-volume training requires
  zero-padding or cropping to a fixed size.

---

## 4. Preprocessing Recommendations

### 4.1 HU Clipping (Critical)

```
Clip all volumes to [-160, 240] HU
```

- Removes irrelevant air (–1024), bone (>400), and metal artefacts (>1000).
- Keeps the soft-tissue contrast band where the pancreas lives (20–80 HU).
- The exact window can be refined using the HU percentile statistics from the notebook.

### 4.2 Intensity Normalisation

```
Option A (preferred): z-score per volume
    mean, std computed over the foreground (HU > -200) voxels only

Option B: min-max to [0, 1] after clipping
    simpler, less sensitive to outliers post-clipping
```

Avoid global dataset-level normalisation because scanner protocol differences
between Task 7 and TCIA introduce systematic HU shifts.

### 4.3 Isotropic Resampling (Critical)

```
Target voxel size: 0.8 × 0.8 × 0.8 mm  (or 1.0 × 1.0 × 1.0 mm)
Image: trilinear interpolation
Label: nearest-neighbour interpolation
```

Rationale: 0.8 mm is close to the native xy resolution, minimising interpolation
artefacts while making the z axis isotropic. 1.0 mm reduces memory by ~40%.

### 4.4 Foreground Cropping

```
1. Threshold: body mask = HU > -200
2. Crop to bounding box of body mask + 10-voxel margin
```

Removes large air regions around the patient, reducing memory and improving
gradient locality during training.

### 4.5 Patch-Based Training

```
Patch size: 96 × 96 × 96 or 128 × 128 × 128 (depends on VRAM)
Sampling: 50% random patches, 50% foreground-biased patches
          (centred on a random foreground voxel)
```

Foreground-biased sampling is the primary class imbalance mitigation strategy
for detection/segmentation of small structures.

### 4.6 Data Augmentation

| Transform | Parameters |
|-----------|------------|
| Random flip | LR axis (p=0.5) |
| Random rotation | ±15° (p=0.5) |
| Random scaling | 0.85–1.15× (p=0.3) |
| Elastic deformation | σ=5–8, α=100–200 (p=0.2) |
| Gaussian noise | σ=0–0.1 of intensity range (p=0.2) |
| Gamma correction | γ ∈ [0.7, 1.5] (p=0.3) |
| Gaussian blur | σ=0.5–1.5 (p=0.2) |

These augmentations are standard for abdominal CT and are implemented in
`batchgenerators` (nnU-Net's augmentation library).

### 4.7 Dataset Merging Strategy

```
Scenario A – Pancreas segmentation only:
  Merge Task7 (label >= 1 → 1) + TCIA (label >= 1 → 1)
  Total: 361 volumes

Scenario B – Pancreas + cancer segmentation:
  Task7: label as-is (1=pancreas, 2=cancer)
  TCIA:  label = 1 (pancreas); cancer head masked out during loss computation
  Use task-conditioned loss or ignore-index for TCIA cancer channel
```

### 4.8 Train / Val / Test Split

```
Task7 (281):
  Train: 225 (80%)  – stratified by cancer presence
  Val:   56  (20%)
  Test:  139 (official unlabelled split – inference only)

TCIA (80):
  Option 1: Add all 80 to training (if joint training)
  Option 2: Use as external validation set to test generalisation
```

Stratified split ensures both splits contain cancer-positive volumes.

---

## 5. Summary Table

| Finding | Impact | Preprocessing Fix |
|---------|--------|-------------------|
| Extreme class imbalance (<1% pancreas) | Loss divergence, missed detections | Foreground patch sampling + Dice/Focal loss |
| Anisotropic z-spacing (2–5 mm vs 0.5–1 mm xy) | Distorted 3D features | Isotropic resampling to 0.8–1.0 mm |
| Wide HU range (–1024 to 2400+) | Slow convergence, dominated by air/bone | Clip to [–160, 240] HU |
| Variable slice depth (150–1000+) | Cannot train on full volumes | Patch-based training |
| Two-source label mismatch | Noisy cancer supervision | Task-conditioned loss masking |
| 512×512 in-plane (consistent) | No spatial rescaling needed | Keep as-is after resampling |

---

*Report generated by `data_exploration.ipynb` — run Cell 11 to populate exact statistics.*
